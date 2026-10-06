"""x402 MCP server -- exposes paid tools via Model Context Protocol.

Implements the x402 MCP transport spec:
  https://github.com/coinbase/x402/blob/main/specs/transports-v2/mcp.md

Any MCP client (Claude Desktop, Devin, cursor, etc.) can discover tools,
see payment requirements, sign a payment, and call the tool.
"""
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent

from x402 import (
    PaymentPayload,
    PaymentRequired,
    PaymentRequirements,
    ResourceConfig,
    ResourceInfo,
    parse_payment_payload,
)
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

load_dotenv()

log = structlog.get_logger()

SELLER_ADDRESS = os.environ["SELLER_ADDRESS"]
FACILITATOR_URL = os.environ.get("FACILITATOR_URL", "https://x402.org/facilitator")
NETWORK: Network = os.environ.get("NETWORK", "eip155:84532")


# ---------------------------------------------------------------------------
# x402 helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolPaymentConfig:
    """Payment config for a single MCP tool."""
    price: str
    description: str
    resource_url: str
    mime_type: str = "application/json"


def build_payment_required_result(
    payment_required: dict[str, Any],
) -> CallToolResult:
    """Build an MCP tool result that signals payment-required per the x402 MCP spec."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payment_required))],
        structured_content=payment_required,
        is_error=True,
    )


def build_success_result(
    content: str,
    settlement_response: dict[str, Any] | None = None,
) -> CallToolResult:
    """Build an MCP tool result with optional settlement receipt in _meta."""
    meta = {}
    if settlement_response:
        meta["x402/payment-response"] = settlement_response
    return CallToolResult(
        content=[TextContent(type="text", text=content)],
        meta=meta if meta else None,
        is_error=False,
    )


def extract_payment_from_meta(ctx: Context) -> dict[str, Any] | None:
    """Extract x402/payment from the MCP request _meta, if present."""
    try:
        meta = ctx.request_context.meta
        if meta and "x402/payment" in meta:
            return meta["x402/payment"]
    except (AttributeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Server setup with lifespan
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    x402_server: x402ResourceServer
    tool_configs: dict[str, ToolPaymentConfig]
    requirements_cache: dict[str, list[PaymentRequirements]]
    payment_required_cache: dict[str, dict[str, Any]]


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[AppState]:
    """Initialize x402 on startup."""
    facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=FACILITATOR_URL))
    x402_srv = x402ResourceServer(facilitator)
    x402_srv.register(NETWORK, ExactEvmServerScheme())
    x402_srv.initialize()

    tool_configs: dict[str, ToolPaymentConfig] = {
        "ping": ToolPaymentConfig(
            price="$0.01",
            description="A paid ping -- proof of life for x402 payments. Costs $0.01 USDC.",
            resource_url="mcp://tool/ping",
        ),
    }

    # Pre-build payment requirements for each tool
    requirements_cache: dict[str, list[PaymentRequirements]] = {}
    payment_required_cache: dict[str, dict[str, Any]] = {}

    for tool_name, config in tool_configs.items():
        rc = ResourceConfig(
            scheme="exact",
            pay_to=SELLER_ADDRESS,
            price=config.price,
            network=NETWORK,
        )
        reqs = x402_srv.build_payment_requirements(rc)
        requirements_cache[tool_name] = reqs

        resource_info = ResourceInfo(
            url=config.resource_url,
            description=config.description,
            mime_type=config.mime_type,
        )
        pr = await x402_srv.create_payment_required_response(
            requirements=reqs,
            resource=resource_info,
            error="Payment required",
        )
        payment_required_cache[tool_name] = pr.model_dump(by_alias=True, exclude_none=True)

    log.info(
        "x402 MCP server initialized",
        seller=SELLER_ADDRESS,
        network=NETWORK,
        tools=list(tool_configs.keys()),
    )

    yield AppState(
        x402_server=x402_srv,
        tool_configs=tool_configs,
        requirements_cache=requirements_cache,
        payment_required_cache=payment_required_cache,
    )


mcp = MCPServer(
    name="x402-mvp",
    instructions=(
        "This server exposes paid tools via the x402 payment protocol. "
        "Tools require USDC payment on Base (Sepolia testnet by default). "
        "When you call a paid tool without payment, you'll receive a "
        "PaymentRequired response with pricing details and a wallet address. "
        "To pay: sign an EIP-3009 transferWithAuthorization for USDC and "
        "retry the tool call with the signed payload in _meta['x402/payment']. "
        "The facilitator handles on-chain settlement -- no gas needed."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="ping",
    description=(
        "A paid ping endpoint -- proof of life for x402 payments. "
        "Costs $0.01 USDC on Base Sepolia. "
        "Call without payment to see PaymentRequired details. "
        "Retry with _meta['x402/payment'] containing your signed PaymentPayload."
    ),
)
async def ping(ctx: Context) -> CallToolResult:
    state: AppState = ctx.request_context.lifespan_context
    tool_name = "ping"
    payment_data = extract_payment_from_meta(ctx)

    if payment_data is None:
        log.info("ping called without payment, returning 402")
        return build_payment_required_result(
            state.payment_required_cache[tool_name]
        )

    # Payment provided -- verify and settle
    log.info("ping called with payment, verifying")
    try:
        payload = parse_payment_payload(payment_data)
        reqs = state.requirements_cache[tool_name]

        # Find matching requirements
        matched_req = None
        for req in reqs:
            if req.scheme == payload.accepted.scheme and req.network == payload.accepted.network:
                matched_req = req
                break

        if matched_req is None:
            log.warning("no matching payment requirements")
            return build_payment_required_result(
                state.payment_required_cache[tool_name]
            )

        # Verify
        verify_result = await state.x402_server.verify_payment(payload, matched_req)
        if not verify_result.is_valid:
            log.warning("payment verification failed", reason=verify_result.invalid_reason)
            pr = state.payment_required_cache[tool_name].copy()
            pr["error"] = f"Payment verification failed: {verify_result.invalid_reason}"
            return build_payment_required_result(pr)

        # Settle
        settle_result = await state.x402_server.settle_payment(payload, matched_req)
        settlement_response = settle_result.model_dump(by_alias=True, exclude_none=True)

        if not settle_result.success:
            log.warning("settlement failed", result=settlement_response)
            pr = state.payment_required_cache[tool_name].copy()
            pr["error"] = f"Settlement failed: {settle_result.error_reason}"
            return build_payment_required_result(pr)

        log.info("payment settled", tx=settle_result.transaction)
        return build_success_result(
            content=json.dumps({"message": "pong", "paid": True, "amount": "$0.01"}),
            settlement_response=settlement_response,
        )

    except Exception as e:
        log.error("payment processing error", error=str(e))
        pr = state.payment_required_cache[tool_name].copy()
        pr["error"] = f"Payment processing error: {e}"
        return build_payment_required_result(pr)


@mcp.tool(
    name="health",
    description="Free health check -- no payment required.",
)
async def health() -> str:
    return json.dumps({
        "status": "ok",
        "seller": SELLER_ADDRESS,
        "network": NETWORK,
        "protocol": "x402",
        "transport": "mcp",
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"

    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    elif transport == "streamable-http":
        asyncio.run(
            mcp.run_streamable_http_async(
                host="0.0.0.0",
                port=4022,
                stateless_http=True,
            )
        )
    else:
        print(f"Unknown transport: {transport}. Use 'stdio' or 'streamable-http'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
