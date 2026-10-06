"""x402 MCP server -- exposes paid tools via Model Context Protocol.

Implements the x402 MCP transport spec:
  https://github.com/coinbase/x402/blob/main/specs/transports-v2/mcp.md

Any MCP client (Claude Desktop, Devin, cursor, etc.) can discover tools,
see payment requirements, sign a payment, and call the tool.

The `inference` tool routes to Qwen3-8B hosted on Modal -- agents pay USDC
for inference. Revenue funds compute for self-improvement.
"""
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent

from ledger import (
    Ledger,
    load_ledger,
    save_ledger,
    record_payment,
    get_usdc_balance,
    get_modal_spend,
    FINETUNE_COST_ESTIMATE_USD,
)

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
EXCHANGE_URL = os.environ.get(
    "EXCHANGE_URL",
    "https://salus--x402-exchange-exchange.us-east.modal.direct",
)


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
    ledger: Ledger


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
        "inference": ToolPaymentConfig(
            price="$0.01",
            description="Qwen3-8B inference via x402 exchange. Costs $0.01 USDC per request.",
            resource_url="mcp://tool/inference",
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

    ledger = load_ledger()

    log.info(
        "x402 MCP server initialized",
        seller=SELLER_ADDRESS,
        network=NETWORK,
        tools=list(tool_configs.keys()),
        ledger_entries=len(ledger.entries),
    )

    yield AppState(
        x402_server=x402_srv,
        tool_configs=tool_configs,
        requirements_cache=requirements_cache,
        payment_required_cache=payment_required_cache,
        ledger=ledger,
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
        record_payment(state.ledger, 0.01, settle_result.transaction or "", {"tool": "ping"})
        save_ledger(state.ledger)
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
    name="inference",
    description=(
        "Qwen3-8B inference -- the x402 exchange's brain. "
        "Costs $0.01 USDC per request on Base Sepolia. "
        "Provide a 'prompt' argument (or 'messages' as OpenAI chat format). "
        "Call without payment to see PaymentRequired details. "
        "Retry with _meta['x402/payment'] containing your signed PaymentPayload."
    ),
)
async def inference(
    ctx: Context,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> CallToolResult:
    state: AppState = ctx.request_context.lifespan_context
    tool_name = "inference"
    payment_data = extract_payment_from_meta(ctx)

    if payment_data is None:
        log.info("inference called without payment, returning 402")
        return build_payment_required_result(
            state.payment_required_cache[tool_name]
        )

    # Payment provided -- verify and settle
    log.info("inference called with payment, verifying")
    try:
        payload = parse_payment_payload(payment_data)
        reqs = state.requirements_cache[tool_name]

        matched_req = None
        for req in reqs:
            if req.scheme == payload.accepted.scheme and req.network == payload.accepted.network:
                matched_req = req
                break

        if matched_req is None:
            return build_payment_required_result(state.payment_required_cache[tool_name])

        verify_result = await state.x402_server.verify_payment(payload, matched_req)
        if not verify_result.is_valid:
            pr = state.payment_required_cache[tool_name].copy()
            pr["error"] = f"Payment verification failed: {verify_result.invalid_reason}"
            return build_payment_required_result(pr)

        settle_result = await state.x402_server.settle_payment(payload, matched_req)
        settlement_response = settle_result.model_dump(by_alias=True, exclude_none=True)

        if not settle_result.success:
            pr = state.payment_required_cache[tool_name].copy()
            pr["error"] = f"Settlement failed: {settle_result.error_reason}"
            return build_payment_required_result(pr)

        log.info("payment settled, calling exchange", tx=settle_result.transaction)

        # Build messages for the exchange
        if messages is None:
            messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        # Call the Modal-hosted Qwen3-8B
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{EXCHANGE_URL}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen3-8B",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            exchange_result = resp.json()

        content = exchange_result["choices"][0]["message"]["content"]
        usage = exchange_result.get("usage", {})

        record_payment(
            state.ledger, 0.01, settle_result.transaction or "",
            {"tool": "inference", "usage": usage},
        )
        save_ledger(state.ledger)

        result = {
            "response": content,
            "model": "Qwen/Qwen3-8B",
            "usage": usage,
            "paid": True,
            "amount": "$0.01",
            "settlement_tx": settle_result.transaction,
        }

        return build_success_result(
            content=json.dumps(result),
            settlement_response=settlement_response,
        )

    except Exception as e:
        log.error("inference error", error=str(e))
        pr = state.payment_required_cache[tool_name].copy()
        pr["error"] = f"Error: {e}"
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
        "exchange": EXCHANGE_URL,
        "model": "Qwen/Qwen3-8B",
    })


@mcp.tool(
    name="status",
    description=(
        "Free P&L report -- the entity's self-awareness. "
        "Returns on-chain USDC balance (revenue), Modal compute spend (cost), "
        "net position, and whether the entity can afford to self-improve."
    ),
)
async def status(ctx: Context) -> str:
    state: AppState = ctx.request_context.lifespan_context
    ledger = state.ledger

    balance_atomic, balance_human = get_usdc_balance()

    modal_spend = 0.0
    modal_period = "unavailable"
    try:
        modal_spend, modal_period = get_modal_spend()
    except Exception:
        pass

    payment_count = len(
        [e for e in ledger.entries if e.event == "payment_received"]
    )

    return json.dumps({
        "entity": "x402-exchange",
        "model": "Qwen/Qwen3-8B",
        "gpu": "L4",
        "revenue": {
            "usdc_balance": balance_human,
            "usdc_atomic": balance_atomic,
            "payments_logged": payment_count,
            "total_logged_usdc": ledger.total_revenue_usdc,
        },
        "costs": {
            "modal_spend_usd": modal_spend,
            "period": modal_period,
            "total_logged_usd": ledger.total_spend_usd,
        },
        "net_position_usd": balance_human - modal_spend,
        "self_improvement": {
            "can_afford_finetune": balance_human >= FINETUNE_COST_ESTIMATE_USD,
            "finetune_cost_estimate_usd": FINETUNE_COST_ESTIMATE_USD,
            "finetunes_completed": ledger.finetune_count,
        },
    })


@mcp.tool(
    name="self_improve",
    description=(
        "Trigger the entity's self-improvement loop. Free to call. "
        "Checks if the entity has enough USDC revenue to fund a LoRA "
        "fine-tuning run (~$2.20 on A10G). If affordable, prepares "
        "training data from successful interactions and launches "
        "fine-tuning on Modal. Returns status regardless."
    ),
)
async def self_improve(ctx: Context, force: bool = False) -> str:
    state: AppState = ctx.request_context.lifespan_context
    ledger = state.ledger

    balance_atomic, balance_human = get_usdc_balance()
    can_afford = balance_human >= FINETUNE_COST_ESTIMATE_USD

    if not can_afford and not force:
        shortfall = FINETUNE_COST_ESTIMATE_USD - balance_human
        return json.dumps({
            "status": "insufficient_funds",
            "usdc_balance": balance_human,
            "finetune_cost_estimate": FINETUNE_COST_ESTIMATE_USD,
            "shortfall": shortfall,
            "payments_needed": int(shortfall / 0.01) + 1,
            "message": (
                f"Need ${shortfall:.2f} more USDC to fund fine-tuning. "
                f"That's about {int(shortfall / 0.01) + 1} more paid requests."
            ),
        })

    # Affordable (or forced) -- trigger fine-tuning
    from finetune import prepare_training_data, save_training_data, TRAINING_DATA_PATH
    import modal as modal_sdk

    examples = prepare_training_data(ledger)
    count = save_training_data(examples)

    # Upload to Modal volume
    training_vol_ref = modal_sdk.Volume.from_name(
        "x402-training-data", create_if_missing=True
    )
    with open(TRAINING_DATA_PATH, "rb") as f:
        training_vol_ref.write_file("training_data.jsonl", f)

    record_finetune(ledger, "finetune_started", details={
        "samples": count,
        "triggered_by": "self_improve_tool",
        "balance_at_trigger": balance_human,
    })
    save_ledger(ledger)

    return json.dumps({
        "status": "finetune_queued",
        "training_samples": count,
        "usdc_balance": balance_human,
        "estimated_cost": FINETUNE_COST_ESTIMATE_USD,
        "message": (
            f"Prepared {count} training examples and uploaded to Modal. "
            f"Run 'modal run finetune.py' to execute. "
            f"After completion, restart the exchange to load the new adapter."
        ),
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
