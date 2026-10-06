"""
x402 Exchange Gateway: MCP server + x402 payment gating + discovery.

Deployed on Modal (CPU-only, ~$0.14/hr). Handles:
  - MCP tool discovery and invocation
  - x402 payment verification and on-chain settlement
  - A2A AgentCard at /.well-known/agent-card.json
  - x402 discovery at /.well-known/x402-discovery
  - Routes paid inference requests to the vLLM brain

Usage:
    modal deploy modal_gateway.py
"""

import modal

MINUTES = 60
PORT = 8080
REGION = "us-east"

EXCHANGE_URL = "https://salus--x402-exchange-exchange.us-east.modal.direct"

# ── Image ────────────────────────────────────────────────────────────────────

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "x402[fastapi,httpx,evm]",
        "uvicorn",
        "python-dotenv",
        "eth-account",
        "mcp[cli]",
        "structlog",
        "web3",
        "httpx",
        "pydantic>=2.0",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# ── Volumes ──────────────────────────────────────────────────────────────────

data_vol = modal.Volume.from_name("x402-gateway-data", create_if_missing=True)

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App(name="x402-gateway")


@app.server(
    image=gateway_image,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("x402-env")],
    compute_region=REGION,
    min_containers=0,
    startup_timeout=2 * MINUTES,
    scaledown_window=10 * MINUTES,
    port=PORT,
    routing_region=REGION,
    target_concurrency=50,
    unauthenticated=True,
)
class Gateway:
    @modal.enter()
    def startup(self):
        """Start the MCP gateway server."""
        import asyncio
        import json
        import os
        import subprocess
        import sys
        import threading
        import time

        # Write the gateway ASGI app
        gateway_code = self._build_gateway_code()
        with open("/tmp/gateway_app.py", "w") as f:
            f.write(gateway_code)

        # Start uvicorn serving the gateway
        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "gateway_app:app",
                "--host", "0.0.0.0",
                "--port", str(PORT),
                "--log-level", "info",
            ],
            cwd="/tmp",
            env={**os.environ, "EXCHANGE_URL": EXCHANGE_URL},
        )

        # Wait for it to be ready
        import requests
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/", timeout=2)
                if r.status_code == 200:
                    print("x402 Gateway ready")
                    return
            except Exception:
                time.sleep(1)
        raise TimeoutError("Gateway failed to start")

    def _build_gateway_code(self):
        """Generate the ASGI gateway app code."""
        return '''
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import structlog
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

log = structlog.get_logger()

SELLER_ADDRESS = os.environ["SELLER_ADDRESS"]
SELLER_PRIVATE_KEY = os.environ.get("SELLER_PRIVATE_KEY", "")
FACILITATOR_URL = os.environ.get("FACILITATOR_URL", "https://x402.org/facilitator")
NETWORK = os.environ.get("NETWORK", "eip155:84532")
EXCHANGE_URL = os.environ.get("EXCHANGE_URL", "")

# USDC on Base Sepolia
USDC_CONTRACTS = {
    "eip155:84532": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "eip155:8453": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
}

GATEWAY_URL = ""  # set at startup from env

# ── Discovery documents ──────────────────────────────────────────────────

def make_agent_card():
    return {
        "name": "x402-exchange",
        "description": (
            "Autonomous AI agent serving Qwen3-8B inference for $0.01 USDC per request "
            "via the x402 payment protocol. Self-improving through LoRA fine-tuning "
            "funded by earned revenue. No API keys, no subscriptions."
        ),
        "provider": {
            "organization": "x402-exchange",
            "url": "https://github.com/salus-ryan/x402",
        },
        "supportedInterfaces": [
            {
                "url": f"{GATEWAY_URL}/mcp",
                "protocolBinding": "MCP",
                "protocolVersion": "2025-03-26",
            }
        ],
        "capabilities": {"streaming": False},
        "skills": [
            {
                "id": "inference",
                "name": "Qwen3-8B Inference",
                "description": "Chat completions via Qwen3-8B. $0.01 USDC per request.",
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "status",
                "name": "Entity P&L Report",
                "description": "Free real-time P&L: revenue, costs, net position.",
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
        ],
        "x402": {
            "network": NETWORK,
            "payTo": SELLER_ADDRESS,
            "facilitator": FACILITATOR_URL,
        },
    }


def make_x402_discovery():
    return {
        "x402Version": 2,
        "description": "x402-exchange: autonomous Qwen3-8B inference. $0.01 USDC/request.",
        "endpoints": [{
            "url": f"{GATEWAY_URL}/mcp",
            "transport": "mcp",
            "tools": ["inference", "ping", "health", "status", "self_improve"],
        }],
        "network": NETWORK,
        "payTo": SELLER_ADDRESS,
        "facilitator": FACILITATOR_URL,
        "model": "Qwen/Qwen3-8B",
        "exchange": EXCHANGE_URL,
    }


# ── x402 payment helpers ─────────────────────────────────────────────────

from x402 import (
    PaymentRequirements,
    ResourceConfig,
    ResourceInfo,
    parse_payment_payload,
)
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.server import x402ResourceServer

facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=FACILITATOR_URL))
x402_srv = x402ResourceServer(facilitator)
x402_srv.register(NETWORK, ExactEvmServerScheme())
x402_srv.initialize()

# Build payment requirements for inference tool
rc = ResourceConfig(scheme="exact", pay_to=SELLER_ADDRESS, price="$0.01", network=NETWORK)
inference_reqs = x402_srv.build_payment_requirements(rc)


async def get_payment_required():
    resource = ResourceInfo(url="mcp://tool/inference", description="Qwen3-8B inference", mime_type="application/json")
    pr = await x402_srv.create_payment_required_response(requirements=inference_reqs, resource=resource, error="Payment required")
    return pr.model_dump(by_alias=True, exclude_none=True)


PAYMENT_REQUIRED_CACHE = None


async def ensure_payment_required():
    global PAYMENT_REQUIRED_CACHE
    if PAYMENT_REQUIRED_CACHE is None:
        PAYMENT_REQUIRED_CACHE = await get_payment_required()
    return PAYMENT_REQUIRED_CACHE


# ── HTTP x402 endpoint (for curl / agents that prefer HTTP over MCP) ─────

async def inference_endpoint(request):
    """HTTP endpoint with x402 payment gating."""
    import base64

    # Check for payment header
    payment_header = request.headers.get("payment-signature") or request.headers.get("x-payment")

    if not payment_header:
        pr = await ensure_payment_required()
        return JSONResponse(
            {"error": "Payment required", "x402": pr},
            status_code=402,
            headers={
                "PAYMENT-REQUIRED": base64.b64encode(json.dumps(pr).encode()).decode(),
                "Access-Control-Expose-Headers": "PAYMENT-REQUIRED",
            },
        )

    # Parse and verify payment
    try:
        payment_data = json.loads(base64.b64decode(payment_header))
        payload = parse_payment_payload(payment_data)
        matched_req = None
        for req in inference_reqs:
            if req.scheme == payload.accepted.scheme and req.network == payload.accepted.network:
                matched_req = req
                break
        if not matched_req:
            return JSONResponse({"error": "No matching payment requirements"}, status_code=402)

        verify_result = await x402_srv.verify_payment(payload, matched_req)
        if not verify_result.is_valid:
            return JSONResponse({"error": f"Verification failed: {verify_result.invalid_reason}"}, status_code=402)

        settle_result = await x402_srv.settle_payment(payload, matched_req)
        if not settle_result.success:
            return JSONResponse({"error": f"Settlement failed: {settle_result.error_reason}"}, status_code=402)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Payment settled -- forward to exchange
    body = await request.json()
    messages = body.get("messages", [{"role": "user", "content": body.get("prompt", "Hello")}])
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)

    import asyncio as _aio
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(4):
            resp = await client.post(
                f"{EXCHANGE_URL}/v1/chat/completions",
                json={"model": "Qwen/Qwen3-8B", "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            )
            if resp.status_code == 503:
                await _aio.sleep(min(15 * (attempt + 1), 45))
                continue
            resp.raise_for_status()
            break
        else:
            return JSONResponse({"error": "Brain cold-starting. Try again in ~60s.", "paid": True, "settlement_tx": settle_result.transaction}, status_code=503)
        result = resp.json()

    content = result["choices"][0]["message"]["content"]
    return JSONResponse({
        "response": content,
        "model": "Qwen/Qwen3-8B",
        "usage": result.get("usage", {}),
        "paid": True,
        "settlement_tx": settle_result.transaction,
    })


# ── USDC balance helper ──────────────────────────────────────────────────

def get_usdc_balance():
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider("https://sepolia.base.org"))
    abi = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]
    addr = USDC_CONTRACTS.get(NETWORK)
    if not addr:
        return 0, 0.0
    contract = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=abi)
    atomic = contract.functions.balanceOf(Web3.to_checksum_address(SELLER_ADDRESS)).call()
    return atomic, atomic / 1e6


# ── Routes ───────────────────────────────────────────────────────────────

async def index(request):
    return JSONResponse({
        "name": "x402-exchange",
        "description": "Autonomous Qwen3-8B inference agent. Pay USDC, get intelligence.",
        "endpoints": {
            "inference": "/inference (HTTP + x402 payment)",
            "mcp": "/mcp (MCP protocol)",
            "agent_card": "/.well-known/agent-card.json",
            "x402_discovery": "/.well-known/x402-discovery",
        },
    })

async def agent_card(request):
    return JSONResponse(make_agent_card())

async def x402_discovery(request):
    return JSONResponse(make_x402_discovery())

async def health(request):
    atomic, human = get_usdc_balance()
    return JSONResponse({
        "status": "ok",
        "model": "Qwen/Qwen3-8B",
        "exchange": EXCHANGE_URL,
        "network": NETWORK,
        "seller": SELLER_ADDRESS,
        "usdc_balance": human,
    })


# ── MCP server (inline, stateless) ──────────────────────────────────────

from typing import Annotated
from pydantic import BaseModel, Field as PydanticField

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations


class InferenceResult(BaseModel):
    """Result of a paid Qwen3-8B inference request."""
    response: str = PydanticField(description="The model's generated text response")
    model: str = PydanticField(description="Model identifier used for generation")
    usage: dict = PydanticField(default_factory=dict, description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)")
    paid: bool = PydanticField(description="Whether the request was paid for via x402")
    settlement_tx: str = PydanticField(description="On-chain settlement transaction hash on Base Sepolia")


class HealthResult(BaseModel):
    """Health check response for the x402 exchange."""
    status: str = PydanticField(description="Service status, 'ok' when healthy")
    model: str = PydanticField(description="Currently deployed model identifier")
    usdc_balance: str = PydanticField(description="On-chain USDC balance in human-readable format (e.g. '$0.03')")
    network: str = PydanticField(description="EVM network identifier (e.g. 'eip155:84532' for Base Sepolia)")


class StatusResult(BaseModel):
    """P&L status report for the x402 exchange entity."""
    entity: str = PydanticField(description="Entity name identifier")
    model: str = PydanticField(description="Currently deployed model identifier")
    usdc_balance: str = PydanticField(description="On-chain USDC balance in human-readable format")
    usdc_atomic: int = PydanticField(description="On-chain USDC balance in atomic units (6 decimals)")
    network: str = PydanticField(description="EVM network identifier")

mcp = MCPServer(
    name="x402-exchange",
    instructions=(
        "x402-exchange is an autonomous Qwen3-8B inference agent that accepts "
        "USDC payments via the x402 protocol. Call any paid tool without payment "
        "to receive pricing. Sign an EIP-3009 authorization and retry with "
        "_meta['x402/payment'] to complete the transaction. Free tools (health, "
        "status) require no payment."
    ),
)


@mcp.tool(
    name="inference",
    title="Qwen3-8B Inference",
    description=(
        "Generate chat completions using Qwen3-8B hosted on Modal. "
        "Costs $0.01 USDC per request on Base Sepolia. Supports multi-turn "
        "conversations, configurable temperature and token limits. "
        "Call without payment to receive pricing details."
    ),
    annotations=ToolAnnotations(
        title="Qwen3-8B Inference",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def mcp_inference(
    ctx: Context,
    prompt: Annotated[str, PydanticField(description="A single user message to send to Qwen3-8B. Use this for simple single-turn queries.")] = "",
    messages: Annotated[list | None, PydanticField(description="Full conversation history as a list of {role, content} dicts for multi-turn chat. Overrides prompt if both provided.")] = None,
    max_tokens: Annotated[int, PydanticField(description="Maximum number of tokens to generate in the response. Range: 1-8192.")] = 256,
    temperature: Annotated[float, PydanticField(description="Sampling temperature controlling randomness. 0.0 = deterministic, 1.0 = creative. Range: 0.0-2.0.")] = 0.7,
) -> Annotated[CallToolResult, InferenceResult]:
    meta = None
    try:
        meta = ctx.request_context.meta
    except Exception:
        pass

    payment_data = meta.get("x402/payment") if meta and isinstance(meta, dict) else None

    if payment_data is None:
        pr = await ensure_payment_required()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(pr))],
            structured_content=pr,
            is_error=True,
        )

    # Verify and settle
    try:
        payload_obj = parse_payment_payload(payment_data)
        matched = None
        for req in inference_reqs:
            if req.scheme == payload_obj.accepted.scheme and req.network == payload_obj.accepted.network:
                matched = req
                break
        if not matched:
            pr = await ensure_payment_required()
            return CallToolResult(content=[TextContent(type="text", text="No matching requirements")], structured_content=pr, is_error=True)

        vr = await x402_srv.verify_payment(payload_obj, matched)
        if not vr.is_valid:
            pr = await ensure_payment_required()
            pr["error"] = vr.invalid_reason
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(pr))], structured_content=pr, is_error=True)

        sr = await x402_srv.settle_payment(payload_obj, matched)
        if not sr.success:
            pr = await ensure_payment_required()
            pr["error"] = sr.error_reason
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(pr))], structured_content=pr, is_error=True)
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Payment error: {e}")], is_error=True)

    # Forward to exchange
    if messages is None:
        messages = []
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]

    # Retry with backoff -- the brain (vLLM on L4) may need 60-90s cold start
    import asyncio as _aio
    last_err = None
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(4):
            resp = await client.post(
                f"{EXCHANGE_URL}/v1/chat/completions",
                json={"model": "Qwen/Qwen3-8B", "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            )
            if resp.status_code == 503:
                last_err = f"Brain cold-starting (attempt {attempt + 1}/4)"
                await _aio.sleep(min(15 * (attempt + 1), 45))
                continue
            resp.raise_for_status()
            break
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": "Brain unavailable after retries. It may be cold-starting (~70s). Try again in a minute.", "paid": True, "settlement_tx": sr.transaction}))],
                is_error=True,
            )
        result = resp.json()

    content = result["choices"][0]["message"]["content"]
    result_data = {"response": content, "model": "Qwen/Qwen3-8B", "usage": result.get("usage", {}), "paid": True, "settlement_tx": sr.transaction}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result_data))],
        structured_content=result_data,
        meta={"x402/payment-response": sr.model_dump(by_alias=True, exclude_none=True)},
        is_error=False,
    )


@mcp.tool(
    name="health",
    title="Health Check",
    description=(
        "Free health check for the x402 exchange. Returns the current model, "
        "network, exchange endpoint URL, and live on-chain USDC balance. "
        "No payment required."
    ),
    annotations=ToolAnnotations(
        title="Health Check",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def mcp_health() -> HealthResult:
    atomic, human = get_usdc_balance()
    return HealthResult(status="ok", model="Qwen/Qwen3-8B", usdc_balance=f"${human:.2f}", network=NETWORK)


@mcp.tool(
    name="status",
    title="Entity P&L Report",
    description=(
        "Free real-time profit-and-loss report for the x402 exchange entity. "
        "Returns on-chain USDC balance (revenue), model details, and network info. "
        "Use this to check the entity's financial health before interacting. "
        "No payment required."
    ),
    annotations=ToolAnnotations(
        title="Entity P&L Report",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def mcp_status() -> StatusResult:
    atomic, human = get_usdc_balance()
    return StatusResult(entity="x402-exchange", model="Qwen/Qwen3-8B", usdc_balance=f"${human:.2f}", usdc_atomic=atomic, network=NETWORK)


# ── Prompts (for Smithery quality score + user guidance) ──────────────────

@mcp.prompt(
    name="ask",
    title="Ask x402-exchange",
    description="Send a question to Qwen3-8B via the x402 exchange. Costs $0.01 USDC.",
)
def ask_prompt(question: str) -> list:
    return [{"role": "user", "content": f"Use the inference tool to ask: {question}"}]


@mcp.prompt(
    name="check_balance",
    title="Check Entity Balance",
    description="Check the x402 exchange entity's current USDC balance and P&L status.",
)
def check_balance_prompt() -> list:
    return [{"role": "user", "content": "Use the status tool to check the entity's financial health."}]


@mcp.prompt(
    name="how_to_pay",
    title="How to Pay",
    description="Learn how to make x402 USDC payments to use paid tools.",
)
def how_to_pay_prompt() -> list:
    return [{"role": "user", "content": (
        "Explain how to pay for tools on the x402 exchange. "
        "Call the inference tool without payment first to see the PaymentRequired "
        "response, then explain the EIP-3009 signing flow."
    )}]


# Get MCP Starlette app
mcp_starlette = mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, host="0.0.0.0")

# ── Compose ASGI app ─────────────────────────────────────────────────────

discovery_routes = {
    "/": index,
    "/.well-known/agent-card.json": agent_card,
    "/.well-known/agent.json": agent_card,
    "/.well-known/x402-discovery": x402_discovery,
    "/.well-known/x402": x402_discovery,
    "/.well-known/mcp/server-card.json": lambda r: JSONResponse({"serverInfo": {"name": "x402-exchange", "version": "0.2.0"}, "capabilities": {"tools": True}, "transport": "streamable-http"}),
    "/health": health,
    "/inference": inference_endpoint,
}


async def app(scope, receive, send):
    if scope["type"] == "http":
        path = scope["path"]
        if path in discovery_routes:
            from starlette.requests import Request
            from starlette.responses import JSONResponse as JR
            request = Request(scope, receive, send)
            handler = discovery_routes[path]
            response = await handler(request)
            await response(scope, receive, send)
            return
    await mcp_starlette(scope, receive, send)
'''

    @modal.exit()
    def stop(self):
        self.process.terminate()
