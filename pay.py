#!/usr/bin/env python3
"""x402 phone client -- pay the exchange from Termux in one command.

Setup (one-time):
    pip install "x402[evm]" httpx

Usage:
    python pay.py "What is consciousness?"
    python pay.py                          # interactive mode
    echo "Explain AGI" | python pay.py     # pipe mode

The private key below is pre-funded with testnet USDC on Base Sepolia.
This is NOT real money -- it's a test wallet for the x402 exchange.
"""
import asyncio
import json
import os
import sys
import time

import httpx
from eth_account import Account
from x402 import x402Client, parse_payment_required
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client

# ---- config (pre-funded testnet wallet) ----
PRIVATE_KEY = os.environ.get(
    "X402_PRIVATE_KEY",
    "c09394e73608ca610968e1cbf8c9158d38648b7ebdef4c1e8fcf1b5af5f22b4e",
)
GATEWAY = os.environ.get(
    "X402_GATEWAY",
    "https://salus--x402-gateway-gateway.us-east.modal.direct",
)
MCP_URL = f"{GATEWAY}/mcp"


async def call_mcp(session_id: str | None, method: str, params: dict, client: httpx.AsyncClient):
    """Send a JSON-RPC request to the MCP endpoint with retry for cold starts."""
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

    for attempt in range(5):
        try:
            resp = await client.post(MCP_URL, json=body, headers=headers, timeout=120)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"  Gateway not ready, retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            raise

        sid = resp.headers.get("Mcp-Session-Id", session_id)
        text = resp.text

        # Parse SSE response
        for line in text.splitlines():
            if line.startswith("data: "):
                return sid, json.loads(line[6:])

        # No data -- gateway may be cold-starting
        if attempt < 4:
            wait = 10 * (attempt + 1)
            print(f"  No response (gateway cold-starting), retrying in {wait}s...")
            await asyncio.sleep(wait)
            # Reset session for fresh init
            if method == "initialize":
                session_id = None
                headers.pop("Mcp-Session-Id", None)
        else:
            return sid, None

    return session_id, None


async def ask(prompt: str):
    """Send a prompt to the x402 exchange, pay, and print the response."""
    account = Account.from_key(PRIVATE_KEY)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    register_exact_evm_client(x402_client, signer)

    print(f"Wallet: {account.address}")
    print(f"Gateway: {GATEWAY}")
    print(f"Prompt: {prompt}")
    print()

    async with httpx.AsyncClient() as http:
        # Initialize MCP session
        sid, init = await call_mcp(None, "initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "x402-phone", "version": "1.0"},
        }, http)

        # Call inference without payment -- get 402 challenge
        print("Requesting inference (no payment)...")
        sid, result = await call_mcp(sid, "tools/call", {
            "name": "inference",
            "arguments": {"prompt": prompt, "max_tokens": 512},
        }, http)

        if not result or "result" not in result:
            print(f"ERROR: Unexpected response: {result}")
            return

        tool_result = result["result"]
        content_text = tool_result.get("content", [{}])[0].get("text", "")

        # Extract PaymentRequired
        try:
            pr_data = json.loads(content_text)
        except json.JSONDecodeError:
            print(f"Response: {content_text}")
            return

        if "x402Version" not in pr_data:
            # Already paid or free -- just print
            if "response" in pr_data:
                print(f"Response: {pr_data['response']}")
            else:
                print(f"Result: {json.dumps(pr_data, indent=2)}")
            return

        amount_atomic = pr_data["accepts"][0]["amount"]
        amount_usd = int(amount_atomic) / 1_000_000
        print(f"Payment required: ${amount_usd:.2f} USDC")

        # Sign payment
        print("Signing EIP-3009 authorization...")
        t0 = time.time()
        payment_required = parse_payment_required(pr_data)
        payment_payload = await x402_client.create_payment_payload(payment_required)
        payload_dict = payment_payload.model_dump(by_alias=True, exclude_none=True)
        print(f"Signed in {time.time() - t0:.1f}s")

        # Retry with payment
        print("Sending paid request...")
        sid, paid_result = await call_mcp(sid, "tools/call", {
            "name": "inference",
            "arguments": {"prompt": prompt, "max_tokens": 512},
            "_meta": {"x402/payment": payload_dict},
        }, http)

        if not paid_result or "result" not in paid_result:
            print(f"ERROR: {paid_result}")
            return

        paid_content = paid_result["result"].get("content", [{}])[0].get("text", "")
        try:
            data = json.loads(paid_content)
            print()
            print("=" * 60)
            print(data.get("response", paid_content))
            print("=" * 60)
            print()
            if data.get("settlement_tx"):
                print(f"Tx: https://sepolia.basescan.org/tx/{data['settlement_tx']}")
            if data.get("usage"):
                u = data["usage"]
                print(f"Tokens: {u.get('prompt_tokens', '?')} in / {u.get('completion_tokens', '?')} out")
            print(f"Cost: $0.01 USDC")
        except json.JSONDecodeError:
            print(paid_content)


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        print("x402 Exchange -- Qwen3-8B ($0.01/request)")
        print("Type your question (Ctrl+D to send):")
        print()
        try:
            prompt = sys.stdin.read().strip()
        except KeyboardInterrupt:
            return

    if not prompt:
        print("No prompt provided.")
        return

    asyncio.run(ask(prompt))


if __name__ == "__main__":
    main()
