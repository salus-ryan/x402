"""x402 MCP buyer -- discovers tools, pays, and calls them via MCP protocol.

Connects to the MCP server, calls a paid tool, receives PaymentRequired,
signs a USDC payment, and retries with the signed payload in _meta['x402/payment'].
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from eth_account import Account
from mcp import Client

from x402 import PaymentRequired, x402Client, parse_payment_required
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client

load_dotenv()

BUYER_PRIVATE_KEY = os.environ["BUYER_PRIVATE_KEY"]
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:4022/mcp")


async def buy_via_mcp():
    account = Account.from_key(BUYER_PRIVATE_KEY)
    signer = EthAccountSigner(account)

    # Set up x402 client for payment signing
    x402_client = x402Client()
    register_exact_evm_client(x402_client, signer)

    print(f"Buyer wallet: {account.address}")
    print(f"MCP server:   {MCP_SERVER_URL}")
    print()

    async with Client(MCP_SERVER_URL) as client:
        # Step 1: Discover tools
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # Step 2: Call health (free)
        print("--- Calling health (free) ---")
        health_result = await client.call_tool("health")
        print(f"  Result: {health_result.content[0].text}")
        print()

        # Step 3: Call ping without payment (expect 402)
        print("--- Calling ping (no payment) ---")
        ping_result = await client.call_tool("ping")
        print(f"  isError: {ping_result.is_error}")

        if not ping_result.is_error:
            print(f"  Result: {ping_result.content[0].text}")
            return

        # Step 4: Extract PaymentRequired from the response
        # Per x402 MCP spec: check structuredContent first, fall back to content[0].text
        payment_required_data = None
        if ping_result.structured_content:
            sc = ping_result.structured_content
            if isinstance(sc, dict) and "x402Version" in sc and "accepts" in sc:
                payment_required_data = sc

        if payment_required_data is None and ping_result.content:
            text = ping_result.content[0].text
            try:
                parsed = json.loads(text)
                if "x402Version" in parsed and "accepts" in parsed:
                    payment_required_data = parsed
            except (json.JSONDecodeError, AttributeError):
                pass

        if payment_required_data is None:
            print("  ERROR: Could not extract PaymentRequired from response")
            return

        print(f"  Payment required:")
        print(f"    scheme:  {payment_required_data['accepts'][0]['scheme']}")
        print(f"    network: {payment_required_data['accepts'][0]['network']}")
        print(f"    amount:  {payment_required_data['accepts'][0]['amount']} (atomic USDC)")
        print(f"    payTo:   {payment_required_data['accepts'][0]['payTo']}")
        print()

        # Step 5: Sign the payment
        print("--- Signing payment ---")
        payment_required = parse_payment_required(payment_required_data)
        payment_payload = await x402_client.create_payment_payload(payment_required)
        payment_payload_dict = payment_payload.model_dump(by_alias=True, exclude_none=True)
        print(f"  Signed by: {account.address}")
        print()

        # Step 6: Retry with payment in _meta['x402/payment']
        print("--- Retrying ping with payment ---")
        paid_result = await client.call_tool(
            "ping",
            arguments={},
            meta={"x402/payment": payment_payload_dict},
        )

        print(f"  isError: {paid_result.is_error}")
        if paid_result.content:
            print(f"  Result:  {paid_result.content[0].text}")

        # Check for settlement receipt in _meta
        if paid_result.meta and "x402/payment-response" in paid_result.meta:
            receipt = paid_result.meta["x402/payment-response"]
            print()
            print("  Settlement receipt:")
            print(f"    success:     {receipt.get('success')}")
            print(f"    transaction: {receipt.get('transaction')}")
            print(f"    network:     {receipt.get('network')}")

        # Step 7: Call inference (paid) -- the exchange's brain
        tool_name = sys.argv[1] if len(sys.argv) > 1 else "ping"
        if tool_name == "inference" or "--inference" in sys.argv:
            print()
            print("=" * 60)
            print("  INFERENCE (paid Qwen3-8B via x402)")
            print("=" * 60)

            # Call without payment to get requirements
            print("--- Calling inference (no payment) ---")
            inf_result = await client.call_tool("inference", arguments={"prompt": "What are you?"})

            inf_pr_data = None
            if inf_result.structured_content:
                sc = inf_result.structured_content
                if isinstance(sc, dict) and "x402Version" in sc:
                    inf_pr_data = sc
            if inf_pr_data is None and inf_result.content:
                try:
                    parsed = json.loads(inf_result.content[0].text)
                    if "x402Version" in parsed:
                        inf_pr_data = parsed
                except (json.JSONDecodeError, AttributeError):
                    pass

            if inf_pr_data is None:
                print("  ERROR: Could not get PaymentRequired for inference")
                return

            print(f"  Payment required: {inf_pr_data['accepts'][0]['amount']} atomic USDC")

            # Sign payment
            print("--- Signing payment for inference ---")
            inf_payment_required = parse_payment_required(inf_pr_data)
            inf_payload = await x402_client.create_payment_payload(inf_payment_required)
            inf_payload_dict = inf_payload.model_dump(by_alias=True, exclude_none=True)

            # Retry with payment + prompt
            print("--- Calling inference with payment ---")
            inf_paid = await client.call_tool(
                "inference",
                arguments={
                    "prompt": "What are you? Respond in exactly 2 sentences.",
                    "max_tokens": 256,
                },
                meta={"x402/payment": inf_payload_dict},
            )

            print(f"  isError: {inf_paid.is_error}")
            if inf_paid.content:
                result_data = json.loads(inf_paid.content[0].text)
                print(f"  Model:    {result_data.get('model')}")
                print(f"  Response: {result_data.get('response')}")
                print(f"  Usage:    {result_data.get('usage')}")
                print(f"  Paid:     {result_data.get('amount')}")
                if result_data.get("settlement_tx"):
                    print(f"  Tx:       {result_data['settlement_tx']}")


def main():
    asyncio.run(buy_via_mcp())


if __name__ == "__main__":
    main()
