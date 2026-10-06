"""Zero-touch bootstrap: generate wallets, fund via CDP faucet, start MCP server, run buyer.

One command. No human. First dollar accepted.

Requires CDP API credentials in .env (free from https://portal.cdp.coinbase.com):
  CDP_API_KEY_ID=...
  CDP_API_KEY_SECRET=...
  CDP_WALLET_SECRET=...
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv, set_key
from eth_account import Account


def ensure_wallets():
    """Generate seller + buyer wallets if they don't exist."""
    load_dotenv()
    if os.environ.get("SELLER_ADDRESS") and os.environ.get("BUYER_PRIVATE_KEY"):
        print(f"Wallets exist:")
        print(f"  Seller: {os.environ['SELLER_ADDRESS']}")
        print(f"  Buyer:  {os.environ['BUYER_ADDRESS']}")
        return

    print("Generating wallets...")
    seller = Account.create()
    buyer = Account.create()

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    set_key(env_path, "SELLER_ADDRESS", seller.address)
    set_key(env_path, "SELLER_PRIVATE_KEY", seller.key.hex())
    set_key(env_path, "BUYER_ADDRESS", buyer.address)
    set_key(env_path, "BUYER_PRIVATE_KEY", buyer.key.hex())
    set_key(env_path, "FACILITATOR_URL", "https://x402.org/facilitator")
    set_key(env_path, "NETWORK", "eip155:84532")

    # Reload
    load_dotenv(override=True)
    print(f"  Seller: {seller.address}")
    print(f"  Buyer:  {buyer.address}")


async def fund_buyer():
    """Fund buyer with testnet USDC via CDP faucet."""
    from cdp import CdpClient
    from web3 import Web3

    buyer_address = os.environ["BUYER_ADDRESS"]

    # Check existing balance
    w3 = Web3(Web3.HTTPProvider("https://sepolia.base.org"))
    usdc_address = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
    usdc_abi = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]
    usdc = w3.eth.contract(address=usdc_address, abi=usdc_abi)
    balance = usdc.functions.balanceOf(buyer_address).call()

    if balance >= 10_000:  # >= $0.01 USDC (enough for one ping)
        print(f"Buyer already funded: {balance / 1_000_000:.6f} USDC")
        return

    print(f"Funding buyer ({buyer_address}) via CDP faucet...")
    async with CdpClient() as cdp:
        usdc_tx_hash = await cdp.evm.request_faucet(
            address=buyer_address,
            network="base-sepolia",
            token="usdc",
        )
        print(f"  Faucet tx: https://sepolia.basescan.org/tx/{usdc_tx_hash}")
        print(f"  Waiting for confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(usdc_tx_hash, timeout=120)
        balance = usdc.functions.balanceOf(buyer_address).call()
        print(f"  Funded: {balance / 1_000_000:.6f} USDC")


def start_mcp_server():
    """Start the MCP server in a subprocess."""
    print("Starting MCP server on :4022...")
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    proc = subprocess.Popen(
        [venv_python, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            import httpx
            resp = httpx.post(
                "http://localhost:4022/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=5,
            )
            if resp.status_code == 200:
                print("  MCP server ready.")
                return proc
        except Exception:
            pass
    print("  WARNING: MCP server may not be ready yet.")
    return proc


async def run_mcp_buyer():
    """Run the MCP buyer -- full autonomous payment flow."""
    from mcp import Client

    from x402 import x402Client, parse_payment_required
    from x402.mechanisms.evm import EthAccountSigner
    from x402.mechanisms.evm.exact.register import register_exact_evm_client

    account = Account.from_key(os.environ["BUYER_PRIVATE_KEY"])
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    register_exact_evm_client(x402_client, signer)

    print()
    print("=" * 60)
    print("  AUTONOMOUS PAYMENT FLOW")
    print("=" * 60)
    print(f"  Buyer:  {account.address}")
    print(f"  Server: http://localhost:4022/mcp")
    print()

    async with Client("http://localhost:4022/mcp") as client:
        # Discover tools
        tools = await client.list_tools()
        print("Tools discovered:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        print()

        # Call ping without payment
        print("1. Calling ping (no payment)...")
        result = await client.call_tool("ping")
        assert result.is_error, "Expected payment-required error"

        # Extract PaymentRequired
        pr_data = result.structured_content
        if not pr_data or "accepts" not in pr_data:
            pr_data = json.loads(result.content[0].text)

        accepts = pr_data["accepts"][0]
        print(f"   Payment required: {accepts['amount']} atomic USDC")
        print(f"   Pay to: {accepts['payTo']}")
        print()

        # Sign payment
        print("2. Signing payment...")
        payment_required = parse_payment_required(pr_data)
        payment_payload = await x402_client.create_payment_payload(payment_required)
        payload_dict = payment_payload.model_dump(by_alias=True, exclude_none=True)
        print(f"   Signed by: {account.address}")
        print()

        # Retry with payment
        print("3. Retrying with payment...")
        paid_result = await client.call_tool(
            "ping",
            arguments={},
            meta={"x402/payment": payload_dict},
        )

        if paid_result.is_error:
            error_data = paid_result.structured_content or json.loads(paid_result.content[0].text)
            print(f"   FAILED: {error_data.get('error', 'unknown error')}")
            return False
        else:
            result_data = json.loads(paid_result.content[0].text)
            print(f"   Result: {result_data}")

            if paid_result.meta and "x402/payment-response" in paid_result.meta:
                receipt = paid_result.meta["x402/payment-response"]
                print()
                print("   Settlement receipt:")
                print(f"     success:     {receipt.get('success')}")
                print(f"     transaction: {receipt.get('transaction')}")
                print(f"     network:     {receipt.get('network')}")
                tx = receipt.get("transaction", "")
                if tx:
                    print(f"     explorer:    https://sepolia.basescan.org/tx/{tx}")

            print()
            print("=" * 60)
            print("  FIRST DOLLAR ACCEPTED.")
            print("=" * 60)
            return True


async def main():
    # Check CDP credentials
    load_dotenv()
    required = ["CDP_API_KEY_ID", "CDP_API_KEY_SECRET", "CDP_WALLET_SECRET"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("Missing CDP credentials in .env:")
        for k in missing:
            print(f"  {k}")
        print()
        print("Get free API keys at: https://portal.cdp.coinbase.com")
        print("Then add to .env and re-run.")
        sys.exit(1)

    # Step 1: Wallets
    ensure_wallets()
    print()

    # Step 2: Fund
    await fund_buyer()
    print()

    # Step 3: Start MCP server
    server_proc = start_mcp_server()
    print()

    try:
        # Step 4: Run buyer
        success = await run_mcp_buyer()
        if not success:
            sys.exit(1)
    finally:
        # Cleanup
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
