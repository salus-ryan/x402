"""Fund the buyer wallet with testnet USDC on Base Sepolia via CDP faucet.

Requires CDP API credentials in .env:
  CDP_API_KEY_ID=...
  CDP_API_KEY_SECRET=...
  CDP_WALLET_SECRET=...

These are free -- sign up at https://portal.cdp.coinbase.com
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


async def fund_wallet():
    from cdp import CdpClient
    from web3 import Web3

    buyer_address = os.environ["BUYER_ADDRESS"]
    print(f"Funding buyer wallet: {buyer_address}")
    print(f"Network: Base Sepolia")
    print()

    async with CdpClient() as cdp:
        # Request USDC from faucet (1 USDC per request)
        print("Requesting 1 USDC from CDP faucet...")
        usdc_tx_hash = await cdp.evm.request_faucet(
            address=buyer_address,
            network="base-sepolia",
            token="usdc",
        )
        print(f"Faucet tx: https://sepolia.basescan.org/tx/{usdc_tx_hash}")

        # Wait for confirmation
        print("Waiting for confirmation...")
        w3 = Web3(Web3.HTTPProvider("https://sepolia.base.org"))
        receipt = w3.eth.wait_for_transaction_receipt(usdc_tx_hash, timeout=60)
        print(f"Confirmed in block {receipt.blockNumber}")

        # Check USDC balance
        usdc_address = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
        usdc_abi = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]
        usdc = w3.eth.contract(address=usdc_address, abi=usdc_abi)
        balance = usdc.functions.balanceOf(buyer_address).call()
        print(f"USDC balance: {balance / 1_000_000:.6f} USDC ({balance} atomic)")
        print()
        print("Buyer wallet funded. Ready for x402 payments.")


def main():
    # Check for CDP credentials
    required = ["CDP_API_KEY_ID", "CDP_API_KEY_SECRET", "CDP_WALLET_SECRET"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("ERROR: Missing CDP credentials in .env:")
        for k in missing:
            print(f"  {k}")
        print()
        print("Get free API keys at: https://portal.cdp.coinbase.com")
        print("Add them to .env:")
        print("  CDP_API_KEY_ID=...")
        print("  CDP_API_KEY_SECRET=...")
        print("  CDP_WALLET_SECRET=...")
        sys.exit(1)

    asyncio.run(fund_wallet())


if __name__ == "__main__":
    main()
