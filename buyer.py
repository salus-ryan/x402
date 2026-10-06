"""x402 buyer -- pays for a resource using USDC on Base Sepolia."""
import asyncio
import os
import sys

from dotenv import load_dotenv
from eth_account import Account

from x402 import x402Client
from x402.http.clients import x402HttpxClient
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client

load_dotenv()

BUYER_PRIVATE_KEY = os.environ["BUYER_PRIVATE_KEY"]
SELLER_URL = os.environ.get("SELLER_URL", "http://localhost:4021")


async def buy():
    account = Account.from_key(BUYER_PRIVATE_KEY)
    signer = EthAccountSigner(account)

    client = x402Client()
    register_exact_evm_client(client, signer)

    async with x402HttpxClient(client) as http:
        print(f"Buyer wallet: {account.address}")
        print(f"Hitting:      {SELLER_URL}/ping")
        print()

        response = await http.get(f"{SELLER_URL}/ping")

        print(f"Status: {response.status_code}")
        print(f"Body:   {response.json()}")

        payment_response = response.headers.get("payment-response")
        if payment_response:
            import base64
            import json
            receipt = json.loads(base64.b64decode(payment_response))
            print()
            print("Settlement receipt:")
            print(f"  success:     {receipt.get('success')}")
            print(f"  transaction: {receipt.get('transaction')}")
            print(f"  network:     {receipt.get('network')}")


def main():
    asyncio.run(buy())


if __name__ == "__main__":
    main()
