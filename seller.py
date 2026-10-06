"""x402 seller -- FastAPI server that charges $0.01 per request on Base Sepolia."""
import os

from dotenv import load_dotenv
from fastapi import FastAPI

from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

load_dotenv()

app = FastAPI(title="x402 MVP Seller")

SELLER_ADDRESS = os.environ["SELLER_ADDRESS"]
FACILITATOR_URL = os.environ.get("FACILITATOR_URL", "https://x402.org/facilitator")
NETWORK: Network = os.environ.get("NETWORK", "eip155:84532")  # Base Sepolia default

facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=FACILITATOR_URL))

server = x402ResourceServer(facilitator)
server.register(NETWORK, ExactEvmServerScheme())

routes: dict[str, RouteConfig] = {
    "GET /ping": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=SELLER_ADDRESS,
                price="$0.01",
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description="A paid ping endpoint -- proof of life for x402 payments",
    ),
}

app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)


@app.get("/ping")
async def ping() -> dict:
    return {"message": "pong", "paid": True, "amount": "$0.01"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "seller": SELLER_ADDRESS, "network": NETWORK}


def main():
    import uvicorn
    uvicorn.run("seller:app", host="0.0.0.0", port=4021, reload=True)


if __name__ == "__main__":
    main()
