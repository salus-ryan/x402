# x402 MVP -- Accept Your First Dollar

Zero-touch x402 payment flow. One command. No human in the loop.

```bash
.venv/bin/python bootstrap.py
```

```
Generating wallets...
  Seller: 0x...
  Buyer:  0x...

Funding buyer via CDP faucet...
  Faucet tx: https://sepolia.basescan.org/tx/0x...
  Funded: 1.000000 USDC

Starting MCP server on :4022...
  MCP server ready.

============================================================
  AUTONOMOUS PAYMENT FLOW
============================================================
  Buyer:  0x...
  Server: http://localhost:4022/mcp

Tools discovered:
  - ping: A paid ping endpoint -- proof of life for x402...
  - health: Free health check -- no payment required.

1. Calling ping (no payment)...
   Payment required: 10000 atomic USDC
   Pay to: 0x...

2. Signing payment...
   Signed by: 0x...

3. Retrying with payment...
   Result: {"message": "pong", "paid": true, "amount": "$0.01"}

   Settlement receipt:
     success:     True
     transaction: 0x...
     network:     eip155:84532
     explorer:    https://sepolia.basescan.org/tx/0x...

============================================================
  FIRST DOLLAR ACCEPTED.
============================================================
```

## Setup

```bash
cd ~/Projects/x402
uv venv
uv pip install "x402[fastapi,httpx,evm]" uvicorn python-dotenv eth-account \
    "mcp[cli]" structlog cdp-sdk web3 httpx
```

### One-time: CDP API keys (free)

1. Go to https://portal.cdp.coinbase.com
2. Create an API key + wallet secret
3. Add to `.env`:

```env
CDP_API_KEY_ID=...
CDP_API_KEY_SECRET=...
CDP_WALLET_SECRET=...
```

That's it. `bootstrap.py` handles wallet generation, faucet funding,
server startup, and payment execution autonomously.

## What happens

```
bootstrap.py
  |
  |  1. Generate seller + buyer EVM wallets (eth_account)
  |  2. Fund buyer with 1 USDC via CDP Faucet API (no browser)
  |  3. Start MCP server on :4022
  |  4. Connect as MCP client
  |  5. Discover tools (tools/list)
  |  6. Call ping -> 402 PaymentRequired
  |  7. Sign EIP-3009 transferWithAuthorization
  |  8. Retry with _meta["x402/payment"]
  |  9. Server verifies + settles on-chain via facilitator
  |  10. Buyer gets "pong" + settlement tx hash
  |
  v
  USDC moves from buyer to seller on Base Sepolia. No human touched anything.
```

## Architecture

### MCP Transport (primary -- for agents)

Any MCP client (Claude Desktop, Devin, Cursor) can connect and pay:

```
Agent -> tools/list -> sees "ping" ($0.01 USDC) + "health" (free)
Agent -> tools/call "ping" -> isError: true + PaymentRequired in structuredContent
Agent reads accepts[0], signs EIP-3009 authorization
Agent -> tools/call "ping" + _meta["x402/payment"] -> pong + settlement receipt
```

### HTTP Transport (alternative)

```
curl -> GET /ping -> 402 + PAYMENT-REQUIRED header
sign -> GET /ping + PAYMENT-SIGNATURE header -> 200 + PAYMENT-RESPONSE header
```

## Running Individually

```bash
# MCP server (agents connect here)
.venv/bin/python mcp_server.py                # streamable-http on :4022
.venv/bin/python mcp_server.py stdio          # for Claude Desktop

# HTTP server (curl / httpx)
.venv/bin/python seller.py                    # FastAPI on :4021

# MCP buyer
.venv/bin/python mcp_buyer.py

# HTTP buyer
.venv/bin/python buyer.py

# Fund buyer wallet only
.venv/bin/python fund.py
```

## Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "x402-mvp": {
      "url": "http://localhost:4022/mcp"
    }
  }
}
```

## Mainnet

Two-line `.env` change:

```env
FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
NETWORK=eip155:8453
```

Fund the buyer with real USDC on Base. Same code, real dollars.

## Files

| File | Purpose |
|---|---|
| `bootstrap.py` | Zero-touch: wallets + fund + server + buyer in one command |
| `mcp_server.py` | MCP server with x402 payment-gated tools |
| `mcp_buyer.py` | MCP client that discovers, pays, and calls tools |
| `seller.py` | FastAPI HTTP server with x402 middleware |
| `buyer.py` | HTTP client with auto 402 handling |
| `fund.py` | Fund buyer wallet via CDP faucet |
| `generate_wallets.py` | Generate seller/buyer EVM keypairs |

## How x402 Works

x402 is a payment protocol built on HTTP 402. Three data structures ride on
any transport (HTTP headers, MCP `_meta`, A2A task metadata):

- **PaymentRequired**: scheme, network, amount, asset, payTo
- **PaymentPayload**: signed EIP-3009 `transferWithAuthorization`
- **SettlementResponse**: on-chain tx hash

Gasless for both buyer and seller. The facilitator verifies the signature,
checks on-chain balance, and broadcasts the USDC transfer.
