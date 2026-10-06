# x402 Exchange -- Autonomous Self-Improving Agent

An autonomous agent that earns revenue by serving Qwen3-8B inference,
accepts payment in USDC via the x402 protocol, and uses revenue to
fund its own improvement through LoRA fine-tuning. No human in the loop.

## Architecture

```
                  Agent (any MCP client)
                        |
                  tools/call "inference"
                        |
              +-------- v --------+
              |  MCP Server (:4022) |  x402 payment gate
              |  5 tools:           |
              |    inference $0.01  |  <- paid: Qwen3-8B chat
              |    ping     $0.01  |  <- paid: proof of life
              |    health   free   |  <- system check
              |    status   free   |  <- P&L report
              |    self_improve    |  <- trigger fine-tuning
              +--------+----------+
                       |
           verify + settle on-chain (Base Sepolia USDC)
                       |
              +--------v----------+
              |  Modal (L4 GPU)    |  $0.80/hr, scale-to-zero
              |  vLLM + Qwen3-8B  |
              |  + LoRA adapters   |
              +-------------------+
                       |
              x402-lora-adapters volume
                       |
              +--------v----------+
              |  finetune.py       |  A10G GPU, ~2hr
              |  LoRA SFT on       |
              |  successful txns   |
              +-------------------+
```

## Quick Start

```bash
# One command -- generates wallets, funds buyer, starts server, pays
.venv/bin/python bootstrap.py
```

## Setup

```bash
cd ~/Projects/x402
uv venv
uv pip install "x402[fastapi,httpx,evm]" uvicorn python-dotenv eth-account \
    "mcp[cli]" structlog cdp-sdk web3 httpx pydantic modal
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

## Running

```bash
# Deploy model to Modal
modal deploy modal_app.py

# Start MCP server (agents connect here)
.venv/bin/python mcp_server.py                # streamable-http on :4022
.venv/bin/python mcp_server.py stdio          # for Claude Desktop

# Check P&L
.venv/bin/python ledger.py

# Check fine-tune affordability
.venv/bin/python finetune.py check

# Prepare training data
.venv/bin/python finetune.py prepare

# Launch fine-tuning on Modal
modal run finetune.py
```

## The Autonomous Loop

1. **Agent discovers tools** via MCP `tools/list`
2. **Calls `inference`** without payment -- receives PaymentRequired
3. **Signs USDC payment** (EIP-3009 authorization)
4. **Retries with payment** -- MCP server verifies + settles on-chain
5. **Routes to Modal brain** -- Qwen3-8B generates response
6. **Records payment** in ledger (on-chain + local)
7. **When revenue >= $2.20** -- entity triggers LoRA fine-tuning
8. **New adapter loaded** -- model improves, earns more, repeats

## P&L

The entity tracks its own finances:

```
$ .venv/bin/python ledger.py

x402 Exchange P&L Report
==================================================
Seller:   0xEd4bc7A1eC9B6e2cd8DEeD2339AA549E38c8094A
Network:  eip155:84532

REVENUE
  USDC balance:  $0.030000
  Payments logged: 3

COSTS
  Modal spend:   $0.02 (inference + training)

NET POSITION
  $+0.01
  Can afford fine-tune: No (need 218 more paid requests)
```

## Files

| File | Purpose |
|---|---|
| `modal_app.py` | Qwen3-8B on Modal (L4 GPU, vLLM, OpenAI-compatible) |
| `mcp_server.py` | MCP server with x402 payment-gated tools |
| `mcp_buyer.py` | MCP client that discovers, pays, and calls tools |
| `ledger.py` | P&L tracking: on-chain USDC revenue + Modal spend |
| `finetune.py` | LoRA fine-tuning on Modal (A10G, QLoRA) |
| `bootstrap.py` | Zero-touch: wallets + fund + server + buyer |
| `seller.py` | FastAPI HTTP server with x402 middleware |
| `buyer.py` | HTTP client with auto 402 handling |
| `fund.py` | Fund buyer wallet via CDP faucet |
| `generate_wallets.py` | Generate seller/buyer EVM keypairs |

## Endpoints

- **Modal exchange**: `https://salus--x402-exchange-exchange.us-east.modal.direct`
- **MCP server**: `http://localhost:4022/mcp` (or wherever deployed)

## Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "x402-exchange": {
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
