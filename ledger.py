"""P&L ledger for the x402 exchange entity.

Tracks two things:
  1. Revenue: USDC received on-chain (seller wallet balance)
  2. Costs:   Modal compute spend (via billing API)

The entity uses this to decide whether it can afford to self-improve
(trigger LoRA fine-tuning) or needs to conserve resources.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel
from web3 import Web3

load_dotenv()

log = structlog.get_logger()

# ── Config ───────────────────────────────────────────────────────────────────

SELLER_ADDRESS = os.environ["SELLER_ADDRESS"]
RPC_URL = os.environ.get("RPC_URL", "https://sepolia.base.org")

# USDC contract addresses by network
USDC_CONTRACTS: dict[str, str] = {
    "eip155:84532": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # Base Sepolia
    "eip155:8453": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # Base Mainnet
}
NETWORK = os.environ.get("NETWORK", "eip155:84532")

LEDGER_PATH = Path(os.environ.get("LEDGER_PATH", "ledger.json"))

ERC20_BALANCE_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]


# ── Models ───────────────────────────────────────────────────────────────────

class LedgerSnapshot(BaseModel):
    """Point-in-time snapshot of the entity's financial state."""
    timestamp: str
    # Revenue side
    usdc_balance_atomic: int
    usdc_balance: float  # human-readable (divided by 1e6)
    total_payments_received: int  # count of payments settled
    # Cost side
    modal_spend_usd: float
    modal_spend_period: str  # e.g. "2026-10-06 to 2026-10-07"
    # Net
    net_position_usd: float  # usdc_balance - modal_spend
    can_afford_finetune: bool  # enough revenue to justify a training run


class LedgerEntry(BaseModel):
    """Individual transaction record."""
    timestamp: str
    event: str  # "payment_received", "finetune_started", "finetune_completed"
    amount_usdc: float | None = None
    settlement_tx: str | None = None
    modal_cost_usd: float | None = None
    details: dict[str, Any] | None = None


class Ledger(BaseModel):
    """Persistent ledger state."""
    created: str
    entries: list[LedgerEntry] = []
    snapshots: list[LedgerSnapshot] = []
    total_revenue_usdc: float = 0.0
    total_spend_usd: float = 0.0
    finetune_count: int = 0


# ── Revenue tracking (on-chain) ─────────────────────────────────────────────

def get_usdc_balance(
    address: str = SELLER_ADDRESS,
    network: str = NETWORK,
    rpc_url: str = RPC_URL,
) -> tuple[int, float]:
    """Query on-chain USDC balance. Returns (atomic, human-readable)."""
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    usdc_addr = USDC_CONTRACTS.get(network)
    if not usdc_addr:
        raise ValueError(f"No USDC contract for network {network}")

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(usdc_addr),
        abi=ERC20_BALANCE_ABI,
    )
    balance_atomic = contract.functions.balanceOf(
        Web3.to_checksum_address(address)
    ).call()
    balance_human = balance_atomic / 1e6
    return balance_atomic, balance_human


# ── Cost tracking (Modal billing) ───────────────────────────────────────────

def get_modal_spend(
    start: datetime | None = None,
    end: datetime | None = None,
    app_filter: str = "x402",
) -> tuple[float, str]:
    """Query Modal billing API for x402-related spend.

    Returns (total_usd, period_description).
    """
    try:
        import modal

        if start is None:
            now = datetime.now(timezone.utc)
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if end is None:
            end = datetime.now(timezone.utc)

        workspace = modal.Workspace.from_context()
        report = workspace.billing.report(start=start, end=end)

        total = 0.0
        for entry in report:
            if app_filter.lower() in entry.app_name.lower():
                total += entry.amount

        period = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        return total, period

    except Exception as e:
        log.warning("modal billing query failed", error=str(e))
        return 0.0, "unknown (billing query failed)"


# ── Ledger persistence ──────────────────────────────────────────────────────

def load_ledger(path: Path = LEDGER_PATH) -> Ledger:
    """Load or create the ledger."""
    if path.exists():
        return Ledger.model_validate_json(path.read_text())
    return Ledger(created=datetime.now(timezone.utc).isoformat())


def save_ledger(ledger: Ledger, path: Path = LEDGER_PATH) -> None:
    """Save the ledger to disk."""
    path.write_text(ledger.model_dump_json(indent=2))


def record_payment(
    ledger: Ledger,
    amount_usdc: float,
    settlement_tx: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Record an incoming payment."""
    entry = LedgerEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event="payment_received",
        amount_usdc=amount_usdc,
        settlement_tx=settlement_tx,
        details=details,
    )
    ledger.entries.append(entry)
    ledger.total_revenue_usdc += amount_usdc
    log.info(
        "payment recorded",
        amount=amount_usdc,
        tx=settlement_tx,
        total_revenue=ledger.total_revenue_usdc,
    )


def record_finetune(
    ledger: Ledger,
    event: str,  # "finetune_started" or "finetune_completed"
    modal_cost_usd: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Record a fine-tuning event."""
    entry = LedgerEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event=event,
        modal_cost_usd=modal_cost_usd,
        details=details,
    )
    ledger.entries.append(entry)
    if event == "finetune_completed":
        ledger.finetune_count += 1
        if modal_cost_usd:
            ledger.total_spend_usd += modal_cost_usd


# ── Snapshot (P&L report) ───────────────────────────────────────────────────

# Estimated cost of one LoRA fine-tune run on A10G (~2 hours)
FINETUNE_COST_ESTIMATE_USD = 2.20  # $1.10/hr * 2hr

def take_snapshot(ledger: Ledger) -> LedgerSnapshot:
    """Capture current P&L state."""
    balance_atomic, balance_human = get_usdc_balance()
    modal_spend, period = get_modal_spend()
    net = balance_human - modal_spend
    can_afford = balance_human >= FINETUNE_COST_ESTIMATE_USD

    snapshot = LedgerSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        usdc_balance_atomic=balance_atomic,
        usdc_balance=balance_human,
        total_payments_received=len(
            [e for e in ledger.entries if e.event == "payment_received"]
        ),
        modal_spend_usd=modal_spend,
        modal_spend_period=period,
        net_position_usd=net,
        can_afford_finetune=can_afford,
    )

    ledger.snapshots.append(snapshot)
    log.info(
        "snapshot taken",
        usdc=balance_human,
        spend=modal_spend,
        net=net,
        can_finetune=can_afford,
    )
    return snapshot


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """Print current P&L report."""
    import sys

    ledger = load_ledger()

    # Always get fresh on-chain balance
    balance_atomic, balance_human = get_usdc_balance()
    print(f"x402 Exchange P&L Report")
    print(f"{'=' * 50}")
    print(f"Seller:   {SELLER_ADDRESS}")
    print(f"Network:  {NETWORK}")
    print()

    # Revenue
    print(f"REVENUE")
    print(f"  USDC balance:  ${balance_human:.6f}")
    print(f"  Atomic:        {balance_atomic}")
    payment_count = len(
        [e for e in ledger.entries if e.event == "payment_received"]
    )
    print(f"  Payments logged: {payment_count}")
    print(f"  Total logged:    ${ledger.total_revenue_usdc:.6f}")
    print()

    # Costs
    print(f"COSTS")
    try:
        modal_spend, period = get_modal_spend()
        print(f"  Modal spend:   ${modal_spend:.4f} ({period})")
    except Exception as e:
        modal_spend = 0.0
        print(f"  Modal spend:   unavailable ({e})")
    print(f"  Total logged:  ${ledger.total_spend_usd:.4f}")
    print()

    # Net
    net = balance_human - modal_spend
    print(f"NET POSITION")
    print(f"  ${net:+.6f}")
    print(f"  Can afford fine-tune: {'Yes' if balance_human >= FINETUNE_COST_ESTIMATE_USD else 'No'}")
    print(f"  Fine-tune cost est:   ${FINETUNE_COST_ESTIMATE_USD:.2f}")
    print(f"  Fine-tunes completed: {ledger.finetune_count}")

    save_ledger(ledger)


if __name__ == "__main__":
    main()
