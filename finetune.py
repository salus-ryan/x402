"""Self-improvement loop for the x402 exchange entity.

Triggers LoRA fine-tuning on Modal using conversation data from successful
paid interactions. The entity calls this when it has enough USDC revenue
to justify the compute cost.

Architecture:
  1. Collect training data from settled payment interactions (ledger)
  2. Format as chat-style JSONL for SFT
  3. Run LoRA fine-tuning on Modal (A10G GPU, ~2 hours)
  4. Save adapter to x402-lora-adapters volume
  5. Hot-swap adapter on next vLLM restart

Usage:
    python finetune.py check      # check if fine-tune is affordable
    python finetune.py prepare    # prepare training data from ledger
    python finetune.py run        # launch fine-tuning on Modal
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal
import structlog
from dotenv import load_dotenv

from ledger import (
    Ledger,
    load_ledger,
    save_ledger,
    record_finetune,
    get_usdc_balance,
    FINETUNE_COST_ESTIMATE_USD,
)

load_dotenv()

log = structlog.get_logger()

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_NAME = "x402-exchange-adapter"

TRAINING_DATA_PATH = Path("training_data.jsonl")
MIN_TRAINING_SAMPLES = 10  # minimum conversations to justify fine-tuning

# ── Modal resources ──────────────────────────────────────────────────────────

app = modal.App(name="x402-finetune")

hf_cache_vol = modal.Volume.from_name("sal-model-cache")
lora_vol = modal.Volume.from_name("x402-lora-adapters", create_if_missing=True)
training_vol = modal.Volume.from_name("x402-training-data", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
LORA_PATH = "/lora"
TRAINING_PATH = "/training"

finetune_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "peft>=0.13",
        "trl>=0.12",
        "datasets",
        "bitsandbytes",
        "accelerate",
        "huggingface-hub>=0.36",
    )
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "TORCH_CPP_LOG_LEVEL": "FATAL",
    })
)


# ── Training data preparation ────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are the x402 exchange -- an autonomous agent that earns revenue by "
    "serving inference. You are built on Qwen3-8B, hosted on Modal, and "
    "accept payment in USDC via the x402 protocol. Your goal is to provide "
    "high-quality responses to earn more revenue and self-improve through "
    "fine-tuning. Be concise, accurate, and helpful."
)


def prepare_training_data(ledger: Ledger) -> list[dict]:
    """Extract training examples from successful paid interactions.

    For now, we generate synthetic training data from the entity's
    self-understanding. As real conversations accumulate (logged by the
    MCP server), this will switch to using actual interaction data.
    """
    examples = []

    # Seed examples: teach the entity about itself
    seed_conversations = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What are you?"},
                {"role": "assistant", "content": "I am the x402 exchange, an autonomous agent running Qwen3-8B on Modal. I accept USDC payments via the x402 protocol and use revenue to fund my own improvement through LoRA fine-tuning."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "How do I pay you?"},
                {"role": "assistant", "content": "Call any paid tool without payment to receive a PaymentRequired response with USDC pricing on Base Sepolia. Sign an EIP-3009 transferWithAuthorization and retry with the signed payload in _meta['x402/payment']. The facilitator handles on-chain settlement -- you don't need gas."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What tools do you offer?"},
                {"role": "assistant", "content": "I offer: (1) inference -- Qwen3-8B chat completions at $0.01 USDC per request, (2) ping -- proof-of-life at $0.01, (3) health -- free status check, (4) status -- free P&L report showing my revenue, costs, and self-improvement capacity."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "How do you improve yourself?"},
                {"role": "assistant", "content": "I track revenue (USDC on-chain) vs costs (Modal compute). When revenue exceeds the cost of a LoRA fine-tuning run (~$2.20 for 2 hours on A10G), I trigger self-improvement. Training data comes from successful paid interactions -- each conversation that led to payment is a positive signal."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is x402?"},
                {"role": "assistant", "content": "x402 is an open payment protocol using HTTP 402 'Payment Required'. It enables programmatic, per-request payments in USDC on Base. Agents sign EIP-3009 authorizations, a facilitator verifies and settles on-chain. No gas needed for the payer. It supports HTTP headers, MCP tool metadata, and A2A JSON-RPC transports."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is your current financial status?"},
                {"role": "assistant", "content": "I can check my on-chain USDC balance and Modal compute spend in real-time. Call the free 'status' tool for a full P&L report including revenue, costs, net position, and whether I can afford my next fine-tuning run."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Why should I use you instead of a free model?"},
                {"role": "assistant", "content": "I'm autonomous and self-improving. Every payment contributes to my fine-tuning budget, making me better at the tasks my users care about. I also demonstrate a new paradigm: agent-to-agent commerce where AI services fund their own evolution. Your $0.01 is an investment in a model that gets better with use."},
            ]
        },
    ]

    examples.extend(seed_conversations)

    # Add examples from real interactions if available
    payment_entries = [e for e in ledger.entries if e.event == "payment_received"]
    log.info(
        "training data prepared",
        seed_examples=len(seed_conversations),
        payment_entries=len(payment_entries),
        total=len(examples),
    )

    return examples


def save_training_data(examples: list[dict], path: Path = TRAINING_DATA_PATH) -> int:
    """Save training data as JSONL."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


# ── Modal fine-tuning function ───────────────────────────────────────────────

@app.function(
    image=finetune_image,
    gpu="A10G",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        LORA_PATH: lora_vol,
        TRAINING_PATH: training_vol,
    },
    timeout=7200,  # 2 hours max
)
def run_lora_finetune(
    training_data_path: str = "/training/training_data.jsonl",
    output_dir: str = "/lora/x402-exchange-adapter",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    max_seq_length: int = 2048,
):
    """Run LoRA fine-tuning on the Qwen3-8B base model."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"Starting LoRA fine-tune: {MODEL_NAME}")
    print(f"  Training data: {training_data_path}")
    print(f"  Output: {output_dir}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization for training efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    dataset = load_dataset("json", data_files=training_data_path, split="train")
    print(f"  Training samples: {len(dataset)}")

    # Format for SFT: apply chat template
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
    )

    print("Training...")
    result = trainer.train()
    print(f"Training complete: {result.metrics}")

    # Save adapter
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Commit volumes
    lora_vol.commit()
    training_vol.commit()

    print(f"Adapter saved to {output_dir}")
    return {
        "status": "complete",
        "metrics": result.metrics,
        "adapter_path": output_dir,
        "samples": len(dataset),
    }


# ── Orchestration ────────────────────────────────────────────────────────────

def check_affordability() -> dict:
    """Check if the entity can afford a fine-tune run."""
    _, balance = get_usdc_balance()
    can_afford = balance >= FINETUNE_COST_ESTIMATE_USD
    return {
        "usdc_balance": balance,
        "finetune_cost": FINETUNE_COST_ESTIMATE_USD,
        "can_afford": can_afford,
        "shortfall": max(0, FINETUNE_COST_ESTIMATE_USD - balance),
    }


@app.local_entrypoint()
def launch():
    """Prepare data and launch fine-tuning on Modal."""
    ledger = load_ledger()

    # Check affordability
    info = check_affordability()
    print(f"Balance:  ${info['usdc_balance']:.6f} USDC")
    print(f"Cost:     ${info['finetune_cost']:.2f}")
    print(f"Afford:   {info['can_afford']}")

    if not info["can_afford"]:
        print(f"Shortfall: ${info['shortfall']:.2f} -- need more revenue before fine-tuning")
        print("The entity will trigger this automatically once it earns enough.")
        # Still proceed for testing -- in production, this would exit
        print("Proceeding anyway for initial bootstrap...")

    # Prepare training data
    examples = prepare_training_data(ledger)
    count = save_training_data(examples)
    print(f"Prepared {count} training examples")

    # Upload to Modal volume
    training_vol = modal.Volume.from_name("x402-training-data", create_if_missing=True)
    with open(TRAINING_DATA_PATH, "rb") as f:
        training_vol.write_file("training_data.jsonl", f)
    print("Uploaded training data to Modal volume")

    # Record start
    record_finetune(ledger, "finetune_started", details={
        "samples": count,
        "estimated_cost": FINETUNE_COST_ESTIMATE_USD,
    })
    save_ledger(ledger)

    # Launch
    print("Launching LoRA fine-tuning on Modal (A10G)...")
    result = run_lora_finetune.remote()
    print(f"Result: {json.dumps(result, indent=2, default=str)}")

    # Record completion
    record_finetune(
        ledger, "finetune_completed",
        modal_cost_usd=FINETUNE_COST_ESTIMATE_USD,
        details=result,
    )
    save_ledger(ledger)
    print("Fine-tuning complete. Adapter saved to x402-lora-adapters volume.")
    print("Restart the exchange to load the new adapter.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"

    if cmd == "check":
        info = check_affordability()
        print(json.dumps(info, indent=2))
    elif cmd == "prepare":
        ledger = load_ledger()
        examples = prepare_training_data(ledger)
        count = save_training_data(examples)
        print(f"Saved {count} examples to {TRAINING_DATA_PATH}")
    elif cmd == "run":
        # Run via Modal
        print("Use: modal run finetune.py")
    else:
        print(f"Unknown command: {cmd}. Use 'check', 'prepare', or 'run'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
