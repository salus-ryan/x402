"""
x402 Exchange: Qwen3-8B served on Modal via vLLM.

The entity's brain. Serves an OpenAI-compatible API behind x402 payment gating.
Agents pay USDC per request; revenue funds compute for self-improvement.

Usage:
    modal deploy modal_app.py          # deploy persistent endpoint
    modal serve modal_app.py           # dev mode (hot-reload)
    modal run modal_app.py             # deploy + test

The endpoint is OpenAI-compatible:
    POST /v1/chat/completions
    GET  /health
    GET  /v1/models
"""

import json
import subprocess
import time

import modal

MINUTES = 60  # seconds

# ── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# ── Infrastructure ───────────────────────────────────────────────────────────

GPU = "L4"  # 24GB VRAM, $0.80/hr -- fits Qwen3-8B in bf16
PORT = 8000
REGION = "us-east"
MIN_CONTAINERS = 0  # scale to zero when idle; set to 1 for always-warm

# ── Image ────────────────────────────────────────────────────────────────────

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
    )
    .env({"TORCH_CPP_LOG_LEVEL": "FATAL"})
)

# ── Volumes ──────────────────────────────────────────────────────────────────
# Reuse existing cached weights from sal-model-cache if available,
# plus a dedicated volume for LoRA adapters produced by self-improvement.

hf_cache_vol = modal.Volume.from_name("sal-model-cache")
lora_vol = modal.Volume.from_name("x402-lora-adapters", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
LORA_PATH = "/lora"

vllm_image = vllm_image.env({"HF_HUB_CACHE": HF_CACHE_PATH})

# ── Helpers ──────────────────────────────────────────────────────────────────

with vllm_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if (rc := process.poll()) is not None:
                raise subprocess.CalledProcessError(rc, cmd=process.args)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(3)
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def warmup():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8,
    }
    for _ in range(2):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=30,
        ).raise_for_status()


# ── Server ───────────────────────────────────────────────────────────────────

APP_NAME = "x402-exchange"
app = modal.App(name=APP_NAME)


@app.server(
    image=vllm_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        LORA_PATH: lora_vol,
    },
    compute_region=REGION,
    min_containers=MIN_CONTAINERS,
    startup_timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
    port=PORT,
    routing_region=REGION,
    target_concurrency=20,
    unauthenticated=True,  # x402 handles auth via payment, not Modal auth
)
class Exchange:
    @modal.enter()
    def startup(self):
        """Start vLLM subprocess serving Qwen3-8B."""
        cmd = [
            "vllm", "serve",
            MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--uvicorn-log-level", "error",
            "--disable-uvicorn-access-log",
            "--disable-log-requests",
            "--enforce-eager",  # faster cold start on L4
            "--max-model-len", "8192",  # conservative for 24GB VRAM
            "--gpu-memory-utilization", "0.92",
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        print(f"x402 Exchange ready: {MODEL_NAME} on {GPU}")

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ── Test entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def test():
    import asyncio

    async def _test():
        url = await Exchange.get_url.aio()
        print(f"Exchange URL: {url}")

        import aiohttp

        # Wait for container to be ready (cold start)
        deadline = time.time() + 10 * MINUTES
        async with aiohttp.ClientSession(base_url=url) as session:
            while time.time() < deadline:
                try:
                    async with session.get("/health") as resp:
                        if resp.status == 200:
                            print(f"Health: {await resp.json()}")
                            break
                        print(f"Waiting for container... ({resp.status})")
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    print("Waiting for container...")
                await asyncio.sleep(5)

            # Chat completion
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are the x402 exchange -- an autonomous agent that earns revenue by serving inference."},
                    {"role": "user", "content": "What are you?"},
                ],
                "max_tokens": 128,
            }
            async with session.post(
                "/v1/chat/completions", json=payload
            ) as resp:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"\nResponse:\n{content}")
                print(f"\nUsage: {data['usage']}")

    asyncio.run(_test())
