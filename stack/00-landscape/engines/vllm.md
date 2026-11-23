# vLLM

**What it is:** a high-throughput inference and serving engine for LLMs, written in Python and PyTorch with custom CUDA kernels. It serves Hugging Face models (safetensors, plus many quantized formats) behind an OpenAI-compatible API.

**Why it matters:** among the engines compared here, it has the most contributors (about 4,200), 2.8M PyPI downloads a month and 45M Docker pulls. It was also the engine mentioned most in r/LocalLLaMA posts about DGX Spark/GB10. It's the default choice for serving a team, and it's what your employer already runs on this PGX.

**Best for:** many concurrent users, full-precision or FP8/NVFP4 models, and production-style serving. For a single user with heavily quantized GGUF models, llama.cpp is usually lighter.

*Verified against the sources below on 2026-11-23. Latest PyPI release: 0.34.0, which includes an aarch64 wheel.*

---

## How it works

Two ideas make vLLM fast:

1. **PagedAttention.** The KV cache is split into fixed-size blocks, like pages in an operating system, instead of one big reserved buffer per request. Memory is allocated only as a sequence grows, so far more requests fit at once. Shared prompt prefixes can reuse the same blocks; prefix caching is on by default in V1.
2. **Continuous batching.** The scheduler adds new requests to the running batch at every decode step, instead of waiting for the whole batch to finish. The GPU stays busy, and short requests don't queue behind long ones.

```
HTTP (OpenAI API) ──► scheduler (continuous batching) ──► model runner (CUDA graphs, FlashInfer/FlashAttention kernels)
                            └── block manager: paged KV cache in a memory pool sized by --gpu-memory-utilization
```

## Before you begin

> **Important: this PGX is shared.** The work vLLM instances already claim most of the unified memory. Always set `--gpu-memory-utilization` explicitly (see Key settings). Check `free -g` first, and keep the combined shares of all instances well below 1.0.

- You don't have Docker access without `sudo`, so the pip install is the realistic path.
- The driver here is 580.x. That decides which containers will start; see the PGX notes.

## Install (pip, no Docker)

Use a virtual environment in your own directory:

```bash
uv venv ~/venvs/vllm --python 3.12 && source ~/venvs/vllm/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
```

Without `uv`, use `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130`.

vLLM's release wheels are built against **CUDA 13.0**, which matches the PGX. The install docs don't mention GB10 (`sm_121`) explicitly, so confirm it works with the quickstart below before relying on it.

## Run a model (offline, in Python)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", gpu_memory_utilization=0.15, max_model_len=4096)
params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
for out in llm.generate(["Explain the KV cache in one paragraph."], params):
    print(out.outputs[0].text)
```

## Serve an OpenAI-compatible API

```bash
vllm serve Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 8100 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 16384 \
    --max-num-seqs 4
```

The default port is 8000. Use another port, such as 8100, so you don't collide with anything already running.

**curl:**

```bash
curl http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "Say this is a test"}]
}'
```

**Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(resp.choices[0].message.content)
```

Other endpoints: `/v1/completions`, `/v1/models`, `/health`.

## Key settings

| Flag | What it controls | Notes |
|---|---|---|
| `--gpu-memory-utilization F` | Share of GPU-visible memory vLLM may claim, for weights and the KV pool | On GB10 this is a share of the **whole 128 GB unified pool**. It's your main cap; NVIDIA suggests starting at 0.7 on a dedicated Spark, and much lower here. |
| `--max-model-len N` | Maximum prompt + output tokens per request | Lower values leave more room for concurrent sequences. |
| `--max-num-seqs N` | Maximum sequences in flight | The vLLM DGX Spark blog found decode is bandwidth-bound above about 4 streams. |
| `--kv-cache-dtype fp8` | KV cache precision | Halves KV memory; check quality first. |
| `--quantization …` | Quantization method | Leave unset for pre-quantized checkpoints (AWQ, GPTQ, FP8, NVFP4); vLLM reads it from the config. |
| `--tensor-parallel-size N` | Split the model across GPUs | Always 1 on a single PGX. |

## PGX / GB10 notes

- **Container and driver mismatch.** NGC `nvcr.io/nvidia/vllm` images from 26.03 onwards need driver 595.58 or newer and refuse to start on driver 580. On 580, `nvcr.io/nvidia/vllm:26.02-py3` is the last that runs. This was confirmed by NVIDIA staff in an August 2026 forum thread. The pip wheel avoids the problem.
- **NVFP4.** This is Blackwell's native 4-bit format and the main reason GB10 exists. It was missing or unreliable through early 2026, a common complaint on r/LocalLLaMA. Recent vLLM builds validated for `sm_121` support it, so leave the attention backend on `auto`.
- **Reference numbers** from the vLLM DGX Spark blog: Nemotron-3-Super-120B NVFP4 decodes at about 23 tokens/s on one Spark; the first request takes about 25 s (JIT compilation); loading large weights took 10–15 minutes.
- On unified memory you can hit out-of-memory errors that look wrong. See the `drop_caches` note in [llama-cpp.md](llama-cpp.md#pgx--gb10-notes).

## Reference

- Docs: https://docs.vllm.ai
- GPU install: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- Quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- Quantization: https://docs.vllm.ai/en/latest/features/quantization/
- GitHub: https://github.com/vllm-project/vllm
- vLLM on DGX Spark (blog): https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- NVIDIA DGX Spark playbook: https://build.nvidia.com/spark/vllm (advanced: https://build.nvidia.com/spark/vllm/advanced)
- NVIDIA Spark inference recipes: https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html
- Driver 580 container issue: https://forums.developer.nvidia.com/t/vllm-26-04-py3-compatibility-mode-is-unavailable-on-driver-580-173-02-no-dgx-dashboard-path-to-595-58/379168
- NVFP4 on Spark status thread: https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069
- PagedAttention paper: https://arxiv.org/abs/2309.06180
