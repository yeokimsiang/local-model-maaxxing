# SGLang

**What it is:** a high-performance serving framework for LLMs and vision-language models. It's the closest alternative to vLLM, with a similar Python and CUDA design and an OpenAI-compatible API.

**Why it matters:** it's the fastest-rising serving engine. It has about 39k GitHub stars, about 2,300 contributors, 660k PyPI downloads a month and 16.6M Docker pulls. Large-scale deployments use it, and NVIDIA has a DGX Spark playbook for it.

**Best for:** workloads that reuse prompt prefixes heavily (agents, multi-turn chat, few-shot prompts, structured output), and throughput-focused serving.

*Verified against the sources below on 2026-11-23. Latest PyPI release: 0.5.23, which includes aarch64 wheels and requires CUDA 13.*

---

## How it works

Its key idea is **RadixAttention**. The KV caches of past requests are kept in a *radix tree* keyed by token sequence. When a new request shares a prefix with an earlier one (the same system prompt, earlier turns of the chat, the same few-shot examples), the cached KV for that prefix is reused instead of recomputed. vLLM's prefix caching does something similar using hashed blocks.

```
HTTP ──► scheduler (continuous batching, cache-aware ordering) ──► model runner (FlashInfer kernels, CUDA graphs)
            └── radix-tree KV cache in a pool sized by --mem-fraction-static
```

## Before you begin

- The same shared-memory warning as [vLLM](vllm.md#before-you-begin) applies. Set `--mem-fraction-static` explicitly.
- Without Docker access, use the pip or uv install.

## Install (pip, no Docker)

```bash
uv venv ~/venvs/sglang --python 3.12 && source ~/venvs/sglang/bin/activate
uv pip install --prerelease=allow sglang
```

The install docs don't mention GB10 explicitly. NVIDIA's playbook uses the container `lmsysorg/sglang:latest-cu130`, which needs `sudo docker` here.

## Run a model (offline, in Python)

```python
import sglang as sgl

if __name__ == "__main__":  # SGLang starts worker processes, so keep the guard
    llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B", mem_fraction_static=0.15)
    outputs = llm.generate(["Explain the KV cache in one paragraph."], {"temperature": 0.7, "max_new_tokens": 128})
    print(outputs[0]["text"])
    llm.shutdown()
```

## Serve an OpenAI-compatible API

```bash
sglang serve --model-path Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 30000 \
    --mem-fraction-static 0.25 \
    --context-length 16384
```

The older form `python3 -m sglang.launch_server …` still works. The default port is 30000.

**curl:**

```bash
curl http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "Say this is a test"}]
}'
```

**Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(resp.choices[0].message.content)
```

There's also a native `POST /generate` endpoint.

## Key settings

| Flag | What it controls | Notes |
|---|---|---|
| `--mem-fraction-static F` | Share of memory for weights plus the KV pool | Your main cap. NVIDIA suggests 0.7–0.75 on a dedicated Spark; go much lower here. |
| `--context-length N` | Maximum context length | Defaults to the model's config. |
| `--max-running-requests N` | Concurrent requests | |
| `--kv-cache-dtype` | `auto`, `fp8_e4m3`, `fp8_e5m2`, `bf16`, … | |
| `--quantization` | `awq`, `gptq`, `fp8`, `gguf`, `modelopt_fp4`, `mxfp4`, … | Use `modelopt_fp4` for NVIDIA NVFP4 checkpoints such as `nvidia/Qwen3-32B-FP4`. |
| `--attention-backend` | Kernel family | NVIDIA's playbook uses `flashinfer`. |
| `--tp-size N` | Tensor parallelism | 1 on a single PGX. |

## PGX / GB10 notes (from NVIDIA's troubleshooting page)

- **Out-of-memory "within capacity"** happens on unified memory. The fix is `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, which is disruptive on a shared box.
- `nvidia-smi --query-gpu=memory.*` shows `N/A` on GB10. That's expected with unified memory; use plain `nvidia-smi` or `free -g`.
- **The first launch is slow** because of CUDA-graph capture; large MoE models can take 45–60 minutes.

## Reference

- Docs: https://docs.sglang.io
- Install: https://docs.sglang.io/get_started/install.html
- Offline engine API: https://docs.sglang.io/basic_usage/offline_engine_api.html
- Server arguments: https://docs.sglang.io/advanced_features/server_arguments.html
- GitHub: https://github.com/sgl-project/sglang
- NVIDIA DGX Spark playbook: https://build.nvidia.com/spark/sglang (and its [troubleshooting page](https://build.nvidia.com/spark/sglang/troubleshooting.md))
- SGLang paper (RadixAttention): https://arxiv.org/abs/2312.07104
