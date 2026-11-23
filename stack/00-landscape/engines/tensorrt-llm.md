# TensorRT-LLM

**What it is:** NVIDIA's open-source library for LLM inference on NVIDIA GPUs, with a server, `trtllm-serve`, that exposes an OpenAI-compatible API.

> **Common misconception:** older guides describe TensorRT-LLM as *compiling a model into a TensorRT engine* before you can run it. That's now out of date. The TensorRT backend has been removed, and TensorRT-LLM is now a **PyTorch-native runtime** with NVIDIA's own attention, GEMM and MoE kernels, plus speculative decoding and disaggregated serving.

**Why it matters:** it's where NVIDIA's newest kernels and quantization formats (FP8, NVFP4) often appear first, and it underpins NVIDIA's NIM and Dynamo stacks. Community use is small, though: 15k stars, and r/LocalLLaMA mentions an order of magnitude below vLLM. Most of its users are enterprises.

**Best for:** squeezing maximum performance out of datacenter NVIDIA GPUs. **On GB10 today it's the least mature of the three serving engines**; see the PGX notes. Treat it as something to try out of curiosity in step 3, not as your main runtime.

*Verified against the sources below on 2026-11-23. Latest GA release: 1.3.0; nightly 1.4.0rc5.*

---

## How it works

```
HF or ModelOpt checkpoint (BF16 / FP8 / NVFP4)
   ▼
LLM API (PyTorch runtime) ──► NVIDIA kernels (attention, GEMM, MoE), CUDA graphs, in-flight batching
   ▼
trtllm-serve: /v1/chat/completions, /v1/completions, /v1/models, /health, /metrics
```

## Before you begin

- The pip install documents **CUDA Toolkit 13.1** and was tested on Ubuntu 24.04 with Python 3.12. The PGX has CUDA 13.0, and aarch64 isn't mentioned.
- The official container is `nvcr.io/nvidia/tensorrt-llm/release`, which needs `sudo docker` here and is subject to the same driver-version checks as the NGC vLLM images.

## Install

From the Linux install guide. On GB10 this route is unverified.

```bash
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm
```

PyPI has only a source distribution, with no prebuilt aarch64 wheel, so expect a long build or a failure. The container is the supported path.

## Run a model (offline, in Python)

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
params = SamplingParams(temperature=0.8, top_p=0.95)
for out in llm.generate(["Explain the KV cache in one paragraph."], params):
    print(out.outputs[0].text)
```

## Serve an OpenAI-compatible API

```bash
trtllm-serve "nvidia/Qwen3-8B-FP8" --host 127.0.0.1 --port 8200 \
    --kv_cache_free_gpu_memory_fraction 0.3
```

**curl:**

```bash
curl http://127.0.0.1:8200/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nvidia/Qwen3-8B-FP8",
  "messages": [{"role": "user", "content": "Say this is a test"}]
}'
```

The same Python `openai` client pattern as [vLLM](vllm.md#serve-an-openai-compatible-api) works, with `base_url="http://127.0.0.1:8200/v1"`.

## Key settings

| Flag | What it controls | Notes |
|---|---|---|
| `--kv_cache_free_gpu_memory_fraction F` | Share of the memory left *after loading weights* that goes to the KV cache | Unlike vLLM, it's relative to free memory, so on a shared box it depends on what else is running. |
| `--max_seq_len N` | Maximum sequence length | |
| `--max_batch_size N` / `--max_num_tokens N` | Batching limits | |
| `--tp_size N` | Tensor parallelism | 1 on a single PGX. |
| `--config file.yaml` | Advanced options, e.g. `kv_cache_config` | Replaces `--extra_llm_api_options`. |

## PGX / GB10 notes

- **NVIDIA's DGX Spark playbook for TensorRT-LLM has been withdrawn.** Its URL now redirects to GitHub.
- **`sm_121` (GB10) is missing from the quantization support matrix**, which lists `sm_120` with NVFP4, MXFP4 and FP8.
- A March 2026 forum report: Qwen3-Next-80B in NVFP4 crashed with "illegal instruction" on GB10, and FP8 failed in the autotuner. The user fell back to vLLM.

## Reference

- Docs: https://nvidia.github.io/TensorRT-LLM
- Linux install: https://nvidia.github.io/TensorRT-LLM/installation/linux.html
- Quick start: https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
- `trtllm-serve` reference: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- Quantization support matrix: https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- GitHub: https://github.com/NVIDIA/TensorRT-LLM
- Container: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release
- GB10 failure report: https://forums.developer.nvidia.com/t/issue-qwen3-next-80b-nvfp4-and-fp8-cannot-be-served-via-trtllm-serve-on-dgx-spark-gb10-trt-llm-1-3-0rc7/363540
