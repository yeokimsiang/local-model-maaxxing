# llama.cpp

**What it is:** a C/C++ inference engine built on the `ggml` tensor library. It runs **GGUF** models on CUDA, Metal, Vulkan, CPU and more, and ships `llama-server`, an HTTP server with an OpenAI-compatible API.

**Why it matters:** it's the engine underneath much of the local ecosystem. Ollama, LM Studio, KoboldCpp and llamafile all build on it, and GGUF is by far the most published local format (about 225,000 models on Hugging Face as of 2026-11-23).

**Best for:** one user or a few users, quantized models (2–8 bit), fast iteration, and lean binaries with no Python.

*Verified against the sources below on 2026-11-23. llama.cpp moves daily, so check `llama-server --help` on your build before trusting any flag here.*

---

## How it works

```
GGUF file (weights + tokenizer + metadata, usually quantized)
   │  loaded via mmap or read
   ▼
ggml compute graph ──► backend kernels (CUDA on the PGX)
   ▼
llama-server: N parallel "slots" (one KV cache region each), continuous batching
   ▼
HTTP: /v1/chat/completions, /v1/completions, /v1/embeddings, /health
```

## Before you begin

- `nvcc` on your `PATH`. If CMake says CUDA isn't found: `export PATH=/usr/local/cuda/bin:$PATH`.
- CMake 3.18+, git, a C++ compiler, and libcurl/libssl headers. Check with `dpkg -l | grep -E "libcurl4-openssl-dev|libssl-dev"`. Installing them needs `sudo`.
- About 30 GB of free memory for a mid-size quantized model. Check `free -g` first, because work services share this machine.

## Install

### Option A: build from source (recommended on the PGX)

This is NVIDIA's DGX Spark playbook build. GB10 is compute capability 12.1, and the architecture string is **`121a-real`**, not plain `121`.

```bash
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev   # skip if already installed
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON \
      -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release -j          # add --target llama-server to build just the server
```

Binaries land in `build/bin/` (`llama-server`, `llama-cli`, `llama-bench`, …).

### Option B: prebuilt binaries

The numbered nightly releases (`bNNNNN`) include `llama-bNNNNN-bin-ubuntu-cuda-13.5-arm64.tar.gz`, plus a separate `cudart-…` tarball. They're built against CUDA 13.5 while the PGX has 13.0 with driver 580, so test before relying on them.

### Option C: the unified `llama` binary (v1.0.0+)

Since v1.0.0 (2026-11-13) there's a single `llama` command with an installer:

```bash
curl -LsSf https://llama.app/install.sh | sh
llama cli   -hf ggml-org/Qwen3.5-0.8B-GGUF
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

GB10 isn't documented for this path yet. NVIDIA's playbook still uses Option A.

## Run a model

`-hf` downloads a GGUF from Hugging Face into `~/.cache/huggingface/hub`. It defaults to the `Q4_K_M` quantization; pick another with `:QUANT`.

```bash
./build/bin/llama-cli -hf ggml-org/Qwen3.5-0.8B-GGUF -p "Explain the KV cache in one paragraph."
```

## Serve an OpenAI-compatible API

```bash
./build/bin/llama-server -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
    --host 127.0.0.1 --port 30000 -c 32768
```

> **Note:** the default port is now **9931**. Older guides say 8080. Bind to `127.0.0.1` unless you mean to expose it on the network.

**curl:**

```bash
curl http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "New York is a great city because..."}],
  "max_tokens": 100
}'
```

**Python** (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")  # key ignored unless --api-key is set
resp = client.chat.completions.create(
    model="local",  # llama-server serves one model; set its name with --alias
    messages=[{"role": "user", "content": "Explain RoPE in two sentences."}],
)
print(resp.choices[0].message.content)
```

## Key settings

| Flag | What it controls | Notes |
|---|---|---|
| `-c, --ctx-size N` | Context length in tokens | `0` uses the model's maximum. The KV cache grows linearly with it. |
| `-ngl, --gpu-layers N\|auto\|all` | Layers placed on the GPU | Default `auto`. On unified memory, use `all`. |
| `-np, --parallel N` | Parallel slots (concurrent requests) | The context is divided across slots unless you use `--kv-unified`. |
| `-ctk / -ctv` | KV cache dtype (`f16`, `bf16`, `q8_0`, `q4_0`, …) | `q8_0` halves KV memory with little quality loss. |
| `-fa, --flash-attn on\|off\|auto` | Flash attention kernel | |
| `-fit on\|off` | Auto-shrink unset settings to fit memory | Default `on`. |
| `-lm, --load-mode` | `auto`, `none`, `mmap`, `mlock`, … | Replaces the old `--no-mmap`. See the PGX notes below. |

Every flag also has an environment variable (`LLAMA_ARG_*`).

## PGX / GB10 notes

- **Model loading is slow with mmap on GB10.** NVIDIA forum measurements for gpt-oss-120b were about 1m48s with mmap against about 22.5s without. If the model fits in free memory, use `--load-mode none`.
- **Unified memory and the page cache.** The OS file cache can hold on to memory after a model is unloaded. NVIDIA's fix is `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`. Check with the work services' owners before running it on a shared box.
- **Slow generation** usually means some layers fell back to the CPU. Check the startup log and use `-ngl all`.

## Reference

- GitHub: https://github.com/ggml-org/llama.cpp
- Build guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Server and API reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Releases: https://github.com/ggml-org/llama.cpp/releases
- NVIDIA DGX Spark playbook: https://build.nvidia.com/spark/llama-cpp (and its [troubleshooting page](https://build.nvidia.com/spark/llama-cpp/troubleshooting.md))
- mmap on GB10 (forum): https://forums.developer.nvidia.com/t/very-slow-mmap-on-dgx-spark-that-affects-model-loading-questions-to-nvidia/349886
