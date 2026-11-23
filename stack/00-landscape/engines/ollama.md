# Ollama

**What it is:** a model runner with a curated model registry (ollama.com/library), a one-line CLI (`ollama run <model>`), and a REST server on port 11434 with both its own API and OpenAI-compatible routes. Its supported backend is llama.cpp.

**Why it matters:** it's the most-starred local LLM project (about 188k GitHub stars) and the most-pulled image (`ollama/ollama`, about 204M Docker pulls). Many tools integrate with it by default.

**Best for:** getting a model running in one command, desktop use, and tools that expect an Ollama endpoint. It hides most tuning knobs, which is convenient for using models and a drawback for learning how they run.

*Verified against the sources below on 2026-11-23.*

---

## How it works

```
ollama CLI ──► ollama server (127.0.0.1:11434)
                 ├─ model store (~/.ollama or $OLLAMA_MODELS): registry blobs, GGUF-based
                 ├─ scheduler: loads/unloads models, keep-alive, parallel requests
                 └─ llama.cpp-based runner on CUDA
HTTP: /api/chat, /api/generate (native) · /v1/chat/completions, /v1/models, … (OpenAI-compatible)
```

## Before you begin

- `sudo` access. The official install creates a systemd service and an `ollama` user.
- Check whether it's already installed: `which ollama; systemctl status ollama`. One DGX OS user found it installed as a snap (`/snap/bin/ollama`).

## Install

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For a manual ARM64 install:

```bash
curl -fsSL https://ollama.com/download/ollama-linux-arm64.tar.zst | sudo tar x -C /usr
```

The manual install also needs a systemd unit; see the Linux install guide. To set environment variables for the service, run `sudo systemctl edit ollama` and add them under `[Service]` as `Environment="KEY=value"`.

## Run a model

```bash
ollama pull gpt-oss:20b
ollama run gpt-oss:20b "Explain the KV cache in one paragraph."
ollama ps        # what's loaded; PROCESSOR should say "100% GPU"
ollama stop gpt-oss:20b
```

To use your own GGUF file, write a `Modelfile` that points at it and run `ollama create -f Modelfile`.

## Serve an OpenAI-compatible API

The server already runs as a service after install. To run it by hand, use `ollama serve`.

**curl:**

```bash
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "Say this is a test"}]
}'
```

**Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")  # required by the client, ignored by Ollama
resp = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(resp.choices[0].message.content)
```

## Key settings (environment variables)

| Variable | What it controls | Notes |
|---|---|---|
| `OLLAMA_HOST` | Bind address | Default `127.0.0.1:11434`. |
| `OLLAMA_CONTEXT_LENGTH` | Default context length | Per request: `"options": {"num_ctx": 8192}`. The docs disagree on the default (4096 in the FAQ, VRAM-based elsewhere), so set it explicitly. |
| `OLLAMA_NUM_PARALLEL` | Concurrent requests per model | Memory grows with it: 4 parallel × 2K context = 8K of KV cache. |
| `OLLAMA_KV_CACHE_TYPE` | KV cache dtype: `f16`, `q8_0`, `q4_0` | Applies to all models. |
| `OLLAMA_FLASH_ATTENTION` | `1` or `0` | |
| `OLLAMA_KEEP_ALIVE` | How long an idle model stays loaded | |
| `OLLAMA_MAX_LOADED_MODELS` | Models resident at once | |
| `OLLAMA_MODELS` | Model storage directory | |

## PGX / GB10 notes

- NVIDIA has **no Ollama playbook** for DGX Spark. It only appears bundled inside the Open WebUI playbook, which needs Docker.
- **Community reports (not official):** several forum users say Ollama is slower than llama.cpp or LM Studio on Spark, and that the Docker image was slower than a native install. A mid-2026 regression (v0.30–v0.31.2) broke streaming for some large models on CUDA 13.
- Because it wraps llama.cpp, comparing `ollama` against a well-tuned `llama-server` on the same GGUF is a good step 1 exercise. It shows what the wrapper's defaults cost you.

## Reference

- Docs: https://docs.ollama.com
- Linux install: https://docs.ollama.com/linux
- OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- FAQ (environment variables): https://docs.ollama.com/faq
- Context length: https://docs.ollama.com/context-length
- GitHub: https://github.com/ollama/ollama
- GB10 performance thread: https://forums.developer.nvidia.com/t/very-poor-performance-with-ollama-on-dgx-spark-looking-for-help/353456
