# LM Studio

**What it is:** a desktop app for finding, downloading and chatting with local models. It also has a CLI (`lms`) and a headless server build (`llmster`) that serves an OpenAI-compatible API on port 1234. On Linux it runs GGUF models on a llama.cpp runtime; on Macs it also runs MLX.

**Why it matters:** it's closed source, so GitHub stars don't measure it, but it's one of the most discussed tools in r/LocalLLaMA (about 14.7k upvotes across the posts sampled) and on Hacker News (726 mentions in 12 months). NVIDIA has an official DGX Spark playbook for it.

**Best for:** a GUI for browsing and comparing models, and a no-sudo, no-Docker install. For learning internals it's the most opaque option here.

*Verified against the sources below on 2026-11-23.*

---

## How it works

```
LM Studio app (GUI)  or  lms CLI
        └─► llmster daemon (headless core) ──► llama.cpp runtime (CUDA) ──► GGUF models in ~/.lmstudio/models
HTTP (127.0.0.1:1234): /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models, /v1/responses
```

## Before you begin

- Linux ARM64 (aarch64) is supported.
- The headless install goes into your home directory (`~/.lmstudio/llmster`), so it should need neither `sudo` nor Docker.

## Install (headless, as in NVIDIA's DGX Spark playbook)

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
source ~/.bashrc          # if `lms` isn't found yet
lms daemon up
```

## Run a model

```bash
lms get llama-3.1-8b@q4_k_m        # download; the @ suffix picks the quantization
lms ls                             # list downloaded models
lms load llama-3.1-8b --context-length 8192 --gpu max
lms chat                           # interactive chat in the terminal
```

## Serve an OpenAI-compatible API

```bash
lms server start --port 1234       # add --bind 0.0.0.0 to expose it on the network
```

**curl:**

```bash
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "llama-3.1-8b",
  "messages": [{"role": "user", "content": "Say this is a test"}]
}'
```

**Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
resp = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(resp.choices[0].message.content)
```

With just-in-time (JIT) loading enabled, a request for a model that isn't loaded loads it automatically, and the model unloads again when idle.

## Key settings

| Option | What it controls |
|---|---|
| `lms load … --context-length N` | Context length |
| `lms load … --gpu max\|off\|0-1` | How much of the model goes on the GPU |
| `lms load … --ttl 3600` | Unload after N idle seconds |
| `lms load … --identifier name` | The model name the API reports |
| `lms load … --estimate-only` | Print the memory estimate without loading |
| `lms server start --bind / --port / --cors` | Server networking |

The CLI docs don't cover parallel slots or KV cache dtype.

## PGX / GB10 notes

- If you get `llama-server did not become healthy` (HTTP 503), the runtime is out of date. Fix it with `lms runtime upgrade`.
- NVIDIA's playbook example uses a large model that needs 65–70 GB free. That won't fit alongside the work services, so pick something smaller.

## Reference

- Docs: https://lmstudio.ai/docs
- System requirements: https://lmstudio.ai/docs/app/system-requirements
- Headless (llmster): https://lmstudio.ai/docs/developer/core/headless
- CLI reference: https://lmstudio.ai/docs/cli
- OpenAI compatibility: https://lmstudio.ai/docs/developer/openai-compat
- NVIDIA DGX Spark playbook: https://build.nvidia.com/spark/lm-studio
