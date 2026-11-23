# Local LLM inference engines: the landscape

Which tools people actually use to run LLMs on their own hardware, how popular each one really is (measured, not guessed), and how to use the ones that matter on the PGX.

*Data collected 2026-11-23. Re-run with `python3 collect.py` and `python3 analyze_reddit.py`; see [Method](#method).*

## Vocabulary

| Term | Meaning |
|---|---|
| **Inference** | Running a trained model to produce outputs, as opposed to training it. |
| **Inference engine / runtime** | Software that loads weights and runs the forward pass efficiently (kernels, memory, KV cache). |
| **Inference server / serving framework** | An engine plus an HTTP API, request scheduling and batching, for many users. |
| **Wrapper / front end** | A friendlier layer on top of an engine (Ollama, LM Studio). |
| **Model format** | How the weights are stored: safetensors (the Hugging Face standard), GGUF (llama.cpp), MLX (Apple), EXL2/EXL3 (ExLlama). |

Nearly all of them expose the same **OpenAI-compatible API** (`/v1/chat/completions`). Client code written for one works against any of them by changing `base_url`.

## The ranking

| Tier | Engine | Type | GitHub stars | Contributors | PyPI/month | Docker pulls | HN (12 mo) | Reddit upvotes¹ | Guide |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **1** | **llama.cpp** | Engine + basic server | 136k | 2,295 | 582k² | — | 2,381 | **48.6k** | [llama-cpp.md](engines/llama-cpp.md) |
| **1** | **Ollama** | Wrapper (llama.cpp) | **188k** | 630 | 12.5M² | **204M** | **2,614** | 16.4k | [ollama.md](engines/ollama.md) |
| **1** | **vLLM** | Serving engine | 97k | **4,178** | **2.76M** | 45M | 1,196 | 19.2k | [vllm.md](engines/vllm.md) |
| 2 | **SGLang** | Serving engine | 39k | 2,335 | 661k | 16.6M | 225 | 3.8k | [sglang.md](engines/sglang.md) |
| 2 | **LM Studio** | Desktop app + server (closed source) | 5k³ | — | 142k | — | 726 | 14.7k | [lm-studio.md](engines/lm-studio.md) |
| 2 | MLX-LM | Engine (Apple silicon only) | 8k | 267 | 592k | — | 866⁴ | 8.5k | n/a on the PGX |
| 3 | TensorRT-LLM | Engine + server (NVIDIA) | 15k | 533 | 6k⁵ | — | 24 | 0.5k | [tensorrt-llm.md](engines/tensorrt-llm.md) |
| 3 | KoboldCpp | llama.cpp fork + UI | 12k | —⁶ | — | — | 23 | 3.5k | — |
| 3 | ExLlamaV2 / V3 | Engine (EXL2/EXL3) | 5k / 2k | 57 / 105 | 2k | — | 4 | 2.0k | — |
| 3 | LocalAI | Wrapper over many backends | 51k | 292 | — | 11M | 28 | 0.2k | — |
| 3 | text-generation-webui | Web UI over many backends | 48k | 393 | — | — | 10 | 1.1k | — |
| 3 | llamafile | Single-file llama.cpp builds | 27k | 80 | — | — | 46 | 0.6k | — |
| 3 | KTransformers | CPU/GPU hybrid for huge MoE models | 20k | 135 | — | — | 8 | 0.2k | — |
| 3 | LMDeploy, MLC LLM, HF TGI | Engines | 8k / 24k / 11k | | 50k / — / 41k | 0.6M / — / — | ≤2 | <0.2k | — |

**Reference implementation (not an engine):** Hugging Face `transformers` has 169k stars, 4,274 contributors and 119M PyPI downloads a month. It's slow, but it's the readable ground truth. Step 2's engine is checked against it.

Notes:
1. Upvotes summed over the past year's top r/LocalLLaMA posts that mention the engine (935 unique posts; see Method).
2. These are the Python *client* packages (`llama-cpp-python`, `ollama`), not the engines themselves. Treat them as adoption proxies.
3. Stars for the open-source `lms` CLI only; the app itself is closed source.
4. "mlx" also matches general Apple MLX discussion.
5. Undercounted: NVIDIA distributes it mainly through NGC containers and its own package index.
6. KoboldCpp inherits llama.cpp's contributor history, so its count isn't meaningful.

### What the numbers say

- **Three tools dominate, each for a different job.** Ollama is the most-installed. llama.cpp is the engine that community discussion and model publishing revolve around: GGUF has about **225k models** on Hugging Face, against about 30k MLX, 9k GPTQ, 8.5k AWQ and 4.7k EXL2. vLLM is where the engineering effort goes, with the most contributors and the most downloads of any serving engine.
- **SGLang is the clear number 4 and still rising.** It's a serving engine in the same class as vLLM.
- **On GB10/DGX Spark hardware like yours, vLLM leads.** Among r/LocalLLaMA posts with Spark or GB10 in the title, vLLM was mentioned in 24, llama.cpp in 10, SGLang in 5, TensorRT-LLM and MLX in 3 each, and Ollama in 1. Spark owners run serving engines on it far more than desktop wrappers.
- **TensorRT-LLM is NVIDIA's engine but has little community use**, and its GB10 support is currently weak. Its DGX Spark playbook has been withdrawn.
- **Fading:** HF TGI (no commits since March 2026), ExLlamaV2 (superseded by V3) and text-generation-webui (last commit August 2026).

## Which to use on the PGX

| If you want… | Use | Why |
|---|---|---|
| To understand how a lean engine works, with quantized models and one user | **llama.cpp** (built from source) | Readable C++, every knob exposed, GGUF quants from 2 to 8 bit |
| The easiest possible start | **Ollama** or **LM Studio** | One command, but the details are hidden |
| To serve a team or an application, with high throughput, FP8/NVFP4 and concurrency | **vLLM**, or **SGLang** for prefix-heavy and agent workloads | PagedAttention or RadixAttention, continuous batching |
| NVIDIA's latest kernels | TensorRT-LLM | Only once its GB10 support matures |

For step 1 of this repo, benchmark **llama.cpp** and **vLLM** on the same model. That pair shows the latency-versus-throughput trade-off most clearly, and they're the two leaders on GB10.

## PGX / GB10 issues that affect every engine

- **Shared memory.** Work vLLM services already use most of the 128 GB unified pool. Every engine needs an explicit memory cap: `--gpu-memory-utilization` (vLLM), `--mem-fraction-static` (SGLang), `--kv_cache_free_gpu_memory_fraction` (TensorRT-LLM), `-c` plus the choice of quant (llama.cpp).
- **`sm_121` is unusual.** GB10 isn't datacenter Blackwell (`sm_100`) or consumer Blackwell (`sm_120`). Some software has no kernels for it, or falls back to slower code paths. This is the most common complaint in the Spark threads on r/LocalLLaMA.
- **Driver 580 versus newer containers.** NGC containers built on CUDA 13.2 or later need driver 595 or newer and won't start here. Prefer pip installs, or older container tags.
- **No Docker without `sudo`.** pip/uv installs and source builds avoid it.
- **`nvidia-smi` shows memory as `N/A`.** Use `free -g`.
- **Unified-memory page cache.** File cache can hold on to memory. NVIDIA's fix is `sync; echo 3 > /proc/sys/vm/drop_caches` with `sudo`; coordinate before doing that on a shared box.

## Workarounds on the PGX

**Why containers fail.** A container ships its own CUDA *runtime* but always uses the host's *driver*. Driver 580 supports CUDA up to **13.0** (`nvidia-smi` shows `CUDA Version: 13.0`). NGC images built on CUDA 13.2 or later need driver 595 or newer, and GB10 doesn't support the forward-compatibility package that would normally bridge the gap. So the image stops at startup with *"compatibility mode is UNAVAILABLE"*.

| Problem | Workaround | Notes |
|---|---|---|
| NGC containers newer than 26.02 won't start | **pip-install the cu130 wheel in a venv** (vLLM, SGLang) | The versions match exactly: wheel CUDA 13.0, driver ceiling 13.0. No root needed. |
| | Fallback: `nvcr.io/nvidia/vllm:26.02-py3` | Pins an older vLLM (0.15). |
| | Avoid: images that set `NVIDIA_DISABLE_REQUIRE=1` | That only skips the check; the image can still crash at runtime. |
| | Upgrade the driver to 595 or newer | Needs a reboot and can break the work services. Only with their owners, through the vendor's update path. |
| No Docker access | `sudo docker …`, or `sudo usermod -aG docker $USER` and log in again | Being in the `docker` group is effectively root. Mostly unnecessary if you use pip. |
| llama.cpp has no GB10 kernels, or is slow | `-DCMAKE_CUDA_ARCHITECTURES=121a-real` | `121a` means compute capability 12.1 with architecture-specific instructions (FP4 tensor cores); `real` means emit machine code (SASS), not PTX. |
| llama.cpp port changed | Always pass `--port` | The default is now 9931. |
| TensorRT-LLM is immature on GB10 | Skip it for now; vLLM and SGLang share many kernels (CUTLASS, FlashInfer) and load NVIDIA's NVFP4 checkpoints | Try its container with `sudo docker` in step 3 if curious. |
| Work services hold most of the memory | Explicit caps (`--gpu-memory-utilization 0.2–0.25`, short `--max-model-len`, few sequences), 8B-class or quantized models, FP8/q8 KV cache, `watch -n1 free -g` while loading | If memory overflows, the OOM killer may kill **the work vLLM**, not yours. vLLM *reserves* its share at startup, so ask the owners whether theirs can shrink. |
| | Untested: hard-cap your own process with `systemd-run --user --scope -p MemoryMax=30G <cmd>` | It's unclear whether GPU allocations on unified memory count towards a cgroup limit; see `questions.md`. |

## Method

All the raw data is in [`data/`](data/).

| Signal | Source | What it measures | Bias |
|---|---|---|---|
| GitHub stars, forks, contributors, last push | GitHub REST API | Interest, community size, whether it's maintained | Stars accumulate over years and measure hype as much as use |
| PyPI downloads, last month | pypistats.org | Installs from Python, including CI | CI inflates counts; client packages aren't engines; NVIDIA ships outside PyPI |
| Docker pulls, all-time | Docker Hub API | Deployments | Only for images on Docker Hub (not ghcr.io or NGC) |
| Hacker News mentions, 12 months | Algolia HN API, exact phrase | Developer mindshare | Generic names ("mlx") over-match |
| Hugging Face model counts per format | huggingface.co model listing | Size of each format's ecosystem | Counts repos, not downloads |
| r/LocalLLaMA posts | Apify `automation-lab/reddit-scraper`: top posts of the past year for 19 search terms, 100 per term | What local-LLM practitioners actually discuss | The pool was built by searching each engine's name, so post counts partly reflect that per-search cap. Upvotes, and mentions inside *neutral* searches ("inference engine", "DGX Spark", "GB10"), are fairer |

No single signal is trustworthy alone. The tiers above come from agreement across them. The usage guides in [`engines/`](engines/) were checked against official docs, NVIDIA's DGX Spark playbooks and the NVIDIA developer forums on 2026-11-23. Each guide links its sources.
