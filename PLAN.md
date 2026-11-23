# Plan: local-model-maaxxing

Run, understand and rebuild a local LLM stack **on the PGX itself** (Lenovo ThinkStation PGX: GB10 Grace Blackwell, 128 GB unified memory, ARM64, CUDA 13, ~273 GB/s memory bandwidth).

This is the **stack track**. The concepts track (maths, micrograd → GPT, RL, papers) lives in its own repo, `research-engineer-curriculum`, and starts once this track is done.

First agreed 2026-11-23, and split into two tracks the same day (see `log/2026-11-23.md`). Change it deliberately: edit this file and note why in `log/`.

## Why two tracks, in this order

Learning concepts alongside the build slows the build down. Building the whole stack first gives a bird's-eye view, so later study can answer questions that span the system ("why is decode memory-bound here?"), not just narrow ones. Running a model also needs no gradients, so the foundations (autograd, backprop) aren't on this path.

The risk is a stack you've **configured but don't understand**. This is guarded against by step 2: you write the forward pass yourself and prove it matches a reference implementation.

## Step 0: landscape

`stack/00-landscape/` surveys the local inference engines: measured popularity (GitHub, PyPI, Docker, Hacker News, Hugging Face, r/LocalLLaMA) and usage guides for the leading ones, with PGX notes.

## The four steps

| Step | Folder | What | Done when |
|---|---|---|---|
| 1. Run | `stack/01-run/` | Run open models on the PGX with llama.cpp and vLLM. Measure prefill and decode tokens/s, memory and time to first token. | A benchmark table for 2–3 models × 2 runtimes, with each number explained in terms of bandwidth and compute. |
| 2. Engine | `stack/02-engine/` | A from-scratch inference engine for one small model (Qwen3-0.6B or 1.7B) in plain PyTorch: safetensors loading, tokenizer, RMSNorm, RoPE, GQA attention, SwiGLU, KV cache, sampling. | Logits match HF `transformers` to within tolerance, and generation works end to end. |
| 3. Faster | `stack/03-faster/` | Quantization (int8, int4, then NVFP4, which GB10 runs natively), batching, and profiling. Compare against vLLM. | Measured speed and quality (perplexity) at each step, and the gap to vLLM explained. |
| 4. Serve | `stack/04-serve/` | An OpenAI-compatible API in front of your engine (and vLLM), with streaming and concurrent requests. | The work eval harness can call it. |

Rough timing at 10–15 h/week: step 1 ~1 week, step 2 ~3–4 weeks, step 3 ~2–3 weeks, step 4 ~1 week. That's about two months.

## Feeding the concepts track

Every "why?" you hit goes in [`questions.md`](questions.md), with the step it came from. Don't stop to answer it unless it blocks the build. That list becomes the starting syllabus of the concepts track.

## PGX constraints

- **Shared machine.** Work vLLM services already run here and use most of the memory (about 44 GB was free on 2026-11-23). Don't stop or reconfigure them. Check `free -g` before loading a model, and cap vLLM with `--gpu-memory-utilization`.
- **Unified memory.** CPU and GPU share the same 128 GB, so "VRAM" and "RAM" are one budget.
- **ARM64 + CUDA 13 (sm_121).** Some wheels and prebuilt binaries don't exist for this platform. Prefer NVIDIA's aarch64 builds and containers, or build from source.
- Keep model weights in your own cache (e.g. `~/models` or `~/.cache/huggingface`), not in the work directories under `/opt`.

## Related projects (separate repos)

1. **Work eval harness.** Local models against the API models in use today, on real work tasks (quality, latency, cost). Step 4 plugs into it.
2. **Research feed.** A weekly digest run on the PGX with a local model. It will run on the stack built here.

## Reinforcement

Flashcards after each session (`flashcards/`) and the learning log (`log/`) continue as before. Break-it drills (`drills/`) now target the stack: a broken RoPE, a stale KV cache, a mis-scaled quantization.
