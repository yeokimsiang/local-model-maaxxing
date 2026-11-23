# Step 1: Run

Run open models on the PGX with **llama.cpp** (GGUF) and **vLLM**, then measure them.

- Metrics: prefill tokens/s, decode tokens/s, time to first token, peak memory.
- Models: 2–3 of different sizes (a dense ~1–8B model, plus a larger MoE such as gpt-oss-120b if memory allows alongside the work services).
- **Done when:** the benchmark table is filled in, and each decode number is checked against the bandwidth limit (decode tokens/s ≲ 273 GB/s ÷ bytes of weights read per token).
