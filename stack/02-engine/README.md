# Step 2: Engine

A from-scratch inference engine for one small model (Qwen3-0.6B, then 1.7B) in plain PyTorch.

Build order: load safetensors and config → tokenizer (via `tokenizers` at first) → embedding → RMSNorm → RoPE → grouped-query attention → SwiGLU MLP → full forward pass → greedy generation → KV cache → sampling (temperature, top-k, top-p).

- **Done when:** logits match HF `transformers` to within tolerance on the same input (layer by layer while debugging), and generation with the KV cache matches generation without it.
