# Step 3: Faster

Make the step 2 engine faster and measure what each change costs in quality.

- Weight quantization: int8, then int4 (group-wise), then NVFP4, which Blackwell runs natively.
- Batching several requests at once; a first look at paged KV cache.
- Profiling with `torch.profiler` and Nsight Systems.
- **Done when:** each change has measured speed and perplexity, and the remaining gap to vLLM is explained.
