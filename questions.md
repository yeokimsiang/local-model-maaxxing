# Open questions

"Why?" questions hit during the build. Note them, keep building, and answer them in the concepts track (`research-engineer-curriculum`).

Format: `- [step] question` (add a note on where it came up if that helps).

- [0] How exactly do the CUDA driver and runtime versions interact, and what does NVIDIA's "forward compatibility" package do that GB10 can't use?
- [0] What do `sm_121`, the `a` suffix and `-real` mean, and how do SASS and PTX differ?
- [0] On GB10 unified memory, do `cudaMalloc` allocations count towards a cgroup `MemoryMax`? (Test with a small model.) Which process does the OOM killer choose?
- [0] Why does vLLM reserve its memory share up front instead of growing on demand?
