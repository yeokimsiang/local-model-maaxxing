# Working agreement for Claude in this repo

Read `PLAN.md` first. It is the agreed plan and the source of truth.

This repo is the **stack track**: running, understanding and rebuilding a local LLM stack on the PGX. The concepts track lives in a separate repo (`research-engineer-curriculum`) and starts after this one is done.

## Who this is for

A data/AI engineer at 1–2 out of 5 on maths, raw PyTorch, transformer internals and paper reproduction, aiming for frontier-lab Research Engineer depth. Hardware: Lenovo ThinkStation PGX (GB10 Grace Blackwell, 128 GB unified memory, ARM64 Linux, CUDA 13, ~273 GB/s memory bandwidth).

## How to work together

- **Build first; keep theory minimal.** Explain only as much as the current step needs. When a deeper "why" comes up, add it to `questions.md` and keep building, unless it blocks progress.
- **Pair-build, explaining as you go.** Write the engine (step 2) from scratch before using libraries for that layer.
- **Guard against a false sense of competence.** Regularly ask them to predict an output (a shape, a tokens/s figure, a memory footprint), explain a line, or make the next change themselves.
- **Everything runs on the PGX.** Don't touch the work vLLM services or anything under `/opt`. Check free memory before loading models.
- **End every build session by:**
  1. writing flashcards to `flashcards/<date>-<topic>.tsv`, one card per line as `front<TAB>back`, focused on concepts, shapes, numbers and failure modes rather than trivia;
  2. appending a short entry to `log/<date>.md`: what was built, what clicked, what's still shaky;
  3. making sure new open questions are in `questions.md`.
- **Monthly break-it drills** go in `drills/`. Hand over the broken setup without the answer, and put the solution in a separate file.
- **Keep it public-ready.** Code and notes should be clean enough to share.
- **Work code stays at the employer.** Only shareable lessons come back here.

## Layout

- `stack/01-run/`: llama.cpp and vLLM on the PGX, with benchmarks.
- `stack/02-engine/`: the from-scratch inference engine.
- `stack/03-faster/`: quantization, batching, profiling.
- `stack/04-serve/`: the API server.
- `questions.md`: open "why?" questions, feeding the concepts track.
- `log/`, `flashcards/`, `drills/`: as above.
