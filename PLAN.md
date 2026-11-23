# Plan: local-model-maaxxing

A 12–18 month, build-first path from data/AI engineer to frontier-lab **Research Engineer / Member of Technical Staff** depth (Research Scientist kept open), run on a Lenovo ThinkStation PGX (GB10, 128 GB unified memory).

Agreed 2026-11-23 through a structured interview. Change it deliberately: edit this file and note why in `log/`.

## Goals

| | |
|---|---|
| Target role | Research Engineer / MTS; Research Scientist kept open |
| Payoff, now | The go-to person for local and private LLMs at work (day job can be used freely) |
| Payoff, later | A frontier lab role, or an own product |
| Starting level | 1–2 out of 5 on maths, raw PyTorch, transformer internals and paper reproduction |
| Time | 10–15 h/week, 12–18 months |
| Style | Build first, pair-building with Claude, theory as needed plus a scheduled maths thread |
| Visibility | Fully public: GitHub, write-ups, posts |

## Scope: T-shaped

- **Backbone:** optimization, generalization, scaling laws, evals.
- **Deep vertical:** LLMs (pretraining → post-training → inference).
- **Second pillar:** reinforcement learning (post-training, reasoning, agents).
- **Breadth modules:** diffusion and flow matching; state-space models and hybrids; multimodal.
- **Optional side branch:** recommender systems, when work calls for it.

## Specialization (provisional, confirm at month 6)

**Evals** as the entry point, growing into **post-training and RL**, with **interpretability** on the side.

## Weekly routine (~12 h)

| Block | Hours | What |
|---|---|---|
| Maths | 3 | 3Blue1Brown → Strang 18.06 → Stat 110 → matrix calculus (first 4–6 months) |
| Build | 6 | Pair-building with Claude; work projects count here |
| Paper | 2 | One paper a week, chosen from the feed and read closely |
| Log | 1 | Learning log in `log/` |

A public post every 2–3 weeks.

## Reinforcement, not gates

- **Flashcards:** after each session, Claude writes an Anki deck (`flashcards/`, imported with FSRS scheduling).
- **Break-it drills:** monthly, Claude hands over a subtly broken training run or eval to diagnose without help (`drills/`). They show where the gaps are and never block progress.
- Checkpoints test whether you can debug and modify things on your own, not verbatim recall.

## Checkpoints (time-boxed)

| Month | Checkpoint |
|---|---|
| 3 | GPT trained from scratch (nanoGPT, GPT-2-small reproduction) and every line explained; backprop through attention derived on paper |
| 6 | One paper replicated with a public write-up; work eval harness in use; specialization confirmed |
| 9 | One small original result; a multi-GPU training exercise (cloud, ~$50–150/month from month 6) |
| 12 | A research-style report (LessWrong or arXiv-quality); fellowship applications sent (MATS, Anthropic Fellows and similar) |

## Projects

1. **Work eval harness** (separate repo; build in days, results over weeks). Local models (gpt-oss-120b, Qwen, Gemma) against the API models in use today, on real work tasks: quality, latency, cost. Hosted securely for colleagues.
2. **Small fine-tune** (months 4–6). A small model that takes over one paid API job (extraction or classification), with a business case.
3. **Research feed** (week 2, separate repo). Runs weekly on the PGX with gpt-oss-120b.
   - Sources: arXiv (cs.LG, cs.CL, cs.AI), OpenReview (NeurIPS, ICML, ICLR), Hugging Face Daily Papers, lab blogs (Anthropic, transformer-circuits.pub, OpenAI, DeepMind), the Alignment Forum and LessWrong, r/MachineLearning, r/LocalLLaMA, Hacker News.
   - Ranking: relevance to this curriculum first, then social signals.
   - Output: an HTML digest plus an RSS feed, sorted into "intuition" and "go deeper", with same-day alerts for major releases.
   - Cross-source trend analysis.
   - **Summary faithfulness is evaluated**, which makes it the first evals research project.

## Community

EleutherAI and GPU MODE Discords, LessWrong and the Alignment Forum now; fellowships around months 9–12.

## First weeks

- **Day 1:** micrograd from scratch (`curriculum/00-foundations/`).
- **Days 2–4:** work eval harness.
- **Week 2:** research feed.

## Working setup

This cloud session connects to the PGX through Remote Control, so code runs on the PGX and the chat can be driven from any device. This repo carries the full context, so any session on any machine can pick it up.
