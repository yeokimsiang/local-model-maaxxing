"""Count inference-engine mentions in r/LocalLLaMA posts scraped with Apify.

Input:  data/reddit-localllama-*.json  (Apify automation-lab/reddit-scraper output:
        top posts of the past year for each search term; see README.md)
Output: data/reddit-summary-<date>.json, plus a printed table.

Bias to keep in mind: the pool was built by searching each engine's name (up to
100 posts per search), so raw post counts partly reflect that cap. Upvotes, and
mentions inside posts found by *neutral* searches ("inference engine",
"DGX Spark", "GB10"), are the fairer signals.
"""

import datetime as dt
import glob
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent

PATTERNS = {
    "llama.cpp":     r"llama\.?cpp|llama-server",
    "Ollama":        r"\bollama\b",
    "vLLM":          r"\bvllm\b",
    "SGLang":        r"\bsglang\b",
    "TensorRT-LLM":  r"tensorrt|\btrt-?llm\b",
    "ExLlama":       r"exllama|\bexl[23]\b|tabbyapi",
    "LM Studio":     r"\blm ?studio\b",
    "MLX":           r"\bmlx\b",
    "KoboldCpp":     r"koboldcpp",
    "llamafile":     r"llamafile",
    "LMDeploy":      r"lmdeploy",
    "KTransformers": r"ktransformers",
    "MLC LLM":       r"\bmlc[- ]?llm\b",
    "text-generation-webui": r"oobabooga|text-generation-webui",
    "LocalAI":       r"\blocalai\b",
    "HF TGI":        r"text-generation-inference|\btgi\b",
}
GB10 = re.compile(r"dgx ?spark|\bgb10\b|\bspark\b", re.I)
NEUTRAL = re.compile(r"inference engine|dgx ?spark|\bgb10\b", re.I)


def load_posts():
    posts = {}
    for f in sorted(glob.glob(str(HERE / "data" / "reddit-localllama-run*.json"))):
        for p in json.load(open(f)):
            posts[p["permalink"]] = p  # dedupe across runs
    return list(posts.values())


def main():
    posts = load_posts()
    text = lambda p: f"{p.get('title') or ''}\n{p.get('selfText') or ''}"
    rows = {}
    for name, pat in PATTERNS.items():
        rx = re.compile(pat, re.I)
        hits = [p for p in posts if rx.search(text(p))]
        neutral = [p for p in hits if NEUTRAL.search(text(p))]
        gb10 = [p for p in hits if GB10.search(p.get("title") or "")]
        rows[name] = dict(
            posts=len(hits),
            upvotes=sum(p.get("score") or 0 for p in hits),
            title_mentions=sum(bool(rx.search(p.get("title") or "")) for p in hits),
            in_neutral_posts=len(neutral),
            in_gb10_titled_posts=len(gb10),
        )
    n_gb10 = sum(bool(GB10.search(p.get("title") or "")) for p in posts)
    n_neutral = sum(bool(NEUTRAL.search(text(p))) for p in posts)
    summary = dict(collected=dt.date.today().isoformat(), total_posts=len(posts),
                   neutral_posts=n_neutral, gb10_titled_posts=n_gb10, engines=rows)
    (HERE / "data" / f"reddit-summary-{summary['collected']}.json").write_text(json.dumps(summary, indent=2))

    print(f"{len(posts)} unique posts, {n_neutral} neutral, {n_gb10} with Spark/GB10 in the title\n")
    print(f"{'engine':24}{'posts':>7}{'upvotes':>10}{'neutral':>9}{'GB10':>6}")
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["upvotes"]):
        print(f"{k:24}{v['posts']:>7}{v['upvotes']:>10}{v['in_neutral_posts']:>9}{v['in_gb10_titled_posts']:>6}")


if __name__ == "__main__":
    main()
