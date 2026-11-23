"""Collect popularity signals for local LLM inference engines.

Sources (all public, no keys needed):
  - GitHub REST API: stars, forks, contributors, last push   (60 requests/hour unauthenticated)
  - pypistats.org:    PyPI downloads over the last month
  - Docker Hub:       total image pulls
  - Hacker News (Algolia): exact-phrase mentions in the last 12 months
  - Hugging Face:     number of models published in each weight format

Reddit (r/LocalLLaMA) is collected separately with Apify, see README.md.

Usage:  python3 collect.py            -> writes data/signals-<date>.json
"""

import datetime as dt
import json
import pathlib
import re
import time
import urllib.parse
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (local-llm-landscape research script)"}

# name -> where to look it up. None means "not published there".
ENGINES = {
    "llama.cpp":     dict(gh="ggml-org/llama.cpp",           pypi="llama-cpp-python",  docker=None,                 hn="llama.cpp"),
    "Ollama":        dict(gh="ollama/ollama",                 pypi="ollama",            docker="ollama/ollama",      hn="ollama"),
    "vLLM":          dict(gh="vllm-project/vllm",             pypi="vllm",              docker="vllm/vllm-openai",   hn="vllm"),
    "SGLang":        dict(gh="sgl-project/sglang",            pypi="sglang",            docker="lmsysorg/sglang",    hn="sglang"),
    "TensorRT-LLM":  dict(gh="NVIDIA/TensorRT-LLM",           pypi="tensorrt-llm",      docker=None,                 hn="tensorrt-llm"),
    "HF TGI":        dict(gh="huggingface/text-generation-inference", pypi="text-generation", docker=None,          hn="text-generation-inference"),
    "MLX-LM":        dict(gh="ml-explore/mlx-lm",             pypi="mlx-lm",            docker=None,                 hn="mlx"),
    "ExLlamaV2":     dict(gh="turboderp-org/exllamav2",       pypi="exllamav2",         docker=None,                 hn="exllama"),
    "ExLlamaV3":     dict(gh="turboderp-org/exllamav3",       pypi=None,                docker=None,                 hn=None),
    "LMDeploy":      dict(gh="InternLM/lmdeploy",             pypi="lmdeploy",          docker="openmmlab/lmdeploy", hn="lmdeploy"),
    "MLC LLM":       dict(gh="mlc-ai/mlc-llm",                pypi=None,                docker=None,                 hn="mlc-llm"),
    "llamafile":     dict(gh="mozilla-ai/llamafile",          pypi=None,                docker=None,                 hn="llamafile"),
    "LocalAI":       dict(gh="mudler/LocalAI",                pypi=None,                docker="localai/localai",    hn="localai"),
    "KoboldCpp":     dict(gh="LostRuins/koboldcpp",           pypi=None,                docker=None,                 hn="koboldcpp"),
    "text-generation-webui": dict(gh="oobabooga/text-generation-webui", pypi=None,      docker=None,                 hn="oobabooga"),
    "KTransformers": dict(gh="kvcache-ai/ktransformers",      pypi=None,                docker=None,                 hn="ktransformers"),
    "LM Studio (CLI)": dict(gh="lmstudio-ai/lms",             pypi="lmstudio",          docker=None,                 hn="lm studio"),
    "NVIDIA Dynamo": dict(gh="ai-dynamo/dynamo",              pypi="ai-dynamo",         docker=None,                 hn="nvidia dynamo"),
    "HF transformers (reference)": dict(gh="huggingface/transformers", pypi="transformers", docker=None,            hn=None),
}

# Hugging Face model counts per weight format: which engines can load it.
HF_FORMATS = {
    "GGUF (llama.cpp, Ollama, LM Studio, KoboldCpp)": "library=gguf",
    "MLX (Apple MLX-LM, LM Studio on Mac)":           "library=mlx",
    "AWQ (vLLM, SGLang, TGI, LMDeploy)":              "other=awq",
    "GPTQ (vLLM, SGLang, TGI, ExLlama)":              "other=gptq",
    "EXL2 (ExLlamaV2)":                               "other=exl2",
    "EXL3 (ExLlamaV3)":                               "other=exl3",
}


def get(url, raw=False, tries=5):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
                return (body, r.headers) if raw else json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(10 * 2**attempt)  # rate-limited: back off 10s, 20s, 40s, ...


def github(repo):
    d = get(f"https://api.github.com/repos/{repo}")
    # One contributor per page, so the "last" page number in the Link header = contributor count.
    _, h = get(f"https://api.github.com/repos/{repo}/contributors?per_page=1&anon=1", raw=True)
    m = re.search(r'page=(\d+)>; rel="last"', h.get("Link", ""))
    return dict(
        repo=d["full_name"], stars=d["stargazers_count"], forks=d["forks_count"],
        contributors=int(m.group(1)) if m else None,
        created=d["created_at"][:10], last_push=d["pushed_at"][:10],
        archived=d["archived"], license=(d.get("license") or {}).get("spdx_id"),
    )


def pypi(pkg):
    return get(f"https://pypistats.org/api/packages/{pkg}/recent")["data"]["last_month"]


def docker(image):
    return get(f"https://hub.docker.com/v2/repositories/{image}/")["pull_count"]


def hn(phrase):
    since = int(time.time()) - 365 * 24 * 3600
    q = urllib.parse.urlencode({
        "query": f'"{phrase}"', "tags": "(story,comment)", "typoTolerance": "false",
        "numericFilters": f"created_at_i>{since}", "hitsPerPage": 0,
    })
    return get(f"https://hn.algolia.com/api/v1/search?{q}")["nbHits"]


def hf_count(query):
    html, _ = get(f"https://huggingface.co/models?{query}", raw=True)
    m = re.search(r"numTotalItems&quot;:(\d+)", html)
    return int(m.group(1)) if m else None


def safe(fn, arg, previous=None):
    """Call fn(arg), reusing a good value from an earlier run today and recording errors as text."""
    if arg is None:
        return None
    if previous is not None and not str(previous).startswith("error"):
        return previous
    try:
        return fn(arg)
    except Exception as e:  # keep going; record the failure instead of crashing
        return f"error: {e}"


def main():
    today = dt.date.today().isoformat()
    path = pathlib.Path(__file__).parent / "data" / f"signals-{today}.json"
    path.parent.mkdir(exist_ok=True)
    # Re-running on the same day only refetches what failed (GitHub allows 60 calls/hour).
    old = json.loads(path.read_text()) if path.exists() else {"engines": {}, "hf_formats": {}}

    out = {"collected": today, "engines": {}, "hf_formats": {}}
    for name, src in ENGINES.items():
        print("…", name)
        prev = old["engines"].get(name, {})
        out["engines"][name] = dict(
            github=safe(github, src["gh"], prev.get("github")),
            pypi_downloads_last_month=safe(pypi, src["pypi"], prev.get("pypi_downloads_last_month")),
            pypi_package=src["pypi"],
            docker_pulls_total=safe(docker, src["docker"], prev.get("docker_pulls_total")),
            docker_image=src["docker"],
            hn_mentions_12mo=safe(hn, src["hn"], prev.get("hn_mentions_12mo")),
            hn_phrase=src["hn"],
        )
    for label, q in HF_FORMATS.items():
        out["hf_formats"][label] = safe(hf_count, q, old["hf_formats"].get(label))

    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
