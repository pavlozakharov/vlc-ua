"""Probe whether a chat endpoint returns usable logprobs.

Provider quota, missing logprobs and a wrong model name all look alike from
the outside: an empty or odd answer. This probe separates them before a
single gold row is spent, and prints one JSON object the handoff report can
paste as is.
"""
from __future__ import annotations

import json
import time

from .backends.logprob import LogprobJudge
from .types import Choice


def probe(base_url: str, model: str, api_key_env: str = "LLM_API_KEY", api_key: str | None = None) -> dict:
    judge = LogprobJudge(base_url=base_url, model=model, api_key_env=api_key_env, api_key=api_key)
    q = Choice(instructions="Яка з літер стоїть в алфавіті першою?",
               criteria={"а": "перша літера", "я": "остання літера", "м": "середина алфавіту"})
    t0 = time.time()
    try:
        scores, degraded = judge.raw_scores("Простий тест.", q)
    except Exception as exc:
        return {"base_url": base_url, "model": model, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": time.time() - t0}
    top = max(scores, key=scores.__getitem__)
    return {"base_url": base_url, "model": model, "ok": True, "logprobs": not degraded,
            "top": top, "scores": scores, "seconds": time.time() - t0,
            "verdict": ("usable: real distribution" if not degraded else
                        "degraded: endpoint ignores logprobs; use only as a one-hot teacher")}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="vlc-judge probe-logprobs")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    a = ap.parse_args(argv)
    print(json.dumps(probe(a.base_url, a.model, a.api_key_env), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
