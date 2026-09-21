"""Command line: build gold, run a backend, report, serve.

    vlc-judge gold-attribution --texts DIR_OR_SQLITE --out gold/attribution.jsonl
    vlc-judge task attribution > gold/attribution.task.json
    vlc-judge run --task gold/attribution.task.json --gold gold/attribution.jsonl \
        --backend keyword --out runs/keyword.json
    vlc-judge run ... --backend logprob --base-url http://127.0.0.1:8000/v1 --model qwen3
    vlc-judge run ... --backend crossencoder --model-dir heads/attribution
    vlc-judge run ... --backend typesafe|cloudflare
    vlc-judge report runs/crossencoder.json --gold gold/attribution.jsonl --baseline runs/keyword.json
    vlc-judge serve --backend crossencoder --model-dir heads/attribution --port 8009
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evalharness as ev
from .gold import attribution as gold_attr
from .gold import departures as gold_dep
from .gold import screening as gold_scr
from .gold import scrub as gold_scrub
from .types import Answer

TASKS = {**{"attribution": gold_attr.TASK}, **{k: {k: v} for k, v in gold_dep.TASK.items()},
         **{k: {k: v} for k, v in gold_scr.TASK.items()}}


def make_backend(args) -> object:
    if args.backend == "keyword":
        from .backends.keyword import make_keyword_judge
        return make_keyword_judge(args.question or next(iter(ev.load_task(args.task))))
    if args.backend == "logprob":
        from .backends.logprob import LogprobJudge
        return LogprobJudge(base_url=args.base_url, model=args.model, api_key_env=args.api_key_env)
    if args.backend == "crossencoder":
        from .backends.crossencoder import CrossEncoderHead
        return CrossEncoderHead(args.model_dir, runtime=getattr(args, "runtime", "auto")).judge()
    if args.backend == "typesafe":
        from .backends.external import TypeSafeJudge
        return TypeSafeJudge(model=args.model or "jev-1.13.0")
    if args.backend == "cloudflare":
        from .backends.external import CloudflareJudge
        return CloudflareJudge()
    if args.backend == "systemone-http":
        from .backends.external import SystemOneHTTPJudge
        return SystemOneHTTPJudge(base_url=args.base_url)
    raise SystemExit(f"unknown backend {args.backend}")


def cmd_task(args) -> None:
    print(json.dumps(TASKS[args.name], ensure_ascii=False, indent=1))


def cmd_gold_attribution(args) -> None:
    src = Path(args.texts)
    if src.is_dir():
        docs = ((p.stem, p.read_text(encoding="utf-8", errors="replace")) for p in sorted(src.glob("*.txt")))
    else:
        docs = gold_attr.read_sqlite_texts(str(src), limit=args.limit, skip=args.skip)
    counts = gold_attr.build(docs, args.out, per_doc=args.per_doc, enriched_share=args.enriched_share)
    print(json.dumps(counts, ensure_ascii=False))


def cmd_gold_departures(args) -> None:
    rows = []
    if args.dep_gold:
        rows += list(gold_dep.from_dep_gold(args.dep_gold))
    if args.positions_db:
        rows += list(gold_dep.from_departures_table(args.positions_db, limit=args.limit))
    if args.rejects:
        rows += list(gold_dep.from_rejects_dump(args.rejects, limit=args.limit))
    if getattr(args, "evidence", False):
        rows = list(gold_dep.relabel_by_evidence(rows, gold_dep.adjudicated_ids(args.adjudication)))
    n = gold_dep.write(gold_dep.merge_adjudications(rows, args.adjudication), args.out)
    print(f"{n} rows -> {args.out}")


def cmd_run(args) -> None:
    task = ev.load_task(args.task)
    gold = ev.load_gold(args.gold)
    backend = make_backend(args)
    res = ev.run(backend, task, gold, cache_dir=args.cache, task_version=args.task_version, limit=args.limit)
    payload = {"backend": res.backend, "task_version": res.task_version,
               "answers": {k: a.as_dict() for k, a in res.answers.items()},
               "seconds": res.seconds, "failures": res.failures}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res.answers)} answers, {len(res.failures)} failures -> {args.out}")


def _load_run(path: str, task) -> ev.RunResult:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    res = ev.RunResult(backend=d["backend"], task_version=d["task_version"],
                       seconds=d.get("seconds", {}), failures=d.get("failures", {}))
    for rid, a in d["answers"].items():
        qn = rid_question.get(rid)
        q = task[qn] if qn else None
        res.answers[rid] = Answer.from_probs(q, a["probabilities"]) if q else Answer(
            type=a["type"], probabilities=a["probabilities"])
    return res


rid_question: dict[str, str] = {}


def cmd_report(args) -> None:
    task = ev.load_task(args.task)
    gold = ev.load_gold(args.gold)
    rid_question.update({g["id"]: g["question"] for g in gold})
    res = _load_run(args.run, task)
    base = _load_run(args.baseline, task) if args.baseline else None
    rep = ev.report(res, gold, target_precision=args.target_precision, baseline=base)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def cmd_serve(args) -> None:
    from .serve import serve
    backend = make_backend(args)
    print(f"serving {getattr(backend, 'name', 'judge')} on {args.host}:{args.port}", file=sys.stderr)
    serve(backend, host=args.host, port=args.port, model_name=getattr(backend, "name", "vlc-judge"))


def _backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", required=True,
                   choices=["keyword", "logprob", "crossencoder", "typesafe", "cloudflare", "systemone-http"])
    p.add_argument("--runtime", default="auto", choices=["auto", "onnx", "torch"],
                   help="crossencoder only: auto picks ONNX when model.onnx is there; "
                        "torch is what runs the head on a GPU")
    p.add_argument("--task")
    p.add_argument("--question")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="LLM_API_KEY")
    p.add_argument("--model-dir")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vlc-judge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("task"); p.add_argument("name", choices=sorted(TASKS)); p.set_defaults(fn=cmd_task)

    p = sub.add_parser("gold-attribution")
    p.add_argument("--texts", required=True, help="directory of *.txt or path to edrsr.db")
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--per-doc", type=int, default=6); p.add_argument("--enriched-share", type=float, default=0.0)
    p.add_argument("--skip", type=int, default=0,
                   help="skip N candidate rulings before collecting: builds a holdout "
                        "from decisions the training slice never saw")
    p.set_defaults(fn=cmd_gold_attribution)

    p = sub.add_parser("gold-departures")
    p.add_argument("--dep-gold"); p.add_argument("--positions-db"); p.add_argument("--rejects")
    p.add_argument("--adjudication"); p.add_argument("--limit", type=int); p.add_argument("--out", required=True)
    p.add_argument("--evidence", action="store_true",
                   help="label each row from the sentence itself and drop rows the sentence "
                        "does not decide (see gold/departures.py: evidence_label)")
    p.set_defaults(fn=cmd_gold_departures)

    p = sub.add_parser("run"); _backend_args(p)
    p.add_argument("--gold", required=True); p.add_argument("--out", required=True)
    p.add_argument("--cache", default=".judge-cache"); p.add_argument("--task-version", default="v1")
    p.add_argument("--limit", type=int); p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report")
    p.add_argument("run"); p.add_argument("--task", required=True); p.add_argument("--gold", required=True)
    p.add_argument("--baseline"); p.add_argument("--target-precision", type=float, default=0.95)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("scrub-gold", help="replace personal names in a gold file before upload")
    p.add_argument("--in", dest="src", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: gold_scrub.main(["--in", a.src, "--out", a.out]))

    p = sub.add_parser("probe-logprobs")
    p.add_argument("--base-url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="LLM_API_KEY")
    p.set_defaults(fn=lambda a: __import__("vlc_ua.judge.probe", fromlist=["probe"]).main(
        ["--base-url", a.base_url, "--model", a.model, "--api-key-env", a.api_key_env]))

    p = sub.add_parser("serve"); _backend_args(p)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8009)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    if getattr(args, "task", None) is None and args.cmd in ("run", "serve") and args.backend == "keyword":
        ap.error("--task is required")
    args.fn(args)


if __name__ == "__main__":
    main()
