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
    missing = gold_dep.unmatched_adjudications(rows, args.adjudication)
    n = gold_dep.write(gold_dep.merge_adjudications(rows, args.adjudication), args.out)
    print(f"{n} rows -> {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} adjudicated rows are not in the sources and "
              f"their verdicts are lost: {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}", file=sys.stderr)


def cmd_run(args) -> None:
    task = ev.load_task(args.task)
    gold = ev.load_gold(args.gold)
    backend = make_backend(args)
    res = ev.run(backend, task, gold, cache_dir=args.cache, task_version=args.task_version,
                 limit=args.limit, accept_legacy_cache=args.accept_legacy_cache,
                 concurrency=args.concurrency)
    payload = {"backend": res.backend, "task_version": res.task_version,
               "fingerprint": res.fingerprint,
               "answers": {k: a.as_dict() for k, a in res.answers.items()},
               "seconds": res.seconds, "failures": res.failures}
    if res.scores or res.temperatures:
        # raw logits and the temperature that made ``answers`` out of them:
        # without these a run cannot be recalibrated (see cmd_calibrate)
        payload["temperatures"] = res.temperatures
        payload["scores"] = res.scores
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res.answers)} answers, {len(res.failures)} failures -> {args.out}")


def _load_run(path: str, task) -> ev.RunResult:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    res = ev.RunResult(backend=d["backend"], task_version=d["task_version"],
                       fingerprint=str(d.get("fingerprint") or ""),
                       seconds=d.get("seconds", {}), failures=d.get("failures", {}),
                       scores=d.get("scores", {}), temperatures=d.get("temperatures", {}))
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
    rep = ev.report(res, gold, target_precision=args.target_precision, baseline=base,
                    run_temperatures=_run_temperatures(args, task))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def _run_temperatures(args, task) -> dict[str, float] | None:
    """``--run-temperature`` values: ``1.2533`` for every question in the task,
    or ``attribution=1.2533`` for one."""
    vals = getattr(args, "run_temperature", None) or []
    if not vals:
        return None
    out: dict[str, float] = {}
    for v in vals:
        if "=" in v:
            qn, t = v.split("=", 1)
            out[qn.strip()] = float(t)
        else:
            for qn in task:
                out[qn] = float(v)
    return out


def _random_items(args, task):
    """-> (run, {question: (items, basis)}) over the random slice. Refuses a
    question whose rows cannot all be turned into logits."""
    gold = ev.load_gold(args.gold)
    rid_question.update({g["id"]: g["question"] for g in gold})
    res = _load_run(args.run, task)
    declared = _run_temperatures(args, task)
    rows_by_q: dict[str, list] = {}
    for row in gold:
        rows_by_q.setdefault(row["question"], []).append(row)
    out = {}
    for qn, rows in rows_by_q.items():
        items, basis = ev.labelled_with_basis(res, rows, "random", declared)
        ids = [r["id"] for r in rows if r.get("sample", "random") == "random" and r["id"] in res.answers]
        if basis == "logits":
            how = ("logits recorded in the run" if all(i in res.scores for i in ids) else
                   f"probabilities of the run un-tempered at the declared run temperature "
                   f"{declared[qn]}")
        else:
            how = None
        out[qn] = (items, how)
    return res, declared, out


_NO_LOGITS = ("{qn}: the run holds probabilities with a temperature already applied, and does "
              "not record which. A temperature fitted on them is a factor on top of that one, "
              "not a temperature — until 23.09.2026 this command wrote exactly that factor into "
              "head.json. Re-run with the current code (it records raw logits), or pass "
              "--run-temperature with the value head.json held when the run was made.")


def cmd_calibrate(args) -> None:
    """Fit the temperature on a HOLDOUT, on raw logits, and write it into head.json.

    WHAT THIS COMMAND GOT WRONG UNTIL 23.09.2026, AND WHAT STANDS. It read the
    run's probabilities as if no temperature had been applied. But a run is
    made through ``CrossEncoderHead.judge()``, which applies head.json's
    temperature, and the Kaggle kernel runs the holdout right after the
    trainer has written one. So the fitted number was a FACTOR on top of the
    run's temperature, and that factor went into head.json as the
    temperature. Recomputed on the same splits from logits recovered as
    T_run * log p (evalharness.logits_for); the run temperatures were checked
    by re-scoring rows with the int8 head, implied T equal to 4 digits:

                         T_run    optimum   ECE@opt  ECE@trainer  ECE@T=1
        v5 torch / T4    1.2533   1.3083    0.0385   0.0537       0.0922
        v5 int8 / CPU    1.2533   1.3312    0.0466   0.0571       0.1011
        v6 torch / T4    1.9075   1.8711    0.0326   0.0317       0.1034
        v6 int8 / CPU    0.9810   1.8974    0.0311   0.0284       0.1078

    The claims of 22.09 that this refutes, and why they looked true:

    * "head v6's stored temperature 1.9075 gives ECE 0.2552, it does not
      transfer from the trainer's dev split" — 0.2552 is the ECE at
      1.9075 applied twice (3.64). At 1.9075 the ECE is 0.0317 on torch and
      0.0284 on int8: the trainer's temperature transfers for v6.
    * "quantisation does not preserve the temperature: torch 0.9810, int8
      1.9341; at the torch T the int8 ECE is 0.1098" — the two numbers were
      factors on top of DIFFERENT run temperatures (1.9075 and 0.9810). The
      optima are 1.8711 and 1.8974, 1.4 % apart; 0.1098 is the int8 ECE at
      0.962. Raw int8 is not "twice as overconfident" either: at T = 1 the
      two runtimes give 0.1034 and 0.1078.
    * "head v5's stored 1.2533 gives 0.0650" — that is 1.2533 applied twice.
      At 1.2533 the ECE is 0.0537 / 0.0571: v5 does miss the 0.05 gate with
      its trainer's temperature, but narrowly, and 1.3312 fixes it.

    What stands: calibrating on a holdout, on the artefact that serves, is
    still the procedure (it caught v5's narrow miss), and the admission
    numbers were right, because a fitted factor on top of T_run gives the
    same calibrated probabilities as the absolute optimum — only the number
    written down was not the one serve applies. head v6 served 1.9341 where
    the optimum is 1.8974 (int8 ECE 0.0299, harmless by luck: the two
    errors nearly cancelled). Applying this command's old output for v5 would
    have written 1.0622 and served ECE 0.0911.

    So: the temperature is fitted on raw logits only. A run from the current
    harness records them; an older run needs ``--run-temperature`` (the
    value head.json held when the run was made), and without it the command
    refuses. The block records which basis it used.
    """
    from .calibration import ece, fit_temperature, split

    task = ev.load_task(args.task)
    res, declared, by_q = _random_items(args, task)
    fingerprint = res.fingerprint
    if "_TorchImpl" in fingerprint and (Path(args.model_dir) / "model.onnx").exists():
        print("NOTE: this run came from the torch runtime, but the model directory holds a "
              "model.onnx, so serving will use ONNX. On heads v5 and v6 the two optimal "
              "temperatures differed by 1.4-1.7 % (v6: 1.8711 torch, 1.8974 int8), so the "
              "number is usable, but it is not the served artefact's.", file=sys.stderr)

    path = Path(args.model_dir) / "head.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    temps = dict(meta.get("temperatures", {}))
    report = {}
    refused = False
    for qn, (items, how) in by_q.items():
        if how is None:
            print(_NO_LOGITS.format(qn=qn), file=sys.stderr)
            refused = True
            continue
        if len(items) < 20:
            print(f"{qn}: {len(items)} rows is too few to calibrate on, left alone", file=sys.stderr)
            continue
        dev, test = split(items)
        t = fit_temperature(dev)
        trainer_t = meta.get("dev_metrics", {}).get(qn, {}).get("temperature")
        report[qn] = {"n_dev": len(dev), "n_test": len(test), "temperature": t,
                      "ece_at_this": ece(test, t), "ece_at_stored": ece(test, temps.get(qn, 1.0)),
                      "ece_uncalibrated": ece(test, 1.0),
                      "ece_at_trainer": ece(test, trainer_t) if trainer_t else None,
                      "temperature_before": temps.get(qn), "basis": how,
                      "run_temperature": (declared or {}).get(qn, res.temperatures.get(qn)),
                      "gold": str(args.gold), "run": str(args.run), "fingerprint": fingerprint}
        # a threshold only holds at the temperature it was fitted at
        tc = meta.get("threshold_calibration", {}).get(qn)
        if tc and abs(float(tc.get("temperature", 0.0)) - t) > 1e-9:
            meta.get("thresholds", {}).pop(qn, None)
            tc["stale"] = f"fitted at T {tc.get('temperature')}; the temperature is now {t}"
            print(f"{qn}: the threshold in head.json was fitted at T {tc.get('temperature')} and "
                  f"is removed; run `vlc-judge threshold` again", file=sys.stderr)
        temps[qn] = t
    meta["temperatures"] = temps
    meta.setdefault("calibration", {}).update(report)
    if args.write and report:
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"head.json updated: {path}")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if refused:
        raise SystemExit(2)


def cmd_threshold(args) -> None:
    """Pick the serving confidence threshold from a bootstrap band and write
    it into head.json next to the temperature it was fitted at.

    One split is a lottery here (head v6: coverage 0.1826 on one split, 0.5007
    in the median of 200), see calibration.threshold_band. The threshold is
    fitted at the temperature head.json holds NOW — the one serve applies —
    so calibrate first; if the temperature later changes, ``calibrate``
    removes the threshold rather than leave one fitted at another T.
    """
    from .calibration import threshold_band

    task = ev.load_task(args.task)
    res, declared, by_q = _random_items(args, task)
    path = Path(args.model_dir) / "head.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    temps = meta.get("temperatures", {})
    cal = meta.get("calibration", {})
    out = {}
    refused = False
    for qn, (items, how) in by_q.items():
        if how is None:
            print(_NO_LOGITS.format(qn=qn), file=sys.stderr)
            refused = True
            continue
        if qn not in temps:
            print(f"{qn}: head.json holds no temperature for it; a threshold is a confidence at "
                  f"the temperature that serves, so run `vlc-judge calibrate` first",
                  file=sys.stderr)
            refused = True
            continue
        if len(items) < 50:
            print(f"{qn}: {len(items)} rows is too few for a threshold, left alone", file=sys.stderr)
            continue
        fp_cal = str(cal.get(qn, {}).get("fingerprint") or "")
        if fp_cal and res.fingerprint and fp_cal != res.fingerprint:
            print(f"{qn}: WARNING the temperature was calibrated on a run of {fp_cal!r}, this "
                  f"run is {res.fingerprint!r}", file=sys.stderr)
        band = threshold_band(items, args.target_precision, float(temps[qn]),
                              resamples=args.resamples, seed=args.seed)
        band.update({"policy": args.policy, "basis": how, "gold": str(args.gold),
                     "run": str(args.run), "fingerprint": res.fingerprint,
                     "sample": "random"})
        out[qn] = band
    if args.write and out:
        meta.setdefault("thresholds", {}).update(
            {qn: b["policies"][args.policy]["threshold"] for qn, b in out.items()})
        meta.setdefault("threshold_calibration", {}).update(out)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"head.json updated: {path}")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if refused:
        raise SystemExit(2)


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
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel requests, for REMOTE backends only: the loop waits on a "
                        "socket, not on this machine. Leave at 1 for a local head")
    p.add_argument("--accept-legacy-cache", action="store_true",
                   help="read cache entries written before backends carried a fingerprint. "
                        "Only for resuming a run whose backend has not changed: such entries "
                        "cannot be told apart from stale ones")
    p.set_defaults(fn=cmd_run)

    run_t_help = ("for a run made before 23.09.2026, which stored tempered probabilities and "
                  "not the logits: the temperature head.json held when the run was made "
                  "(a number, or QUESTION=number)")
    p = sub.add_parser("report")
    p.add_argument("run"); p.add_argument("--task", required=True); p.add_argument("--gold", required=True)
    p.add_argument("--baseline"); p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--run-temperature", action="append", help=run_t_help)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("scrub-gold", help="replace personal names in a gold file before upload")
    p.add_argument("--in", dest="src", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: gold_scrub.main(["--in", a.src, "--out", a.out]))

    p = sub.add_parser("probe-logprobs")
    p.add_argument("--base-url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="LLM_API_KEY")
    p.set_defaults(fn=lambda a: __import__("vlc_ua.judge.probe", fromlist=["probe"]).main(
        ["--base-url", a.base_url, "--model", a.model, "--api-key-env", a.api_key_env]))

    p = sub.add_parser("calibrate",
                       help="fit the temperature on a holdout run and write it into head.json")
    p.add_argument("run"); p.add_argument("--task", required=True); p.add_argument("--gold", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--write", action="store_true", help="actually write head.json")
    p.add_argument("--run-temperature", action="append", help=run_t_help)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("threshold",
                       help="pick the serving confidence threshold from a bootstrap band "
                            "and write it into head.json")
    p.add_argument("run"); p.add_argument("--task", required=True); p.add_argument("--gold", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--policy", choices=["guarded", "median"], default="guarded",
                   help="guarded: precision holds in 95%% of resamples; median: the median "
                        "single-split threshold, a little under target on average")
    p.add_argument("--resamples", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--write", action="store_true", help="actually write head.json")
    p.add_argument("--run-temperature", action="append", help=run_t_help)
    p.set_defaults(fn=cmd_threshold)

    p = sub.add_parser("serve"); _backend_args(p)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8009)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    if getattr(args, "task", None) is None and args.cmd in ("run", "serve") and args.backend == "keyword":
        ap.error("--task is required")
    args.fn(args)


if __name__ == "__main__":
    main()
