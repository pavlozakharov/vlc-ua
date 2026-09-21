"""Evaluation harness: one gold file, one backend, one honest report.

Gold format (JSONL, one object per line)::

    {"id": "...", "state": <str|dict>, "question": "<name>", "gold": "<option>",
     "sample": "random" | "enriched", "source": "<where the label came from>"}

Questions are defined once per task in a *task file* (JSON) mapping the
question name to its wire form, so the same gold can be run against any
backend and the instruction text is a versioned artifact::

    {"attribution": {"type": "choice", "instructions": "...", "criteria": {...}}}

The report separates what the project's rules say must not be mixed:

* sensitivity and confusion on the enriched slice;
* precision, ECE and queue size (coverage at threshold) on the random slice;
* per-item wins/losses against a baseline backend when one is given.

Nothing here writes to a database. It reads gold, calls a backend, and
prints JSON. Caching of raw scores per (backend, item, task) is on by
default so a rerun with a new temperature or threshold costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .backend import Judge
from .calibration import (Labelled, accuracy, apply_threshold, brier, confusion, ece,
                          fit_temperature, fit_threshold, split)
from .types import Answer, Question, question_from_dict


def load_task(path: str | Path) -> dict[str, Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: question_from_dict(q) for name, q in data.items()}


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _item_key(backend_name: str, task_version: str, row: Mapping[str, Any],
              fingerprint: str = "") -> str:
    """Cache key for one answer.

    ``fingerprint`` is what the backend itself says it is: the rules it
    compiled, the model directory it loaded, the remote model it will call.
    Without it the key was (name | task_version | state | question), and a
    backend whose BEHAVIOUR changed under an unchanged name served its old
    answers as new ones. That is not hypothetical: after the tie-break fix of
    21.09.2026 a run of ``keyword:departure_pair`` over departures-v7 hit the
    default cache 2 070 times out of 2 070, and 352 of those answers (17.0%)
    had a different top choice than the current code produces.
    """
    h = hashlib.sha256()
    h.update(backend_name.encode()); h.update(b"|"); h.update(task_version.encode())
    h.update(b"|"); h.update(fingerprint.encode())
    h.update(b"|"); h.update(json.dumps(row.get("state"), ensure_ascii=False, sort_keys=True).encode())
    h.update(b"|"); h.update(str(row.get("question")).encode())
    return h.hexdigest()


@dataclass
class RunResult:
    backend: str
    task_version: str
    fingerprint: str = ""
    answers: dict[str, Answer] = field(default_factory=dict)   # item id -> answer
    seconds: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def run(backend: Judge, task: Mapping[str, Question], gold: Iterable[Mapping[str, Any]],
        cache_dir: str | Path | None = ".judge-cache", task_version: str = "v1",
        limit: int | None = None, accept_legacy_cache: bool = False,
        concurrency: int = 1) -> RunResult:
    """Ask the backend every gold item's question; cache raw answers on disk.

    ``accept_legacy_cache`` reads entries written before backends carried a
    fingerprint. It is an explicit escape hatch for resuming a long run whose
    backend demonstrably has not changed, never a default: the whole point of
    the fingerprint is that such entries cannot be told apart otherwise.

    ``concurrency`` is for REMOTE backends, where the loop spends its life
    waiting on a socket. Measured against TypeSafe jev on 21.09.2026, per
    request latency flat at every level and no refusals:

        1 thread   1.5 rows/s      8 threads  11.5 rows/s
        4 threads  5.6 rows/s     32 threads  24.9 rows/s

    which turns the 3 001-row holdout from half an hour into two minutes.
    Leave it at 1 for a local head: the ONNX session already spreads one
    forward pass across the cores, and competing sessions only thrash. The
    per-request timings stay honest either way — each row is timed around its
    own call, not around the batch.
    """
    res = RunResult(backend=getattr(backend, "name", backend.__class__.__name__),
                    task_version=task_version)
    fingerprint = str(getattr(backend, "fingerprint", "") or "")
    res.fingerprint = fingerprint
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    rows = list(gold)[:limit] if limit is not None else list(gold)

    def cache_path(row):
        if not cache:
            return None
        cpath = cache / f"{_item_key(res.backend, task_version, row, fingerprint)}.json"
        if accept_legacy_cache and not cpath.exists():
            legacy = cache / f"{_item_key(res.backend, task_version, row)}.json"
            if legacy.exists():
                return legacy
        return cpath

    def ask_one(row):
        """-> (id, answer|None, seconds, failure|None). Pure per row, so the
        only shared state is the dicts the caller fills in one thread."""
        qn = row["question"]
        q = task[qn]
        cpath = cache_path(row)
        if cpath and cpath.exists():
            d = json.loads(cpath.read_text(encoding="utf-8"))
            return row["id"], Answer.from_probs(q, d["probabilities"]), d.get("seconds", 0.0), None
        t0 = time.time()
        try:
            ans = backend.ask(row["state"], {qn: q})[qn]
        except Exception as exc:  # provider quota, network, malformed: a separate class
            return row["id"], None, time.time() - t0, f"{type(exc).__name__}: {exc}"[:300]
        dt = time.time() - t0
        if cpath:
            cpath.write_text(json.dumps({"probabilities": ans.probabilities, "seconds": dt},
                                        ensure_ascii=False), encoding="utf-8")
        return row["id"], ans, dt, None

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            out = list(pool.map(ask_one, rows))
    else:
        out = [ask_one(row) for row in rows]

    for rid, ans, dt, failure in out:
        if failure is not None:
            res.failures[rid] = failure
            continue
        res.answers[rid] = ans
        res.seconds[rid] = dt
    return res


def labelled(result: RunResult, gold: Iterable[Mapping[str, Any]], sample: str | None = None) -> list[Labelled]:
    out = []
    for row in gold:
        if sample and row.get("sample", "random") != sample:
            continue
        a = result.answers.get(row["id"])
        if a is None:
            continue
        out.append(Labelled(scores=a.probabilities, gold=row["gold"], is_probability=True))
    return out


def report(result: RunResult, gold: list[Mapping[str, Any]], target_precision: float = 0.95,
           baseline: RunResult | None = None) -> dict[str, Any]:
    """Build the report dict. Temperature is fitted on a dev half of the
    random slice and applied to the test half; the enriched slice is reported
    at the same temperature but only for sensitivity and confusion."""
    rand = labelled(result, gold, "random")
    enr = labelled(result, gold, "enriched")
    dev, test = split(rand) if len(rand) >= 20 else ([], rand)
    temp = fit_temperature(dev) if dev else 1.0
    out: dict[str, Any] = {
        "backend": result.backend, "task_version": result.task_version,
        "n_random": len(rand), "n_enriched": len(enr), "n_failures": len(result.failures),
        "failures_sample": dict(list(result.failures.items())[:5]),
        "latency_s": _latency(result.seconds.values()),
        "temperature": temp,
    }
    if test:
        out["random"] = {
            "accuracy": accuracy(test, temp), "ece": ece(test, temp), "brier": brier(test, temp),
            "accuracy_uncalibrated": accuracy(test), "ece_uncalibrated": ece(test),
            "threshold": fit_threshold(test, target_precision, "random", temp).as_dict(),
            "confusion": confusion(test, temp),
        }
        # The honest one: threshold chosen on the dev half, queue measured on
        # the test half. ``threshold`` above is fitted on the very rows it
        # then scores, so its "achieved_precision" is always the target — it
        # is kept for continuity with earlier reports, not because it means
        # anything about a production queue. Admission should read this.
        if dev:
            tau = fit_threshold(dev, target_precision, "random", fit_temperature(dev)).threshold
            out["random"]["threshold_heldout"] = apply_threshold(
                test, tau, target_precision, "random", temp).as_dict()
    if enr:
        out["enriched"] = {
            "sensitivity_accuracy": accuracy(enr, temp), "ece": ece(enr, temp),
            "confusion": confusion(enr, temp),
            "note": "enriched slice: sensitivity only; thresholds fitted here would lie",
        }
    if baseline is not None:
        out["vs_baseline"] = wins_losses(result, baseline, gold)
    return out


def wins_losses(a: RunResult, b: RunResult, gold: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-item comparison: where A is right and B wrong, and the reverse."""
    wins, losses, both, neither = [], [], 0, 0
    for row in gold:
        ans_a, ans_b = a.answers.get(row["id"]), b.answers.get(row["id"])
        if ans_a is None or ans_b is None:
            continue
        pa = max(ans_a.probabilities, key=ans_a.probabilities.__getitem__)
        pb = max(ans_b.probabilities, key=ans_b.probabilities.__getitem__)
        ok_a, ok_b = pa == row["gold"], pb == row["gold"]
        if ok_a and not ok_b:
            wins.append(row["id"])
        elif ok_b and not ok_a:
            losses.append(row["id"])
        elif ok_a:
            both += 1
        else:
            neither += 1
    return {"baseline": b.backend, "wins": len(wins), "losses": len(losses),
            "both_right": both, "both_wrong": neither,
            "win_ids": wins[:50], "loss_ids": losses[:50]}


def _latency(vals: Iterable[float]) -> dict[str, float]:
    xs = sorted(vals)
    if not xs:
        return {}
    def pct(p: float) -> float:
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]
    return {"p50": pct(0.5), "p90": pct(0.9), "max": xs[-1], "n": float(len(xs))}
