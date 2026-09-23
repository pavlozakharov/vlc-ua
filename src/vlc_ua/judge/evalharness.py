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
    # Scoring backends only. ``scores`` are the raw per-option logits, and
    # ``temperatures`` what the backend applied to turn them into
    # ``answers``. A run that keeps only the probabilities has the
    # temperature baked in with nothing to say which one — and every run
    # before 23.09.2026 was that kind (see cli.cmd_calibrate).
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    temperatures: dict[str, float] = field(default_factory=dict)


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

    A SCORING backend (one with ``raw_scores``: the trained head, keyword
    rules) is cached as raw scores, and its temperature is applied on the
    way out. Until 23.09.2026 the cache and the run file kept the tempered
    probabilities, and three things went wrong with that at once:

    * the key carries the fingerprint (head directory, runtime, window) but
      not the temperature, so after ``calibrate --write`` a rerun was served
      the probabilities of the OLD temperature under the new head.json;
    * the run file did not say which temperature was baked in, and
      ``calibrate`` read the probabilities as if none were. What it fitted was
      a factor on top of the run's temperature, and it wrote that factor into
      head.json as if it were the temperature itself;
    * reports printed "ECE at the stored temperature" by applying it a second
      time, and "uncalibrated" was the ECE at the run's temperature.

    Entries of a scoring backend that hold no scores are therefore misses:
    nothing in them says which temperature made them. ``accept_legacy_cache``
    still serves them, as it serves unfingerprinted ones, and such rows come
    out without scores, so ``calibrate`` and ``threshold`` will not take them
    without being told the temperature.
    """
    res = RunResult(backend=getattr(backend, "name", backend.__class__.__name__),
                    task_version=task_version)
    fingerprint = str(getattr(backend, "fingerprint", "") or "")
    res.fingerprint = fingerprint
    scoring = callable(getattr(backend, "raw_scores", None)) and callable(getattr(backend, "answer", None))
    if scoring:
        res.temperatures = dict(getattr(backend, "temperatures", {}) or {})
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
        """-> (id, answer|None, seconds, failure|None, raw scores|None). Pure
        per row, so the only shared state is the dicts the caller fills in one
        thread."""
        qn = row["question"]
        q = task[qn]
        cpath = cache_path(row)
        if cpath and cpath.exists():
            d = json.loads(cpath.read_text(encoding="utf-8"))
            if not scoring:
                return row["id"], Answer.from_probs(q, d["probabilities"]), d.get("seconds", 0.0), None, None
            if "scores" in d:
                return row["id"], backend.answer(qn, q, d["scores"]), d.get("seconds", 0.0), None, d["scores"]
            if accept_legacy_cache:
                return row["id"], Answer.from_probs(q, d["probabilities"]), d.get("seconds", 0.0), None, None
            # a scoring backend's entry without scores: re-score (see above)
        t0 = time.time()
        raw = None
        try:
            if scoring:
                raw = backend.raw_scores(row["state"], {qn: q})[qn]
                ans = backend.answer(qn, q, raw)
            else:
                ans = backend.ask(row["state"], {qn: q})[qn]
        except Exception as exc:  # provider quota, network, malformed: a separate class
            return row["id"], None, time.time() - t0, f"{type(exc).__name__}: {exc}"[:300], None
        dt = time.time() - t0
        if cpath:
            entry = {"probabilities": ans.probabilities, "seconds": dt}
            if raw is not None:
                entry["scores"] = raw
            cpath.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        return row["id"], ans, dt, None, raw

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            out = list(pool.map(ask_one, rows))
    else:
        out = [ask_one(row) for row in rows]

    for rid, ans, dt, failure, raw in out:
        if failure is not None:
            res.failures[rid] = failure
            continue
        res.answers[rid] = ans
        res.seconds[rid] = dt
        if raw is not None:
            res.scores[rid] = dict(raw)
    return res


def logits_for(result: RunResult, rid: str,
               run_temperatures: Mapping[str, float] | None = None,
               question: str | None = None) -> dict[str, float] | None:
    """Raw per-option logits for one row, or None when they cannot be known.

    Recorded scores are taken as they are. Otherwise the stored probabilities
    are un-tempered: softmax(z / T) has log p_k = z_k / T - log Z, so
    T * log p_k recovers z_k up to one constant per row, which softmax does
    not see. T comes only from ``run_temperatures`` — what the caller says was
    in force when the run was made. Not from the run's own ``temperatures``:
    in a run that records them, a row without scores is one served from a
    legacy cache entry, made at some earlier temperature nobody wrote down.
    Without a declared T the answer is None, because guessing T = 1 is
    exactly the mistake this function exists to stop.
    """
    if rid in result.scores:
        return dict(result.scores[rid])
    a = result.answers.get(rid)
    if a is None or not run_temperatures or question not in run_temperatures:
        return None
    import math
    t = run_temperatures[question]
    return {k: t * math.log(max(v, 1e-300)) for k, v in a.probabilities.items()}


def labelled_with_basis(result: RunResult, gold: Iterable[Mapping[str, Any]],
                        sample: str | None = None,
                        run_temperatures: Mapping[str, float] | None = None
                        ) -> tuple[list[Labelled], str]:
    """Items plus what their scores ARE.

    ``"logits"``: raw logits for every row, so a temperature fitted on them is
    the temperature, and ECE at T = 1 is the uncalibrated one.
    ``"probabilities"``: at least one row has no logits, so all rows go in as
    the stored probabilities; a temperature fitted on them is a FACTOR on top
    of whatever the run applied, and "T = 1" means "as the run left them".
    One basis per list: mixing the two would fit one number to two scales.
    """
    rows = []
    for row in gold:
        if sample and row.get("sample", "random") != sample:
            continue
        a = result.answers.get(row["id"])
        if a is None:
            continue
        rows.append((row, a, logits_for(result, row["id"], run_temperatures, row.get("question"))))
    if rows and all(z is not None for _, _, z in rows):
        return [Labelled(scores=z, gold=row["gold"]) for row, _, z in rows], "logits"
    return [Labelled(scores=a.probabilities, gold=row["gold"], is_probability=True)
            for row, a, _ in rows], "probabilities"


def labelled(result: RunResult, gold: Iterable[Mapping[str, Any]], sample: str | None = None,
             run_temperatures: Mapping[str, float] | None = None) -> list[Labelled]:
    return labelled_with_basis(result, gold, sample, run_temperatures)[0]


def report(result: RunResult, gold: list[Mapping[str, Any]], target_precision: float = 0.95,
           baseline: RunResult | None = None,
           run_temperatures: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Build the report dict. Temperature is fitted on a dev half of the
    random slice and applied to the test half; the enriched slice is reported
    at the same temperature but only for sensitivity and confusion.

    ``temperature_basis`` says what the fitted temperature is: on
    ``"logits"`` it is the temperature, on ``"probabilities"`` it is a factor
    on top of the one the run applied (``run_temperatures``), and
    ``ece_uncalibrated`` is then the ECE as the run left it, not at T = 1.
    Calibrated ECE, accuracy and thresholds are the same on either basis."""
    rand, basis = labelled_with_basis(result, gold, "random", run_temperatures)
    enr, enr_basis = labelled_with_basis(result, gold, "enriched", run_temperatures)
    dev, test = split(rand) if len(rand) >= 20 else ([], rand)
    temp = fit_temperature(dev) if dev else 1.0
    out: dict[str, Any] = {
        "backend": result.backend, "task_version": result.task_version,
        "n_random": len(rand), "n_enriched": len(enr), "n_failures": len(result.failures),
        "failures_sample": dict(list(result.failures.items())[:5]),
        "latency_s": _latency(result.seconds.values()),
        "temperature": temp, "temperature_basis": basis,
        "run_temperatures": dict(run_temperatures or result.temperatures),
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
        # a temperature fitted on one basis means nothing on the other
        t_enr = temp if enr_basis == basis else 1.0
        out["enriched"] = {
            "sensitivity_accuracy": accuracy(enr, t_enr), "ece": ece(enr, t_enr),
            "confusion": confusion(enr, t_enr), "temperature_basis": enr_basis,
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
