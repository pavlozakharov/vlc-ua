"""Calibration, thresholds and the metrics the eval harness reports.

Everything here is pure Python on purpose: it runs on the server without
numpy and inside a Kaggle kernel alike, and it is small enough to read.

Vocabulary:

* ``ECE`` (expected calibration error): weighted gap between predicted
  top-probability and observed accuracy over 15 equal-width bins. Lower is
  better; 0.02 is well calibrated, 0.15 is not.
* temperature scaling: one scalar per question that rescales logits so the
  probabilities match observed frequencies on a held-out labelled split.
* confidence threshold: the smallest confidence at which accepted answers
  reach a target precision on a *random* sample from the real stream. Fitted
  on an enriched slice it will lie, which is why :func:`fit_threshold`
  insists on being told which sample it is given.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .backend import softmax


@dataclass(frozen=True)
class Labelled:
    """One evaluated item: raw per-option scores (or probabilities) plus gold."""

    scores: Mapping[str, float]   # logits or probabilities per option
    gold: str
    is_probability: bool = False  # True if ``scores`` already sums to one


def _probs(item: Labelled, temperature: float) -> dict[str, float]:
    if item.is_probability:
        if temperature == 1.0:
            return dict(item.scores)
        logits = {k: math.log(max(v, 1e-12)) for k, v in item.scores.items()}
        return softmax(logits, temperature)
    return softmax(item.scores, temperature)


def nll(items: Sequence[Labelled], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold option."""
    total = 0.0
    for it in items:
        p = _probs(it, temperature).get(it.gold, 0.0)
        total -= math.log(max(p, 1e-12))
    return total / max(len(items), 1)


def fit_temperature(items: Sequence[Labelled], lo: float = 0.05, hi: float = 20.0,
                    iters: int = 60) -> float:
    """Golden-section search of the temperature minimizing NLL on ``items``.

    Use a held-out split, never the training data, and refit per question:
    a temperature fitted on one task overshoots by 2-3x on another.
    """
    if not items:
        return 1.0
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = nll(items, math.exp(c)), nll(items, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(items, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(items, math.exp(d))
    return math.exp((a + b) / 2.0)


def accuracy(items: Sequence[Labelled], temperature: float = 1.0) -> float:
    if not items:
        return 0.0
    ok = 0
    for it in items:
        pr = _probs(it, temperature)
        if max(pr, key=pr.__getitem__) == it.gold:
            ok += 1
    return ok / len(items)


def ece(items: Sequence[Labelled], temperature: float = 1.0, bins: int = 15) -> float:
    """Expected calibration error on the top-probability prediction."""
    if not items:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for it in items:
        pr = _probs(it, temperature)
        top = max(pr, key=pr.__getitem__)
        conf = pr[top]
        idx = min(int(conf * bins), bins - 1)
        buckets[idx].append((conf, top == it.gold))
    total = len(items)
    err = 0.0
    for b in buckets:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1.0 for _, ok in b if ok) / len(b)
        err += (len(b) / total) * abs(avg_conf - avg_acc)
    return err


def brier(items: Sequence[Labelled], temperature: float = 1.0) -> float:
    """Multi-class Brier score (mean squared error of the distribution)."""
    if not items:
        return 0.0
    total = 0.0
    for it in items:
        pr = _probs(it, temperature)
        total += sum((p - (1.0 if k == it.gold else 0.0)) ** 2 for k, p in pr.items())
    return total / len(items)


def confidence_of_probs(pr: Mapping[str, float]) -> float:
    from .types import confidence_of
    return confidence_of(pr)


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    target_precision: float
    achieved_precision: float
    coverage: float          # share of items accepted at the threshold
    accepted: int
    total: int
    sample_kind: str         # "random" or "enriched"

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def fit_threshold(items: Sequence[Labelled], target_precision: float,
                  sample_kind: str, temperature: float = 1.0) -> ThresholdReport:
    """Smallest confidence at which accepted answers meet ``target_precision``.

    ``sample_kind`` must be ``"random"`` for the number to mean anything about
    the production queue; an ``"enriched"`` slice may be used to look at
    sensitivity, and the report carries the label so nobody forgets.
    """
    if sample_kind not in ("random", "enriched"):
        raise ValueError("sample_kind must be 'random' or 'enriched'")
    rows = []
    for it in items:
        pr = _probs(it, temperature)
        top = max(pr, key=pr.__getitem__)
        rows.append((confidence_of_probs(pr), top == it.gold))
    rows.sort(key=lambda r: -r[0])
    best = None
    ok = 0
    for i, (conf, hit) in enumerate(rows, start=1):
        ok += 1 if hit else 0
        prec = ok / i
        if prec >= target_precision:
            best = (conf, prec, i)
    if best is None:
        return ThresholdReport(threshold=1.0, target_precision=target_precision,
                               achieved_precision=0.0, coverage=0.0, accepted=0,
                               total=len(rows), sample_kind=sample_kind)
    conf, prec, n = best
    return ThresholdReport(threshold=conf, target_precision=target_precision,
                           achieved_precision=prec, coverage=n / max(len(rows), 1),
                           accepted=n, total=len(rows), sample_kind=sample_kind)


def apply_threshold(items: Sequence[Labelled], threshold: float, target_precision: float,
                    sample_kind: str, temperature: float = 1.0) -> ThresholdReport:
    """What a threshold fitted elsewhere actually delivers on THESE items.

    :func:`fit_threshold` searches a set and returns the point of maximum
    coverage that still meets the target ON THAT SET. Reporting that number
    as the queue size is the oldest trick there is: the threshold has seen
    the answers it is being scored against. Measured 21.09.2026, fitting on
    the dev half and applying here instead:

        head v5, first 500 holdout rows   coverage 0.680 -> 0.588 (precision 0.973)
        jev-1.13.0, same rows             coverage 0.244 -> 0.424 (precision 0.887)
        jev-1.13.0, departure pairs       coverage 0.333 -> 0.000 on this split

    Coverage can move either way; what does not survive is the PROMISE. The
    in-sample number always reads "precision 0.95" because that is what it
    was chosen to read.
    """
    if sample_kind not in ("random", "enriched"):
        raise ValueError("sample_kind must be 'random' or 'enriched'")
    ok = n = 0
    for it in items:
        pr = _probs(it, temperature)
        top = max(pr, key=pr.__getitem__)
        if confidence_of_probs(pr) >= threshold:
            n += 1
            ok += 1 if top == it.gold else 0
    return ThresholdReport(threshold=threshold, target_precision=target_precision,
                           achieved_precision=(ok / n) if n else 0.0,
                           coverage=n / max(len(items), 1), accepted=n,
                           total=len(items), sample_kind=sample_kind)


def confusion(items: Sequence[Labelled], temperature: float = 1.0) -> dict[str, dict[str, int]]:
    """gold -> predicted -> count. The direction of errors is the diagnosis."""
    out: dict[str, dict[str, int]] = {}
    for it in items:
        pr = _probs(it, temperature)
        pred = max(pr, key=pr.__getitem__)
        out.setdefault(it.gold, {}).setdefault(pred, 0)
        out[it.gold][pred] += 1
    return out


def _quantiles(xs: Sequence[float], qs: Sequence[float] = (0.05, 0.5, 0.95)) -> list[float]:
    s = sorted(xs)
    if not s:
        return [0.0 for _ in qs]
    return [s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] for q in qs]


def threshold_band(items: Sequence[Labelled], target_precision: float, temperature: float = 1.0,
                   resamples: int = 200, seed: int = 20260923,
                   lower_quantile: float = 0.05) -> dict:
    """A serving threshold taken from a bootstrap band, not from one split.

    ``fit_threshold`` returns the lowest confidence at which the accepted
    answers still reach the target ON THE SET IT IS GIVEN. Near the target
    the precision curve is flat, so which crossing it lands on is decided by
    a handful of rows: head v6 on its holdout gave coverage 0.1826 on one
    split and 0.5007 in the median of 200 (22.09.2026). Serving one of those
    numbers is serving a coin toss.

    Here every candidate threshold is scored on ``resamples`` bootstrap
    resamples of ``items`` (drawn with replacement, same size), at the
    temperature that will serve. Two policies come out:

    * ``guarded`` — the lowest threshold whose precision stays at or above
      the target in all but ``lower_quantile`` of the resamples. This is the
      one that keeps the promise: "precision 0.95" holds in 95 % of
      resamples, not in the one that happened to be drawn.
    * ``median`` — the median of the thresholds ``fit_threshold`` picks on
      each resample. Closer to what earlier reports quoted, and it keeps the
      optimism of max-coverage selection: expect precision a little under
      the target.

    ``single_fit_out_of_bag`` is the old procedure measured honestly: fit on
    a resample, apply to the rows that resample left out. Its band is the
    lottery a single split buys.

    Rows are taken as given — use the random slice. Enriched rows make any
    threshold lie, whichever policy picks it.
    """
    import random

    rows = []
    for it in items:
        pr = _probs(it, temperature)
        top = max(pr, key=pr.__getitem__)
        rows.append((confidence_of_probs(pr), top == it.gold))
    n = len(rows)
    out: dict = {"n": n, "temperature": temperature, "target_precision": target_precision,
                 "resamples": resamples, "seed": seed, "lower_quantile": lower_quantile}
    if n == 0:
        return out
    order = sorted(range(n), key=lambda i: -rows[i][0])
    # cut positions: accept order[:j+1]; only where the next confidence is
    # strictly lower, so tied rows are never split between accepted and not
    cuts = [j for j in range(n) if j == n - 1 or rows[order[j + 1]][0] < rows[order[j]][0]]
    conf_at = [rows[order[j]][0] for j in cuts]

    def curve(weights: Sequence[int]) -> tuple[list[float], list[float]]:
        """precision and coverage at every cut, for rows weighted by counts"""
        prec, cov = [], []
        w = hit = 0
        k = 0
        total = sum(weights)
        for j in range(n):
            i = order[j]
            w += weights[i]
            hit += weights[i] if rows[i][1] else 0
            if k < len(cuts) and cuts[k] == j:
                prec.append(hit / w if w else float("nan"))
                cov.append(w / total if total else 0.0)
                k += 1
        return prec, cov

    def fit_on(prec: Sequence[float]) -> float:
        best = None
        for k, p in enumerate(prec):
            if p == p and p >= target_precision:
                best = k
        return conf_at[best] if best is not None else 1.0

    def apply_on(weights: Sequence[int], tau: float) -> tuple[float, float]:
        acc = ok = 0
        total = sum(weights)
        for i, (c, h) in enumerate(rows):
            if weights[i] and c >= tau:
                acc += weights[i]
                ok += weights[i] if h else 0
        return (ok / acc if acc else float("nan")), (acc / total if total else 0.0)

    rnd = random.Random(seed)
    prec_by_b, cov_by_b, fitted, oob = [], [], [], []
    for _ in range(resamples):
        weights = [0] * n
        for _ in range(n):
            weights[rnd.randrange(n)] += 1
        prec, cov = curve(weights)
        prec_by_b.append(prec)
        cov_by_b.append(cov)
        tau = fit_on(prec)
        fitted.append(tau)
        left_out = [0 if w else 1 for w in weights]
        oob.append(apply_on(left_out, tau))

    def band_at(tau: float) -> dict:
        # on every resample, tau accepts exactly the rows up to the last cut
        # whose confidence is >= tau
        k = max((k for k, c in enumerate(conf_at) if c >= tau), default=None)
        if k is None:
            ps, cs = [float("nan")] * resamples, [0.0] * resamples
        else:
            ps = [prec_by_b[b][k] for b in range(resamples)]
            cs = [cov_by_b[b][k] for b in range(resamples)]
        full_p, full_c = apply_on([1] * n, tau)
        clean = [p for p in ps if p == p]
        return {"threshold": tau, "precision_band": _quantiles(clean),
                "coverage_band": _quantiles(cs),
                "precision_full": full_p, "coverage_full": full_c,
                "share_of_resamples_meeting_target": (sum(1 for p in clean if p >= target_precision)
                                                      / len(clean)) if clean else 0.0}

    # guarded: lowest threshold whose lower-quantile precision meets the target
    guarded_k = None
    for k in range(len(cuts)):
        col = [prec_by_b[b][k] for b in range(resamples)]
        col = [p for p in col if p == p]
        if col and _quantiles(col, (lower_quantile,))[0] >= target_precision:
            guarded_k = k
    tau_guarded = conf_at[guarded_k] if guarded_k is not None else 1.0
    tau_median = _quantiles(fitted, (0.5,))[0]

    out["fitted_threshold_band"] = _quantiles(fitted)
    out["single_fit_out_of_bag"] = {
        "precision_band": _quantiles([p for p, _ in oob if p == p]),
        "coverage_band": _quantiles([c for _, c in oob]),
    }
    out["policies"] = {"guarded": band_at(tau_guarded), "median": band_at(tau_median)}
    return out


def split(items: Sequence[Labelled], dev_share: float = 0.5, seed: int = 20260920) -> tuple[list[Labelled], list[Labelled]]:
    """Deterministic dev/test split; fit temperature on dev, report on test."""
    import random

    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    cut = int(len(idx) * dev_share)
    dev = [items[i] for i in idx[:cut]]
    test = [items[i] for i in idx[cut:]]
    return dev, test
