"""Typed decision primitives.

The shape mirrors the "System One" contract (state + named questions in,
probability distributions out) so any backend, a local head, a logprob
reader over an LLM, or an external service, is interchangeable and any
consumer sees the same thing: a distribution and a confidence, never text.

Design rules that the rest of the package relies on:

* A question never produces free text. Every answer is a distribution over a
  closed set of options that the caller defined.
* ``confidence`` is derived from the shape of the distribution, not from the
  model's self-report. Noul answers carry only ``p`` (probability that the
  statement holds); Choice and Score also carry ``confidence``.
* A backend must not decide anything: thresholds, escalation and gates live
  in caller code (see :mod:`vlc_ua.judge.calibration`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

State = Union[str, Mapping[str, Any], Sequence[Any]]


@dataclass(frozen=True)
class Noul:
    """A yes/no judgement: does ``instructions`` hold for the state?"""

    instructions: str
    true: str | None = None   # optional clarification of what "yes" means
    false: str | None = None  # optional clarification of what "no" means

    type: str = field(default="noul", init=False)

    @property
    def options(self) -> tuple[str, ...]:
        return ("yes", "no")


@dataclass(frozen=True)
class Choice:
    """Pick exactly one of ``criteria``; values are optional descriptions."""

    instructions: str
    criteria: Mapping[str, str | None]

    type: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("Choice needs at least two options")

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.criteria)


@dataclass(frozen=True)
class Score:
    """Rate on an ordered scale; ``levels`` go from lowest to highest."""

    instructions: str
    levels: Sequence[str]

    type: str = field(default="score", init=False)

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError("Score needs at least two levels")

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.levels)


Question = Union[Noul, Choice, Score]


def normalize(probs: Mapping[str, float]) -> dict[str, float]:
    """Rescale a non-negative mapping to sum to one (uniform if all zero)."""
    total = float(sum(max(v, 0.0) for v in probs.values()))
    if total <= 0.0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: max(v, 0.0) / total for k, v in probs.items()}


def confidence_of(probs: Mapping[str, float]) -> float:
    """Collapse a distribution into one number in [0, 1].

    Defined as one minus normalized entropy: 1.0 when all mass sits on one
    option, 0.0 when the distribution is uniform. Two options at 0.58/0.42
    give about 0.02, which is the point: a slim top-1 margin is *not*
    confidence. Callers gate on this, never on the top probability alone.
    """
    ps = [p for p in probs.values() if p > 0.0]
    n = len(probs)
    if n < 2:
        return 1.0
    ent = -sum(p * math.log(p) for p in ps)
    return max(0.0, min(1.0, 1.0 - ent / math.log(n)))


@dataclass(frozen=True)
class Answer:
    """Distribution over a question's options plus derived fields."""

    type: str
    probabilities: dict[str, float]
    choice: str | None = None      # choice/score: argmax option
    p: float | None = None         # noul: probability of "yes"
    confidence: float | None = None
    score: float | None = None     # score: expected level index (0-based)
    legend: dict[str, str] | None = None

    @classmethod
    def from_probs(cls, question: Question, probs: Mapping[str, float]) -> "Answer":
        pr = normalize({k: float(probs.get(k, 0.0)) for k in question.options})
        if isinstance(question, Noul):
            return cls(type="noul", probabilities=pr, p=pr["yes"],
                       confidence=confidence_of(pr))
        top = max(pr, key=pr.__getitem__)
        if isinstance(question, Score):
            levels = list(question.levels)
            expected = sum(pr[l] * i for i, l in enumerate(levels))
            return cls(type="score", probabilities=pr, choice=top,
                       confidence=confidence_of(pr), score=expected,
                       legend={str(i): l for i, l in enumerate(levels)})
        return cls(type="choice", probabilities=pr, choice=top,
                   confidence=confidence_of(pr))

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "probabilities": self.probabilities}
        if self.type == "noul":
            d["noul"] = self.p
        else:
            d["choice"] = self.choice
            d["confidence"] = self.confidence
        if self.type == "score":
            d["score"] = self.score
            d["legend"] = self.legend
        return d


def question_from_dict(d: Mapping[str, Any]) -> Question:
    """Parse the wire form ``{"type": ..., "instructions": ..., ...}``."""
    t = d.get("type")
    if t == "noul":
        crit = d.get("criteria") or {}
        return Noul(instructions=d.get("instructions", ""),
                    true=crit.get("true"), false=crit.get("false"))
    if t == "choice":
        return Choice(instructions=d.get("instructions", ""),
                      criteria=dict(d["criteria"]))
    if t == "score":
        crit = d.get("criteria")
        levels = [c if isinstance(c, str) else c.get("name") or c.get("level")
                  for c in crit]
        return Score(instructions=d.get("instructions", ""), levels=levels)
    raise ValueError(f"unknown question type: {t!r}")


def question_to_dict(q: Question) -> dict[str, Any]:
    if isinstance(q, Noul):
        d: dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        if q.true or q.false:
            d["criteria"] = {"true": q.true, "false": q.false}
        return d
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions,
                "criteria": dict(q.criteria)}
    return {"type": "score", "instructions": q.instructions,
            "criteria": list(q.levels)}
