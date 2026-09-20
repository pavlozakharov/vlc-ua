"""Backend protocol and the two generic adapters every concrete head uses.

``Judge.ask`` takes one state and several named questions and returns one
answer per question. Batching many questions over one state is the normal
call shape: state is read once, heads are cheap.

:class:`ScoringJudge` turns any "score this option for this question on this
state" function into a Judge. That is the whole trick behind trained heads
(cross-encoder logits), logprob readers (token log-probabilities) and
external services alike: everything reduces to per-option scores that are
softmaxed and, optionally, temperature-scaled per question.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Protocol

from .types import Answer, Question, State

Scores = Mapping[str, float]
ScoreFn = Callable[[State, Question, str], float]


class Judge(Protocol):
    name: str

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        ...


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Numerically stable softmax over a mapping; temperature scales logits."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    m = max(scores.values())
    exps = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


class ScoringJudge:
    """Judge built from a per-option scoring function.

    ``temperatures`` maps question *names* to a temperature fitted with
    :func:`vlc_ua.judge.calibration.fit_temperature`; unlisted questions use
    1.0. Temperatures are per question on purpose: calibration does not
    transfer across tasks, so a head calibrated for attribution says nothing
    about its calibration for departures.
    """

    def __init__(self, score_fn: ScoreFn, name: str = "scoring",
                 temperatures: Mapping[str, float] | None = None) -> None:
        self.score_fn = score_fn
        self.name = name
        self.temperatures = dict(temperatures or {})

    def raw_scores(self, state: State, questions: Mapping[str, Question]) -> dict[str, dict[str, float]]:
        return {qname: {opt: float(self.score_fn(state, q, opt)) for opt in q.options}
                for qname, q in questions.items()}

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for qname, scores in self.raw_scores(state, questions).items():
            t = self.temperatures.get(qname, 1.0)
            out[qname] = Answer.from_probs(questions[qname], softmax(scores, t))
        return out


class ConstantJudge:
    """Returns the same distribution for everything. For tests and baselines."""

    name = "constant"

    def __init__(self, probs: Mapping[str, float] | None = None) -> None:
        self.probs = dict(probs or {})

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        return {qn: Answer.from_probs(q, self.probs or {o: 1.0 for o in q.options})
                for qn, q in questions.items()}
