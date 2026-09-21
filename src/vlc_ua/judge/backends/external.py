"""External System One services: TypeSafe direct and the Cloudflare route.

Both speak the same schema: ``{"state": ..., "questions": {name: {...}}}``
in, ``{"answers": {name: {...}}}`` out (Cloudflare wraps it in ``result``).
These clients exist for two reasons only: to use the service as a *teacher*
when bootstrapping labels, and to compare it against local heads in the
eval harness. Nothing in the production path should depend on them.

Keys are taken from explicit arguments or environment variables. Nothing is
read from disk.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from ..types import Answer, Question, State, question_to_dict


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "vlc-ua-judge/0.1", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def answers_from_wire(questions: Mapping[str, Question], wire: Mapping[str, Any]) -> dict[str, Answer]:
    """Convert service answers to :class:`Answer`, recomputing confidence.

    Confidence is recomputed from the distribution on purpose: gateways drop
    the field, and one definition across backends keeps thresholds portable.
    """
    out: dict[str, Answer] = {}
    for qn, q in questions.items():
        a = wire.get(qn) or {}
        if a.get("type") == "noul" or "noul" in a:
            p = float(a.get("noul", 0.5))
            out[qn] = Answer.from_probs(q, {"yes": p, "no": 1.0 - p})
        else:
            probs = a.get("probabilities") or {}
            out[qn] = Answer.from_probs(q, {k: float(v) for k, v in probs.items()})
    return out


@dataclass
class TypeSafeJudge:
    """Direct TypeSafe API. ``TYPESAFE_API_KEY`` or ``api_key``."""

    model: str = "jev-1.13.0"   # pin a version; the alias drifts
    api_key: str | None = None
    base_url: str = "https://api.typesafe.ai/v1/systemone"
    timeout: float = 30.0
    zero_data_retention: bool = True
    name: str = "typesafe"

    @property
    def fingerprint(self) -> str:
        """Which remote model, at which endpoint — the cache must not mix them."""
        return f"{self.base_url}|{self.model}"

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        body: dict[str, Any] = {"model": self.model, "state": state,
                                "questions": {qn: question_to_dict(q) for qn, q in questions.items()}}
        headers = {"Authorization": f"Bearer {key}"}
        if self.zero_data_retention:
            headers["X-Zero-Data-Retention"] = "true"   # verify header name against docs
        resp = _post(self.base_url, body, headers, self.timeout)
        return answers_from_wire(questions, resp.get("answers") or {})


@dataclass
class CloudflareJudge:
    """Cloudflare AI route for ``typesafe/jev`` (32k context on this route).

    Needs a Cloudflare API token with AI Gateway permission and either a
    TypeSafe key stored as BYOK in the gateway or prepaid AI Gateway credits.
    Not covered by the free Workers AI neurons.
    """

    account_id: str | None = None
    api_token: str | None = None
    model: str = "typesafe/jev"
    timeout: float = 30.0
    name: str = "cloudflare-jev"

    @property
    def fingerprint(self) -> str:
        """Which remote model, at which endpoint — the cache must not mix them."""
        return f"{self.model}"

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        acct = self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        tok = self.api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        if not (acct and tok):
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not set")
        url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run"
        body = {"model": self.model,
                "input": {"state": state,
                          "questions": {qn: question_to_dict(q) for qn, q in questions.items()}}}
        resp = _post(url, body, {"Authorization": f"Bearer {tok}"}, self.timeout)
        result = resp.get("result") or {}
        return answers_from_wire(questions, result.get("answers") or result)


@dataclass
class SystemOneHTTPJudge:
    """Any local server speaking the System One contract (kev, luce serve, ours)."""

    base_url: str = "http://127.0.0.1:8009/v1/systemone"
    model: str = "local"
    timeout: float = 60.0
    name: str = "systemone-http"

    @property
    def fingerprint(self) -> str:
        """Which remote model, at which endpoint — the cache must not mix them."""
        return f"{self.base_url}|{self.model}"

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        body = {"model": self.model, "state": state,
                "questions": {qn: question_to_dict(q) for qn, q in questions.items()}}
        resp = _post(self.base_url, body, {}, self.timeout)
        return answers_from_wire(questions, resp.get("answers") or {})
