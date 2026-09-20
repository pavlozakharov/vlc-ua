"""Logprob reader: typed decisions from any chat model that exposes logprobs.

The model is asked a multiple-choice question and forced to answer with one
letter. The distribution over letters is read from the ``top_logprobs`` of
the first generated token, not from the text. That gives a true model
distribution (calibratable with a temperature) instead of a self-reported
"confidence: 0.8" which is not calibrated at all.

Works with any OpenAI-compatible chat completions endpoint that honours
``logprobs``/``top_logprobs`` (vLLM, llama.cpp server, OpenAI, some cloud
providers). Endpoints that ignore ``logprobs`` are detected: the answer then
degrades to a one-hot on the sampled letter and ``degraded=True`` is set, so
the eval harness can refuse to calibrate on it.

No provider keys are read from files here: pass ``api_key`` explicitly or
via the environment variable named in ``api_key_env``.
"""
from __future__ import annotations

import json
import math
import os
import string
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from ..backend import softmax
from ..types import Answer, Choice, Noul, Question, Score, State

LETTERS = string.ascii_uppercase


def render_state(state: State) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=1)


def build_prompt(state: State, question: Question, lang: str = "uk") -> tuple[str, list[str]]:
    """Return (prompt, option_keys) with options lettered A, B, C..."""
    opts = list(question.options)
    if len(opts) > len(LETTERS):
        raise ValueError("too many options for a single-letter answer")
    if isinstance(question, Noul):
        labels = {"yes": question.true or ("так" if lang == "uk" else "yes"),
                  "no": question.false or ("ні" if lang == "uk" else "no")}
        lines = [f"{LETTERS[i]}. {labels[o]}" for i, o in enumerate(opts)]
        head = question.instructions
    elif isinstance(question, Choice):
        lines = []
        for i, o in enumerate(opts):
            desc = question.criteria.get(o)
            lines.append(f"{LETTERS[i]}. {o}" + (f": {desc}" if desc else ""))
        head = question.instructions
    else:
        lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)]
        head = question.instructions + (" Рівні впорядковано від найнижчого до найвищого."
                                        if lang == "uk" else
                                        " Levels are ordered from lowest to highest.")
    tail = ("Відповідай рівно однією літерою варіанта, без пояснень."
            if lang == "uk" else "Answer with exactly one option letter, nothing else.")
    prompt = (f"ДАНІ:\n{render_state(state)}\n\nПИТАННЯ: {head}\n\nВАРІАНТИ:\n"
              + "\n".join(lines) + f"\n\n{tail}\nВідповідь:")
    return prompt, opts


@dataclass
class LogprobJudge:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local"
    api_key: str | None = None
    api_key_env: str = "LLM_API_KEY"
    top_logprobs: int = 20
    timeout: float = 60.0
    temperatures: Mapping[str, float] | None = None
    lang: str = "uk"
    system: str = ("Ти — класифікатор. Ти не пишеш пояснень, лише обираєш варіант.")
    name: str = "logprob"
    user_agent: str = "vlc-ua-judge/0.1"
    note: str = ""   # why a call degraded, filled per request

    def _key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "User-Agent": self.user_agent}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # "HTTP Error 400: Bad Request" says nothing: a wrong model name, an
            # exhausted quota and an unsupported parameter all look the same.
            # Carry the provider's own message into the error text.
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                detail = ""
            raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {detail}",
                                         exc.headers, None) from None

    def letter_logprobs(self, prompt: str) -> tuple[dict[str, float], bool, str]:
        """(logprob per letter, degraded, sampled_text)."""
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 2,
            "logprobs": True, "top_logprobs": self.top_logprobs,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": prompt}],
        }
        self.note = ""
        try:
            resp = self._post(body)
        except urllib.error.HTTPError as exc:
            # A provider that refuses the parameter outright (Groq 400:
            # "`logprobs` is not supported with this model"; Cohere 422:
            # "top_logprobs is not supported") is not a dead channel: it is a
            # one-hot teacher. Retry without the parameter and mark it degraded,
            # so the caller can tell "no distribution" from "no channel".
            if exc.code in (400, 422) and "logprob" in str(exc).lower():
                self.note = "endpoint rejects the logprobs parameter"
                plain = {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs")}
                resp = self._post(plain)
            else:
                raise
        choice = resp["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        content = ((choice.get("logprobs") or {}).get("content")) or []
        out: dict[str, float] = {}
        if content:
            first = content[0]
            cands = list(first.get("top_logprobs") or [])
            if first.get("token") is not None and first.get("logprob") is not None:
                cands.append({"token": first["token"], "logprob": first["logprob"]})
            for c in cands:
                tok = (c.get("token") or "").strip().upper()
                if len(tok) == 1 and tok in LETTERS:
                    lp = float(c["logprob"])
                    out[tok] = max(out.get(tok, -math.inf), lp)
        if out:
            return out, False, text
        # Endpoint gave no usable logprobs: degrade to a one-hot on the text.
        letter = text.strip()[:1].upper()
        return ({letter: 0.0} if letter in LETTERS else {}), True, text

    def raw_scores(self, state: State, question: Question) -> tuple[dict[str, float], bool]:
        prompt, opts = build_prompt(state, question, self.lang)
        lps, degraded, _ = self.letter_logprobs(prompt)
        floor = min(lps.values()) - 5.0 if lps else 0.0
        scores = {o: lps.get(LETTERS[i], floor) for i, o in enumerate(opts)}
        return scores, degraded

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        temps = dict(self.temperatures or {})
        out: dict[str, Answer] = {}
        for qn, q in questions.items():
            scores, degraded = self.raw_scores(state, q)
            probs = softmax(scores, temps.get(qn, 1.0))
            ans = Answer.from_probs(q, probs)
            if degraded:
                ans = Answer(**{**ans.__dict__, "confidence": 0.0})
            out[qn] = ans
        return out
