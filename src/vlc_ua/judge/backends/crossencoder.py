"""Trained head: a cross-encoder scores (question + option, state) pairs.

This is the "own System One" backend. The same architecture the project
already runs in production for reranking (bge-reranker-v2-m3, ONNX int8 on
CPU) is fine-tuned so that for each option of a question it returns a
logit; softmax over the options gives the distribution, a per-question
temperature (fitted on held-out gold) makes it honest.

Why a cross-encoder and not a generative model:

* Ukrainian legal text is what bge-m3 was measured best on in this corpus;
* one forward pass per option, hundreds of milliseconds on the CPU box,
  no GPU in serving, no external quota;
* the training set is exactly the gold format of the eval harness, so
  "train" and "measure" never drift apart.

The trade-off is the one to keep in mind: a trained head answers only the
questions it was trained on. New questions go to the logprob reader or an
external teacher until enough labels exist.

Inference here is dependency-light: it works with ``onnxruntime`` +
``tokenizers`` (production) or with ``torch`` + ``transformers`` (dev). The
training script lives in :mod:`vlc_ua.judge.train.crossencoder`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..backend import ScoringJudge
from ..types import Choice, Noul, Question, Score, State


def render_state(state: State, max_chars: int = 6000) -> str:
    if isinstance(state, str):
        s = state
    elif isinstance(state, Mapping):
        s = "\n".join(f"{k}: {v}" for k, v in state.items())
    else:
        s = "\n".join(str(x) for x in state)
    return s[:max_chars]


def render_query(question: Question, option: str) -> str:
    """The "query" side of the pair: instruction plus the option's meaning."""
    if isinstance(question, Noul):
        stance = (question.true if option == "yes" else question.false) or option
        return f"{question.instructions} [{option}: {stance}]"
    if isinstance(question, Choice):
        desc = question.criteria.get(option) or ""
        return f"{question.instructions} [{option}: {desc}]"
    return f"{question.instructions} [рівень: {option}]"


@dataclass
class CrossEncoderHead:
    """Loads a fine-tuned head and exposes ``score(state, question, option)``."""

    model_dir: str
    max_length: int = 1024
    runtime: str = "auto"    # "onnx" | "torch" | "auto"

    def __post_init__(self) -> None:
        self.model_dir = str(self.model_dir)
        meta = Path(self.model_dir) / "head.json"
        self.meta: dict[str, Any] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        self._impl = None
        rt = self.runtime
        if rt == "auto":
            rt = "onnx" if (Path(self.model_dir) / "model.onnx").exists() else "torch"
        if rt == "onnx":
            self._impl = _OnnxImpl(self.model_dir, self.max_length)
        else:
            self._impl = _TorchImpl(self.model_dir, self.max_length)

    def score(self, state: State, question: Question, option: str) -> float:
        return float(self._impl.score(render_query(question, option), render_state(state)))

    def judge(self, name: str = "crossencoder", temperatures: Mapping[str, float] | None = None) -> ScoringJudge:
        temps = temperatures if temperatures is not None else self.meta.get("temperatures", {})
        return ScoringJudge(self.score, name=name, temperatures=temps)


class _OnnxImpl:
    def __init__(self, model_dir: str, max_length: int) -> None:
        import onnxruntime as ort  # optional dependency
        from tokenizers import Tokenizer

        self.sess = ort.InferenceSession(str(Path(model_dir) / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tok.enable_truncation(max_length)
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def score(self, query: str, doc: str) -> float:
        import numpy as np

        enc = self.tok.encode(query, doc)
        feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        out = self.sess.run(None, feeds)[0]
        return float(out.reshape(-1)[0])


class _TorchImpl:
    """Dev/eval runtime. Serving uses ONNX int8 on the CPU; this one follows
    the hardware it is given, because measuring a holdout pair-by-pair on a
    CPU costs hours and on the training GPU costs minutes — and both paths
    must stay inside the same harness, or "train" and "measure" drift apart.
    ``VLC_JUDGE_DEVICE`` overrides the choice (``cpu`` to force parity with
    production)."""

    def __init__(self, model_dir: str, max_length: int) -> None:
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.device = os.environ.get("VLC_JUDGE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def score(self, query: str, doc: str) -> float:
        with self.torch.no_grad():
            enc = self.tok(query, doc, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            return float(self.model(**enc).logits.reshape(-1)[0])
