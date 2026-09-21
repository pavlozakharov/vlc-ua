"""Tests for the fixes made while building the first gold sets on the server.

Each test pins one defect measured on 2026-09-20 against real data, so a
regression shows up as a failing test instead of as a quietly wrong gold.
"""
import json
import urllib.error

import pytest

from vlc_ua.judge.gold import attribution as attr
from vlc_ua.judge.gold import departures as dep
from vlc_ua.judge.backends.logprob import LogprobJudge
from vlc_ua.judge.types import Choice


class TestAttributionIds:
    """155 ids repeated because start offset + length is not unique."""

    def test_fragments_of_one_section_get_distinct_ids(self, tmp_path):
        head = "Позиція Верховного Суду\n"
        para = "а" * 600
        text = head + "\n\n".join([para, para, para]) + "\n"
        out = tmp_path / "gold.jsonl"
        attr.build([("doc1", text)], out, per_doc=6)
        ids = [json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)), ids
        assert len(ids) >= 2


class TestDeparturesIds:
    def test_digest_is_stable_across_calls(self):
        assert dep._digest("речення") == dep._digest("речення")
        assert dep._digest("а") != dep._digest("б")

    def test_write_drops_repeated_ids(self, tmp_path):
        rows = [{"id": "x", "gold": "departure"}, {"id": "x", "gold": "other"},
                {"id": "y", "gold": "other"}]
        out = tmp_path / "g.jsonl"
        assert dep.write(rows, out) == 2


class TestEvidenceLabel:
    """Direction of the pair is the whole question: same verb, opposite label."""

    def test_target_after_performative_is_departure(self):
        s = ("Велика Палата Верховного Суду відступила від висновку, викладеного "
             "у постанові у справі № 910/1111/20.")
        assert dep.evidence_label(s, "910/1111/20")[0] == "departure"

    def test_target_before_performative_is_other(self):
        s = ("У постанові від 01.02.2021 у справі № 910/1111/20 об'єднана палата "
             "відступила від висновку, викладеного у постанові у справі № 905/2222/19.")
        assert dep.evidence_label(s, "910/1111/20")[0] == "other"

    def test_refusal_wins_over_the_verb(self):
        s = ("Верховний Суд не вбачає підстав для відступу від висновку, викладеного "
             "у постанові у справі № 910/1111/20.")
        assert dep.evidence_label(s, "910/1111/20")[0] == "refusal"

    def test_referral_as_a_noun_is_not_a_norm_quote(self):
        s = ("Суд дійшов висновку про необхідність передачі справи № 902/575/18 на "
             "розгляд об'єднаної палати на підставі частини 2 статті 302 ГПК України, "
             "оскільки вважає за необхідне відступити від висновку у справі № 924/853/18.")
        assert dep.evidence_label(s, "924/853/18")[0] == "other"

    def test_bare_intent_is_not_labelled(self):
        s = ("Колегія суддів вважає за необхідне відступити від висновку, викладеного "
             "у постанові у справі № 910/1111/20.")
        assert dep.evidence_label(s, "910/1111/20")[0] is None

    def test_missing_target_is_not_labelled(self):
        s = "Велика Палата Верховного Суду відступила від висновку, викладеного у постанові"
        label, why = dep.evidence_label(s, "910/1111/20")
        assert label is None and "цілі немає" in why

    def test_relabel_keeps_only_decidable_rows(self):
        rows = [
            {"id": "a", "state": {"sentence": "Велика Палата відступила від висновку у справі № 1/2/20",
                                  "target_case": "1/2/20"}, "gold": "other", "source": "grammar"},
            {"id": "b", "state": {"sentence": "Жодної ознаки тут немає, справа № 3/4/21",
                                  "target_case": "3/4/21"}, "gold": "departure", "source": "grammar"},
        ]
        out = list(dep.relabel_by_evidence(rows))
        assert [r["id"] for r in out] == ["a"]
        assert out[0]["gold"] == "departure"
        assert out[0]["source"].startswith("evidence(departure)")


class TestLogprobFallback:
    """A provider that refuses the parameter is a one-hot teacher, not a dead channel."""

    def _judge(self, code, body, ok_payload):
        j = LogprobJudge(base_url="http://x/v1", model="m", api_key="k")
        calls = []

        def fake_post(payload):
            calls.append(payload)
            if "logprobs" in payload:
                raise urllib.error.HTTPError("http://x", code, f"Bad Request: {body}", None, None)
            return ok_payload

        j._post = fake_post
        j._calls = calls
        return j

    @pytest.mark.parametrize("code", [400, 422])
    def test_retries_without_logprobs_and_marks_degraded(self, code):
        payload = {"choices": [{"message": {"content": "A"}}]}
        j = self._judge(code, "`logprobs` is not supported with this model", payload)
        scores, degraded = j.raw_scores("текст", Choice(instructions="?", criteria={"a": None, "b": None}))
        assert degraded is True
        assert j.note
        assert len(j._calls) == 2 and "logprobs" not in j._calls[1]
        assert scores["a"] > scores["b"]

    def test_unrelated_400_still_raises(self):
        j = self._judge(400, "model not found", {})
        with pytest.raises(urllib.error.HTTPError):
            j.raw_scores("текст", Choice(instructions="?", criteria={"a": None, "b": None}))


class TestTrainingStepActuallyLearns:
    """Frozen embeddings + checkpointing is the combination that silently
    trains nothing: the step runs, the loss prints, no weight moves."""

    def _tiny(self, tmp_path, **flags):
        import json
        import torch
        from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification
        from vlc_ua.judge.train import crossencoder as ce

        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        cfg = XLMRobertaConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=64,
                               max_position_embeddings=130, num_labels=1, type_vocab_size=1)
        base = tmp_path / "base"
        XLMRobertaForSequenceClassification(cfg).save_pretrained(base)
        tok.save_pretrained(base)

        task = {"attribution": {"type": "choice", "instructions": "розділ?",
                                "criteria": {"court": "суд", "party": "сторона"}}}
        (tmp_path / "task.json").write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        rows = [{"id": f"r{i}", "state": {"fragment": "Верховний Суд зазначає, що " * 5},
                 "question": "attribution", "gold": "court" if i % 2 else "party",
                 "sample": "random"} for i in range(8)]
        (tmp_path / "gold.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

        out = tmp_path / "head"
        argv = ["--gold", str(tmp_path / "gold.jsonl"), "--task", str(tmp_path / "task.json"),
                "--base", str(base), "--out", str(out), "--epochs", "1", "--batch-rows", "2",
                "--max-length", "64", "--dev-share", "0.25", "--lr", "1e-3"]
        for f in flags.get("flags", []):
            argv.append(f)
        before = {k: v.clone() for k, v in
                  XLMRobertaForSequenceClassification.from_pretrained(base).state_dict().items()}
        ce.main(argv)
        after = XLMRobertaForSequenceClassification.from_pretrained(out).state_dict()
        return before, after, torch

    def test_weights_move_with_frozen_embeddings_and_checkpointing(self, tmp_path):
        before, after, torch = self._tiny(
            tmp_path, flags=["--freeze-embeddings", "--grad-checkpointing"])
        moved = [k for k in after
                 if k in before and not torch.equal(before[k].float(), after[k].float())]
        assert moved, "no weight changed: the training step did nothing"
        emb = [k for k in moved if "word_embeddings" in k]
        assert not emb, f"frozen embedding moved: {emb}"


class TestRuntimeFlag:
    """--runtime існує і парситься.

    Прапорець додавався у _backend_args і з першого разу пішов з чужим іменем
    парсера (`r.add_argument` замість `p.add_argument`) — CLI падав на імпорті
    в КОЖНІЙ підкоманді, не лише в run. Дешевий прогін --help це ловить.
    """

    def test_run_help_lists_runtime(self):
        import subprocess
        import sys
        from pathlib import Path as _P

        root = str(_P(__file__).resolve().parents[2])
        out = subprocess.run([sys.executable, "-m", "vlc_ua.judge.cli", "run", "--help"],
                             capture_output=True, text=True, cwd=root,
                             env={"PYTHONPATH": root + "/src", "PATH": "/usr/bin:/bin"})
        assert out.returncode == 0, out.stderr
        assert "--runtime" in out.stdout
        assert "torch" in out.stdout
