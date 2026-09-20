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
