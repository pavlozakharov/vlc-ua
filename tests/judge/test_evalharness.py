"""Tests for evalharness.py: run, report, caching."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from vlc_ua.judge.backend import ConstantJudge
from vlc_ua.judge.types import Choice, Answer
from vlc_ua.judge import evalharness as ev


class TestLoadTask:
    """Test task loading."""

    def test_load_task_json(self, tmp_path):
        task_file = tmp_path / "task.json"
        task_data = {
            "q1": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {"a": "Option A", "b": "Option B"}
            }
        }
        task_file.write_text(json.dumps(task_data))

        task = ev.load_task(task_file)

        assert "q1" in task
        assert task["q1"].type == "choice"


class TestLoadGold:
    """Test gold file loading."""

    def test_load_gold_jsonl(self, tmp_path):
        gold_file = tmp_path / "gold.jsonl"
        rows = [
            {"id": "1", "state": "text1", "question": "q", "gold": "a"},
            {"id": "2", "state": "text2", "question": "q", "gold": "b"},
        ]
        gold_file.write_text("\n".join(json.dumps(r) for r in rows))

        gold = ev.load_gold(gold_file)

        assert len(gold) == 2
        assert gold[0]["id"] == "1"

    def test_load_gold_skips_empty_lines(self, tmp_path):
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text('{"id": "1", "state": "text1", "question": "q", "gold": "a"}\n\n{"id": "2", "state": "text2", "question": "q", "gold": "b"}')

        gold = ev.load_gold(gold_file)

        assert len(gold) == 2


class TestRun:
    """Test run() with backend execution and caching."""

    def test_run_basic(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge(probs={"a": 0.7, "b": 0.3})
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {"text": "test"}, "question": "q", "gold": "a"},
            {"id": "2", "state": {"text": "test2"}, "question": "q", "gold": "b"},
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir, limit=2)

        assert len(result.answers) == 2
        assert "1" in result.answers
        assert "2" in result.answers
        assert result.answers["1"].choice == "a"

    def test_run_caching(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge(probs={"a": 0.7, "b": 0.3})
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {"text": "test"}, "question": "q", "gold": "a"},
        ]

        # First run
        result1 = ev.run(judge, task, gold, cache_dir=cache_dir)
        assert len(result1.answers) == 1

        # Verify cache files exist
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) > 0

        # Second run should hit cache
        mock_judge = ConstantJudge()
        with patch.object(mock_judge, 'ask', side_effect=Exception("Should use cache")):
            result2 = ev.run(mock_judge, task, gold, cache_dir=cache_dir)

        assert len(result2.answers) == 1

    def test_run_no_cache(self, tmp_path):
        judge = ConstantJudge()
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {"text": "test"}, "question": "q", "gold": "a"},
        ]

        result = ev.run(judge, task, gold, cache_dir=None)

        assert len(result.answers) == 1

    def test_run_with_failures(self, tmp_path):
        cache_dir = tmp_path / "cache"

        def failing_ask(state, questions):
            raise ValueError("Backend error")

        judge = ConstantJudge()
        judge.ask = failing_ask

        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {"text": "test"}, "question": "q", "gold": "a"},
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir)

        assert len(result.failures) == 1
        assert "1" in result.failures
        assert "ValueError" in result.failures["1"]

    def test_run_respects_limit(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge()
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {"text": "test"}, "question": "q", "gold": "a"},
            {"id": "2", "state": {"text": "test2"}, "question": "q", "gold": "b"},
            {"id": "3", "state": {"text": "test3"}, "question": "q", "gold": "a"},
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir, limit=2)

        assert len(result.answers) == 2


class TestLabelled:
    """Test labelled() conversion."""

    def test_labelled_from_result(self, tmp_path):
        judge = ConstantJudge(probs={"a": 0.6, "b": 0.4})
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {}, "question": "q", "gold": "a", "sample": "random"},
            {"id": "2", "state": {}, "question": "q", "gold": "b", "sample": "enriched"},
        ]

        result = ev.run(judge, task, gold, cache_dir=None)
        labelled_items = ev.labelled(result, gold)

        assert len(labelled_items) == 2
        assert labelled_items[0].gold == "a"
        assert labelled_items[0].is_probability

    def test_labelled_filters_by_sample(self, tmp_path):
        judge = ConstantJudge()
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {}, "question": "q", "gold": "a", "sample": "random"},
            {"id": "2", "state": {}, "question": "q", "gold": "b", "sample": "enriched"},
        ]

        result = ev.run(judge, task, gold, cache_dir=None)

        random_items = ev.labelled(result, gold, sample="random")
        enriched_items = ev.labelled(result, gold, sample="enriched")

        assert len(random_items) == 1
        assert len(enriched_items) == 1


class TestReport:
    """Test report() generation."""

    def test_report_random_slice_only(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge(probs={"a": 0.8, "b": 0.2})
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": str(i), "state": {}, "question": "q", "gold": "a", "sample": "random"}
            for i in range(50)
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir)
        report = ev.report(result, gold)

        assert "random" in report
        assert "accuracy" in report["random"]
        assert "temperature" in report
        assert report["n_random"] > 0

    def test_report_enriched_slice(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge()
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": "1", "state": {}, "question": "q", "gold": "a", "sample": "random"},
            {"id": "2", "state": {}, "question": "q", "gold": "b", "sample": "enriched"},
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir)
        report = ev.report(result, gold)

        assert "enriched" in report
        assert report["n_enriched"] == 1

    def test_report_keys(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge = ConstantJudge(probs={"a": 0.8, "b": 0.2})
        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": str(i), "state": {}, "question": "q", "gold": "a", "sample": "random"}
            for i in range(50)
        ]

        result = ev.run(judge, task, gold, cache_dir=cache_dir)
        report = ev.report(result, gold)

        assert "backend" in report
        assert "task_version" in report
        assert "n_random" in report
        assert "n_enriched" in report
        assert "temperature" in report


class TestWinsLosses:
    """Test wins_losses comparison."""

    def test_wins_losses_both_right(self, tmp_path):
        cache_dir = tmp_path / "cache"

        # Both judges always predict correctly
        judge_a = ConstantJudge(probs={"a": 0.99, "b": 0.01})
        judge_b = ConstantJudge(probs={"a": 0.99, "b": 0.01})

        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": str(i), "state": {}, "question": "q", "gold": "a", "sample": "random"}
            for i in range(10)
        ]

        result_a = ev.run(judge_a, task, gold, cache_dir=cache_dir / "a")
        result_b = ev.run(judge_b, task, gold, cache_dir=cache_dir / "b")

        comp = ev.wins_losses(result_a, result_b, gold)

        assert comp["wins"] == 0
        assert comp["losses"] == 0
        assert comp["both_right"] == 10

    def test_wins_losses_clear_winner(self, tmp_path):
        cache_dir = tmp_path / "cache"

        judge_a = ConstantJudge(probs={"a": 0.99, "b": 0.01})
        judge_b = ConstantJudge(probs={"a": 0.01, "b": 0.99})

        task = {"q": Choice(instructions="Pick", criteria={"a": None, "b": None})}
        gold = [
            {"id": str(i), "state": {}, "question": "q", "gold": "a", "sample": "random"}
            for i in range(10)
        ]

        result_a = ev.run(judge_a, task, gold, cache_dir=cache_dir / "a")
        result_b = ev.run(judge_b, task, gold, cache_dir=cache_dir / "b")

        comp = ev.wins_losses(result_a, result_b, gold)

        assert comp["wins"] == 10
        assert comp["losses"] == 0


class TestCacheKeyCarriesTheBackendFingerprint:
    """A backend whose behaviour changed under an unchanged name used to serve
    its old answers as new. Measured on 21.09.2026: after the tie-break fix a
    run of keyword:departure_pair over 2 070 rows hit the default cache 2 070
    times, 352 of them (17.0%) with a different top choice."""

    def test_a_different_fingerprint_is_a_different_key(self):
        from vlc_ua.judge import evalharness as ev

        row = {"id": "r1", "state": {"s": "текст"}, "question": "q", "gold": "a"}
        a = ev._item_key("keyword:q", "v1", row, "rules-before")
        b = ev._item_key("keyword:q", "v1", row, "rules-after")

        assert a != b

    def test_no_fingerprint_keeps_the_legacy_key(self):
        from vlc_ua.judge import evalharness as ev

        row = {"id": "r1", "state": {"s": "текст"}, "question": "q", "gold": "a"}

        assert ev._item_key("keyword:q", "v1", row) == ev._item_key("keyword:q", "v1", row, "")

    def test_changed_rules_are_not_served_from_the_cache(self, tmp_path):
        from vlc_ua.judge import evalharness as ev
        from vlc_ua.judge.backend import ScoringJudge
        from vlc_ua.judge.types import question_from_dict

        task = {"q": question_from_dict({"type": "choice", "instructions": "i",
                                         "criteria": {"yes": "y", "no": "n"}})}
        gold = [{"id": "r1", "state": "текст", "question": "q", "gold": "yes"}]
        before = ScoringJudge(lambda s, q, o: 1.0 if o == "yes" else 0.0,
                              name="rule", fingerprint="v1")
        after = ScoringJudge(lambda s, q, o: 1.0 if o == "no" else 0.0,
                             name="rule", fingerprint="v2")

        ev.run(before, task, gold, cache_dir=tmp_path)
        res = ev.run(after, task, gold, cache_dir=tmp_path)
        probs = res.answers["r1"].probabilities

        assert probs["no"] > probs["yes"]

    def test_the_legacy_escape_hatch_is_opt_in(self, tmp_path):
        from vlc_ua.judge import evalharness as ev
        from vlc_ua.judge.backend import ScoringJudge
        from vlc_ua.judge.types import question_from_dict

        task = {"q": question_from_dict({"type": "choice", "instructions": "i",
                                         "criteria": {"yes": "y", "no": "n"}})}
        gold = [{"id": "r1", "state": "текст", "question": "q", "gold": "yes"}]
        legacy = ScoringJudge(lambda s, q, o: 1.0 if o == "yes" else 0.0, name="rule")
        fingerprinted = ScoringJudge(lambda s, q, o: 1.0 if o == "no" else 0.0,
                                     name="rule", fingerprint="v2")

        ev.run(legacy, task, gold, cache_dir=tmp_path)
        reused = ev.run(fingerprinted, task, gold, cache_dir=tmp_path, accept_legacy_cache=True)

        assert reused.answers["r1"].probabilities["yes"] > 0.5


class TestTheCacheHoldsLogitsNotATemperature:
    """Until 23.09.2026 the cache and the run file held tempered probabilities
    under a key without the temperature: after ``calibrate --write`` a rerun
    was served the OLD temperature, and the run could not say which one it
    carried. Measured: the int8 run of head v6 carried T 0.9810 and was
    calibrated as if it carried none."""

    def _setup(self, temperature):
        from vlc_ua.judge.backend import ScoringJudge
        from vlc_ua.judge.types import question_from_dict

        task = {"q": question_from_dict({"type": "choice", "instructions": "i",
                                         "criteria": {"yes": "y", "no": "n"}})}
        gold = [{"id": "r1", "state": "текст", "question": "q", "gold": "yes"}]
        judge = ScoringJudge(lambda s, q, o: 2.0 if o == "yes" else 0.0, name="head",
                             temperatures={"q": temperature}, fingerprint="head-v1")
        return task, gold, judge

    def test_a_changed_temperature_applies_to_cached_rows(self, tmp_path):
        import math

        task, gold, cold = self._setup(1.0)
        ev.run(cold, task, gold, cache_dir=tmp_path)
        _, _, hot = self._setup(2.0)
        hot.score_fn = lambda s, q, o: (_ for _ in ()).throw(AssertionError("must come from cache"))

        res = ev.run(hot, task, gold, cache_dir=tmp_path)

        assert res.answers["r1"].probabilities["yes"] == pytest.approx(1 / (1 + math.exp(-1.0)))

    def test_the_run_records_logits_and_the_temperature_it_applied(self, tmp_path):
        task, gold, judge = self._setup(0.5)

        res = ev.run(judge, task, gold, cache_dir=tmp_path)

        assert res.scores["r1"] == {"yes": 2.0, "no": 0.0}
        assert res.temperatures == {"q": 0.5}
        assert ev.logits_for(res, "r1", None, "q") == {"yes": 2.0, "no": 0.0}

    def test_an_entry_without_logits_is_a_miss_for_a_scoring_backend(self, tmp_path):
        import json

        task, gold, judge = self._setup(1.0)
        key = ev._item_key("head", "v1", gold[0], "head-v1")
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"probabilities": {"yes": 0.01, "no": 0.99}, "seconds": 1.0}), encoding="utf-8")

        res = ev.run(judge, task, gold, cache_dir=tmp_path)

        assert res.answers["r1"].probabilities["yes"] > 0.5
        assert "r1" in res.scores

    def test_under_the_legacy_hatch_such_rows_have_no_logits(self, tmp_path):
        import json

        task, gold, judge = self._setup(1.0)
        key = ev._item_key("head", "v1", gold[0], "head-v1")
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"probabilities": {"yes": 0.3, "no": 0.7}, "seconds": 1.0}), encoding="utf-8")

        res = ev.run(judge, task, gold, cache_dir=tmp_path, accept_legacy_cache=True)

        assert "r1" not in res.scores
        # the run's own temperature is NOT this row's: it was made at some other one
        assert ev.logits_for(res, "r1", None, "q") is None
        assert ev.logits_for(res, "r1", {"q": 2.0}, "q") is not None

    def test_a_basis_is_one_basis(self, tmp_path):
        task, gold, judge = self._setup(1.0)
        res = ev.run(judge, task, gold + [{"id": "r2", "state": "інше", "question": "q", "gold": "no"}],
                     cache_dir=tmp_path)
        del res.scores["r2"]

        items, basis = ev.labelled_with_basis(res, gold + [{"id": "r2", "question": "q", "gold": "no"}])

        assert basis == "probabilities"
        assert all(it.is_probability for it in items)
