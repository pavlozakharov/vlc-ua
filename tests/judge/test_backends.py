"""Tests for backend implementations: ScoringJudge, ConstantJudge, keyword, logprob, external."""
import json
import math
import pytest
from unittest.mock import MagicMock, patch

from vlc_ua.judge.types import Noul, Choice, Score, Answer
from vlc_ua.judge.backend import ScoringJudge, ConstantJudge, softmax
from vlc_ua.judge.backends.keyword import make_keyword_judge
from vlc_ua.judge.backends.logprob import LogprobJudge, build_prompt, LETTERS
from vlc_ua.judge.backends.external import answers_from_wire


class TestScoringJudge:
    """Test ScoringJudge with a toy score function."""

    def test_scoring_judge_basic(self):
        def score_fn(state, question, option):
            return 1.0 if option == "a" else 0.0

        judge = ScoringJudge(score_fn, name="test")
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        answers = judge.ask({"text": "test"}, {"q1": q})

        assert "q1" in answers
        assert answers["q1"].choice == "a"
        assert answers["q1"].type == "choice"

    def test_scoring_judge_temperatures(self):
        def score_fn(state, question, option):
            return {"a": 2.0, "b": 1.0}[option]

        temps = {"q1": 0.5}
        judge = ScoringJudge(score_fn, name="test", temperatures=temps)

        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        answers = judge.ask({"text": "test"}, {"q1": q})

        # Low temperature should sharpen the distribution
        assert answers["q1"].probabilities["a"] > 0.5

    def test_scoring_judge_noul(self):
        def score_fn(state, question, option):
            return 1.0 if option == "yes" else 0.5

        judge = ScoringJudge(score_fn)
        q = Noul(instructions="Is it true?")
        answers = judge.ask({}, {"q": q})

        ans = answers["q"]
        assert ans.type == "noul"
        # softmax gives e^1.0 / (e^1.0 + e^0.5) for yes
        assert ans.p == pytest.approx(math.exp(1.0) / (math.exp(1.0) + math.exp(0.5)))

    def test_scoring_judge_score(self):
        def score_fn(state, question, option):
            levels = ["low", "medium", "high"]
            idx = levels.index(option)
            return float(idx)

        judge = ScoringJudge(score_fn)
        q = Score(instructions="Rate", levels=["low", "medium", "high"])
        answers = judge.ask({}, {"q": q})

        ans = answers["q"]
        assert ans.type == "score"
        assert ans.choice == "high"

    def test_scoring_judge_raw_scores(self):
        def score_fn(state, question, option):
            return {"a": 1.0, "b": 2.0}[option]

        judge = ScoringJudge(score_fn)
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        scores = judge.raw_scores({}, {"q": q})

        assert scores["q"] == {"a": 1.0, "b": 2.0}


class TestConstantJudge:
    """Test ConstantJudge for tests and baselines."""

    def test_constant_judge_default(self):
        judge = ConstantJudge()
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        answers = judge.ask({}, {"q": q})

        assert answers["q"].type == "choice"
        assert answers["q"].choice == "a"  # First option

    def test_constant_judge_custom_probs(self):
        judge = ConstantJudge(probs={"yes": 0.7, "no": 0.3})
        q = Noul(instructions="Is it true?")
        answers = judge.ask({}, {"q": q})

        assert answers["q"].p == 0.7

    def test_constant_judge_multiple_questions(self):
        judge = ConstantJudge(probs={"a": 0.6, "b": 0.4})
        q1 = Choice(instructions="Pick", criteria={"a": None, "b": None})
        q2 = Choice(instructions="Pick", criteria={"a": None, "b": None})

        answers = judge.ask({}, {"q1": q1, "q2": q2})

        assert len(answers) == 2
        assert answers["q1"].choice == "a"
        assert answers["q2"].choice == "a"


class TestKeywordJudge:
    """Test keyword judge for attribution."""

    def test_keyword_judge_court_attribution(self):
        judge = make_keyword_judge("attribution")
        q = Choice(instructions="Who?", criteria={
            "court": None, "party": None, "lower": None, "facts": None, "procedural": None
        })

        # Text mentioning "Верховний Суд виходить з того"
        text = "Верховний Суд виходить з того, що это важно."
        answers = judge.ask({"fragment": text}, {"q": q})

        assert answers["q"].choice == "court"

    def test_keyword_judge_party_attribution(self):
        judge = make_keyword_judge("attribution")
        q = Choice(instructions="Who?", criteria={
            "court": None, "party": None, "lower": None, "facts": None, "procedural": None
        })

        # Text mentioning "скаржник зазначає"
        text = "скаржник зазначає, що его права порушені."
        answers = judge.ask({"fragment": text}, {"q": q})

        assert answers["q"].choice == "party"

    def test_keyword_judge_dict_state(self):
        judge = make_keyword_judge("attribution")
        q = Choice(instructions="Who?", criteria={
            "court": None, "party": None, "lower": None, "facts": None, "procedural": None
        })

        state = {"fragment": "Верховний Суд", "other": "info"}
        answers = judge.ask(state, {"q": q})

        assert answers["q"].choice == "court"

    def test_keyword_judge_no_matches(self):
        judge = make_keyword_judge("attribution")
        q = Choice(instructions="Who?", criteria={
            "court": None, "party": None, "lower": None, "facts": None, "procedural": None
        })

        text = "This has nothing to do with attribution."
        answers = judge.ask({"fragment": text}, {"q": q})

        # Should pick one (lowest score defaults to "procedural" or alphabetically first)
        assert answers["q"].choice in q.options


class TestLogprobJudge:
    """Test LogprobJudge with mocked responses."""

    def test_build_prompt_noul_ukrainian(self):
        q = Noul(instructions="Is it true?", true="Так", false="Ні")
        state = "Some text"
        prompt, opts = build_prompt(state, q, lang="uk")

        assert "ДАНІ:" in prompt
        assert "ПИТАННЯ:" in prompt
        assert "ВАРІАНТИ:" in prompt
        assert "A. Так" in prompt
        assert "B. Ні" in prompt
        assert opts == ["yes", "no"]

    def test_build_prompt_choice_ukrainian(self):
        q = Choice(instructions="Pick one", criteria={"a": "Option A", "b": "Option B"})
        state = "Some text"
        prompt, opts = build_prompt(state, q, lang="uk")

        assert "A. a: Option A" in prompt
        assert "B. b: Option B" in prompt
        assert opts == ["a", "b"]

    def test_build_prompt_score_ukrainian(self):
        q = Score(instructions="Rate", levels=["low", "high"])
        state = "Some text"
        prompt, opts = build_prompt(state, q, lang="uk")

        assert "Рівні впорядковано від найнижчого до найвищого" in prompt
        assert "A. low" in prompt
        assert "B. high" in prompt

    def test_logprob_judge_letter_logprobs_with_mock(self):
        """Mock the _post method to test letter_logprobs parsing."""
        judge = LogprobJudge()

        # Mock response with proper logprobs structure
        mock_response = {
            "choices": [{
                "message": {"content": "A"},
                "logprobs": {
                    "content": [{
                        "token": "A",
                        "logprob": -0.5,
                        "top_logprobs": [
                            {"token": " B", "logprob": -1.0},
                            {"token": " C", "logprob": -1.5}
                        ]
                    }]
                }
            }]
        }

        with patch.object(judge, '_post', return_value=mock_response):
            logprobs, degraded, text = judge.letter_logprobs("test prompt")

        assert not degraded
        assert "A" in logprobs
        assert "B" in logprobs
        assert "C" in logprobs
        assert text == "A"

    def test_logprob_judge_degraded_case(self):
        """Test degradation when no logprobs available."""
        judge = LogprobJudge()

        # Mock response without logprobs
        mock_response = {
            "choices": [{
                "message": {"content": "A"},
                "logprobs": {"content": []}
            }]
        }

        with patch.object(judge, '_post', return_value=mock_response):
            logprobs, degraded, text = judge.letter_logprobs("test prompt")

        assert degraded
        assert logprobs == {"A": 0.0}

    def test_logprob_judge_degraded_invalid_letter(self):
        """Test degradation with invalid letter response."""
        judge = LogprobJudge()

        # Mock response with invalid letter
        mock_response = {
            "choices": [{
                "message": {"content": "Invalid"},
                "logprobs": {"content": []}
            }]
        }

        with patch.object(judge, '_post', return_value=mock_response):
            logprobs, degraded, text = judge.letter_logprobs("test prompt")

        assert degraded
        # "Invalid" starts with "I" which is in LETTERS, so degraded one-hot
        assert logprobs == {"I": 0.0}

    def test_logprob_judge_full_ask(self):
        """Test full ask workflow with mocked endpoint."""
        judge = LogprobJudge(model="test-model")

        q = Choice(instructions="Pick", criteria={"a": None, "b": None})

        mock_response = {
            "choices": [{
                "message": {"content": "A"},
                "logprobs": {
                    "content": [{
                        "token": "A",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": " B", "logprob": -2.0}
                        ]
                    }]
                }
            }]
        }

        with patch.object(judge, '_post', return_value=mock_response):
            answers = judge.ask({"text": "test"}, {"q": q})

        assert "q" in answers
        assert answers["q"].choice == "a"
        assert not hasattr(answers["q"], 'degraded') or answers["q"].confidence > 0


class TestExternalJudge:
    """Test external judge wire protocol."""

    def test_answers_from_wire_noul(self):
        q = Noul(instructions="Is it true?")
        wire = {
            "q": {
                "type": "noul",
                "noul": 0.8
            }
        }

        answers = answers_from_wire({"q": q}, wire)

        assert answers["q"].type == "noul"
        assert answers["q"].p == 0.8

    def test_answers_from_wire_noul_missing_field(self):
        q = Noul(instructions="Is it true?")
        wire = {
            "q": {
                "type": "noul"
            }
        }

        answers = answers_from_wire({"q": q}, wire)

        # Should default to 0.5
        assert answers["q"].p == 0.5

    def test_answers_from_wire_choice(self):
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        wire = {
            "q": {
                "type": "choice",
                "probabilities": {"a": 0.3, "b": 0.7}
            }
        }

        answers = answers_from_wire({"q": q}, wire)

        assert answers["q"].type == "choice"
        assert answers["q"].choice == "b"

    def test_answers_from_wire_missing_option(self):
        """Missing option in wire should be treated as 0 probability."""
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        wire = {
            "q": {
                "type": "choice",
                "probabilities": {"a": 0.7}
            }
        }

        answers = answers_from_wire({"q": q}, wire)

        # Probabilities should be normalized
        assert answers["q"].probabilities["a"] == pytest.approx(1.0)
        assert answers["q"].probabilities["b"] == pytest.approx(0.0)

    def test_answers_from_wire_confidence_recomputed(self):
        """Confidence should be recomputed from distribution."""
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        wire = {
            "q": {
                "type": "choice",
                "probabilities": {"a": 0.9, "b": 0.1}
            }
        }

        answers = answers_from_wire({"q": q}, wire)

        # Confidence should be recomputed, not taken from wire
        assert answers["q"].confidence > 0.5


class TestKeywordFallbackWinsTheTie:
    """The docstring promised it; until 21.09.2026 the code did not do it."""

    def test_fallback_option_wins_when_no_rule_fires(self):
        from vlc_ua.judge.backends.keyword import make_keyword_judge
        from vlc_ua.judge.evalharness import load_task

        judge = make_keyword_judge("departure_pair")
        task = load_task("/srv/work/judge/departure_pair.task.json")
        state = {"sentence": "Текст без жодної ознаки відступу.", "target_case": "1/2/20"}

        ans = judge.ask(state, task)["departure_pair"]
        top = max(ans.probabilities, key=ans.probabilities.__getitem__)

        assert top == "other"

    def test_a_firing_rule_still_beats_the_fallback(self):
        from vlc_ua.judge.backends.keyword import make_keyword_judge
        from vlc_ua.judge.evalharness import load_task

        judge = make_keyword_judge("departure_pair")
        task = load_task("/srv/work/judge/departure_pair.task.json")
        state = {"sentence": "Велика Палата відступила від висновку у справі № 1/2/20.",
                 "target_case": "1/2/20"}

        ans = judge.ask(state, task)["departure_pair"]

        assert max(ans.probabilities, key=ans.probabilities.__getitem__) == "departure"


class TestOnnxThreadBudget:
    """Free speed measured 21.09.2026: 6 threads 6.03 s/question against
    9.99 s when onnxruntime helps itself to all eight, identical logits."""

    def test_leaves_two_cores_to_the_machine(self, monkeypatch):
        from vlc_ua.judge.backends import crossencoder as ce

        monkeypatch.delenv("VLC_JUDGE_ONNX_THREADS", raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 8)

        assert ce.onnx_threads() == 6

    def test_env_overrides_and_zero_means_let_ort_decide(self, monkeypatch):
        from vlc_ua.judge.backends import crossencoder as ce

        monkeypatch.setenv("VLC_JUDGE_ONNX_THREADS", "3")
        assert ce.onnx_threads() == 3

        monkeypatch.setenv("VLC_JUDGE_ONNX_THREADS", "0")
        assert ce.onnx_threads() == 0

    def test_never_asks_for_zero_cores_on_a_tiny_box(self, monkeypatch):
        from vlc_ua.judge.backends import crossencoder as ce

        monkeypatch.delenv("VLC_JUDGE_ONNX_THREADS", raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 1)

        assert ce.onnx_threads() == 1
