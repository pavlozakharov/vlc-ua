"""Tests for types.py: Noul, Choice, Score, Answer, and helpers."""
import math
import pytest
from vlc_ua.judge.types import (
    Noul, Choice, Score, Answer, normalize, confidence_of,
    question_from_dict, question_to_dict
)


class TestNoul:
    """Test Noul yes/no questions."""

    def test_noul_creation(self):
        q = Noul(instructions="Is it true?")
        assert q.instructions == "Is it true?"
        assert q.true is None
        assert q.false is None
        assert q.options == ("yes", "no")
        assert q.type == "noul"

    def test_noul_with_clarifications(self):
        q = Noul(instructions="Is it true?", true="Indeed", false="Nope")
        assert q.true == "Indeed"
        assert q.false == "Nope"
        assert q.options == ("yes", "no")

    def test_noul_frozen(self):
        q = Noul(instructions="Is it true?")
        with pytest.raises(AttributeError):
            q.instructions = "Something else"


class TestChoice:
    """Test Choice questions."""

    def test_choice_creation(self):
        q = Choice(instructions="Pick one", criteria={"a": "Option A", "b": "Option B"})
        assert q.instructions == "Pick one"
        assert q.criteria == {"a": "Option A", "b": "Option B"}
        assert q.options == ("a", "b")
        assert q.type == "choice"

    def test_choice_requires_two_options(self):
        with pytest.raises(ValueError, match="at least two options"):
            Choice(instructions="Pick one", criteria={"a": "Only one"})

    def test_choice_no_options(self):
        with pytest.raises(ValueError, match="at least two options"):
            Choice(instructions="Pick one", criteria={})

    def test_choice_criteria_without_descriptions(self):
        q = Choice(instructions="Pick one", criteria={"a": None, "b": None})
        assert q.options == ("a", "b")

    def test_choice_frozen(self):
        q = Choice(instructions="Pick one", criteria={"a": "A", "b": "B"})
        with pytest.raises(AttributeError):
            q.instructions = "Something else"


class TestScore:
    """Test Score ordered scale questions."""

    def test_score_creation(self):
        q = Score(instructions="Rate it", levels=["low", "medium", "high"])
        assert q.instructions == "Rate it"
        assert q.levels == ["low", "medium", "high"]
        assert q.options == ("low", "medium", "high")
        assert q.type == "score"

    def test_score_requires_two_levels(self):
        with pytest.raises(ValueError, match="at least two levels"):
            Score(instructions="Rate it", levels=["only_one"])

    def test_score_no_levels(self):
        with pytest.raises(ValueError, match="at least two levels"):
            Score(instructions="Rate it", levels=[])

    def test_score_frozen(self):
        q = Score(instructions="Rate it", levels=["low", "high"])
        with pytest.raises(AttributeError):
            q.levels = ["a", "b", "c"]


class TestNormalize:
    """Test probability normalization."""

    def test_normalize_already_normalized(self):
        probs = {"a": 0.5, "b": 0.5}
        normalized = normalize(probs)
        assert normalized == {"a": 0.5, "b": 0.5}

    def test_normalize_sums_to_one(self):
        probs = {"a": 1.0, "b": 2.0, "c": 3.0}
        normalized = normalize(probs)
        assert abs(sum(normalized.values()) - 1.0) < 1e-9
        assert normalized == {"a": 1/6, "b": 2/6, "c": 3/6}

    def test_normalize_zero_sum_uniform(self):
        probs = {"a": 0.0, "b": 0.0, "c": 0.0}
        normalized = normalize(probs)
        assert normalized == {"a": 1/3, "b": 1/3, "c": 1/3}

    def test_normalize_negative_values(self):
        probs = {"a": 1.0, "b": -0.5, "c": 0.5}
        normalized = normalize(probs)
        assert abs(sum(normalized.values()) - 1.0) < 1e-9
        assert normalized["b"] == 0.0
        assert normalized["a"] + normalized["c"] == 1.0

    def test_normalize_single_option(self):
        probs = {"only": 5.0}
        normalized = normalize(probs)
        assert normalized == {"only": 1.0}


class TestConfidenceOf:
    """Test confidence computation."""

    def test_confidence_one_hot(self):
        probs = {"a": 1.0, "b": 0.0}
        conf = confidence_of(probs)
        assert conf == 1.0

    def test_confidence_uniform_two_options(self):
        probs = {"a": 0.5, "b": 0.5}
        conf = confidence_of(probs)
        assert abs(conf - 0.0) < 1e-9

    def test_confidence_uniform_three_options(self):
        probs = {"a": 1/3, "b": 1/3, "c": 1/3}
        conf = confidence_of(probs)
        assert abs(conf - 0.0) < 1e-9

    def test_confidence_slight_preference(self):
        probs = {"a": 0.58, "b": 0.42}
        conf = confidence_of(probs)
        # Should be small but positive, less than 0.05 according to docstring
        assert 0.0 < conf < 0.05

    def test_confidence_strong_preference(self):
        probs = {"a": 0.9, "b": 0.1}
        conf = confidence_of(probs)
        assert 0.5 < conf < 1.0

    def test_confidence_single_option(self):
        probs = {"only": 1.0}
        conf = confidence_of(probs)
        assert conf == 1.0

    def test_confidence_zero_mass(self):
        probs = {"a": 0.0, "b": 0.0}
        conf = confidence_of(probs)
        assert conf == 1.0  # Only one valid option


class TestAnswerFromProbs:
    """Test Answer.from_probs for all question types."""

    def test_answer_noul_yes_dominant(self):
        q = Noul(instructions="Is it true?")
        ans = Answer.from_probs(q, {"yes": 0.8, "no": 0.2})
        assert ans.type == "noul"
        assert ans.p == 0.8
        assert ans.confidence == pytest.approx(confidence_of({"yes": 0.8, "no": 0.2}))
        assert ans.choice is None
        assert ans.score is None

    def test_answer_noul_no_dominant(self):
        q = Noul(instructions="Is it true?")
        ans = Answer.from_probs(q, {"yes": 0.2, "no": 0.8})
        assert ans.type == "noul"
        assert ans.p == 0.2
        assert ans.confidence == pytest.approx(confidence_of({"yes": 0.2, "no": 0.8}))

    def test_answer_choice_top_option(self):
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        ans = Answer.from_probs(q, {"a": 0.3, "b": 0.7})
        assert ans.type == "choice"
        assert ans.choice == "b"
        assert ans.confidence == pytest.approx(confidence_of({"a": 0.3, "b": 0.7}))
        assert ans.p is None
        assert ans.score is None

    def test_answer_score_expected_value(self):
        q = Score(instructions="Rate", levels=["low", "medium", "high"])
        ans = Answer.from_probs(q, {"low": 0.2, "medium": 0.5, "high": 0.3})
        assert ans.type == "score"
        assert ans.choice == "medium"
        # Expected value: 0 * 0.2 + 1 * 0.5 + 2 * 0.3 = 1.1
        assert ans.score == pytest.approx(1.1)
        assert ans.legend == {"0": "low", "1": "medium", "2": "high"}
        assert ans.p is None

    def test_answer_score_all_high(self):
        q = Score(instructions="Rate", levels=["low", "high"])
        ans = Answer.from_probs(q, {"low": 0.0, "high": 1.0})
        assert ans.score == pytest.approx(1.0)

    def test_answer_score_all_low(self):
        q = Score(instructions="Rate", levels=["low", "high"])
        ans = Answer.from_probs(q, {"low": 1.0, "high": 0.0})
        assert ans.score == pytest.approx(0.0)

    def test_answer_normalizes_probs(self):
        q = Noul(instructions="Is it true?")
        ans = Answer.from_probs(q, {"yes": 2.0, "no": 1.0})
        # Should normalize to yes=2/3, no=1/3
        assert ans.p == pytest.approx(2/3)

    def test_answer_missing_option(self):
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        ans = Answer.from_probs(q, {"a": 0.7})  # Missing "b"
        assert ans.probabilities == {"a": 1.0, "b": 0.0}


class TestAnswerAsDict:
    """Test Answer.as_dict serialization."""

    def test_as_dict_noul(self):
        q = Noul(instructions="Is it true?")
        ans = Answer.from_probs(q, {"yes": 0.7, "no": 0.3})
        d = ans.as_dict()
        assert d["type"] == "noul"
        assert d["probabilities"] == {"yes": 0.7, "no": 0.3}
        assert d["noul"] == 0.7
        assert "choice" not in d
        assert "score" not in d

    def test_as_dict_choice(self):
        q = Choice(instructions="Pick", criteria={"a": None, "b": None})
        ans = Answer.from_probs(q, {"a": 0.3, "b": 0.7})
        d = ans.as_dict()
        assert d["type"] == "choice"
        assert d["probabilities"] == {"a": 0.3, "b": 0.7}
        assert d["choice"] == "b"
        assert d["confidence"] == pytest.approx(ans.confidence)
        assert "noul" not in d
        assert "score" not in d

    def test_as_dict_score(self):
        q = Score(instructions="Rate", levels=["low", "high"])
        ans = Answer.from_probs(q, {"low": 0.4, "high": 0.6})
        d = ans.as_dict()
        assert d["type"] == "score"
        assert d["probabilities"] == {"low": 0.4, "high": 0.6}
        assert d["choice"] == "high"
        assert d["confidence"] == pytest.approx(ans.confidence)
        assert d["score"] == pytest.approx(0.6)
        assert d["legend"] == {"0": "low", "1": "high"}
        assert "noul" not in d


class TestQuestionFromDict:
    """Test question_from_dict round-trip conversion."""

    def test_noul_from_dict_minimal(self):
        d = {"type": "noul", "instructions": "Is it true?"}
        q = question_from_dict(d)
        assert isinstance(q, Noul)
        assert q.instructions == "Is it true?"
        assert q.true is None
        assert q.false is None

    def test_noul_from_dict_with_criteria(self):
        d = {"type": "noul", "instructions": "Is it true?", "criteria": {"true": "Yes", "false": "No"}}
        q = question_from_dict(d)
        assert isinstance(q, Noul)
        assert q.true == "Yes"
        assert q.false == "No"

    def test_choice_from_dict(self):
        d = {"type": "choice", "instructions": "Pick", "criteria": {"a": "Option A", "b": "Option B"}}
        q = question_from_dict(d)
        assert isinstance(q, Choice)
        assert q.instructions == "Pick"
        assert q.criteria == {"a": "Option A", "b": "Option B"}

    def test_score_from_dict_list_strings(self):
        d = {"type": "score", "instructions": "Rate", "criteria": ["low", "medium", "high"]}
        q = question_from_dict(d)
        assert isinstance(q, Score)
        assert q.instructions == "Rate"
        assert list(q.levels) == ["low", "medium", "high"]

    def test_score_from_dict_list_dicts_with_name(self):
        d = {
            "type": "score",
            "instructions": "Rate",
            "criteria": [
                {"name": "low", "description": "Not good"},
                {"name": "high", "description": "Great"}
            ]
        }
        q = question_from_dict(d)
        assert isinstance(q, Score)
        assert list(q.levels) == ["low", "high"]

    def test_score_from_dict_list_dicts_with_level(self):
        d = {
            "type": "score",
            "instructions": "Rate",
            "criteria": [
                {"level": "low"},
                {"level": "high"}
            ]
        }
        q = question_from_dict(d)
        assert isinstance(q, Score)
        assert list(q.levels) == ["low", "high"]

    def test_unknown_type_raises(self):
        d = {"type": "unknown", "instructions": "What?"}
        with pytest.raises(ValueError, match="unknown question type"):
            question_from_dict(d)


class TestQuestionToDict:
    """Test question_to_dict conversion."""

    def test_noul_to_dict_minimal(self):
        q = Noul(instructions="Is it true?")
        d = question_to_dict(q)
        assert d["type"] == "noul"
        assert d["instructions"] == "Is it true?"
        assert "criteria" not in d

    def test_noul_to_dict_with_criteria(self):
        q = Noul(instructions="Is it true?", true="Yes", false="No")
        d = question_to_dict(q)
        assert d["type"] == "noul"
        assert d["instructions"] == "Is it true?"
        assert d["criteria"] == {"true": "Yes", "false": "No"}

    def test_noul_to_dict_partial_criteria(self):
        q = Noul(instructions="Is it true?", true="Yes", false=None)
        d = question_to_dict(q)
        assert d["type"] == "noul"
        assert d["criteria"] == {"true": "Yes", "false": None}

    def test_choice_to_dict(self):
        q = Choice(instructions="Pick", criteria={"a": "Option A", "b": None})
        d = question_to_dict(q)
        assert d["type"] == "choice"
        assert d["instructions"] == "Pick"
        assert d["criteria"] == {"a": "Option A", "b": None}

    def test_score_to_dict(self):
        q = Score(instructions="Rate", levels=["low", "high"])
        d = question_to_dict(q)
        assert d["type"] == "score"
        assert d["instructions"] == "Rate"
        assert d["criteria"] == ["low", "high"]


class TestRoundTrip:
    """Test round-trip conversions."""

    def test_noul_round_trip(self):
        q1 = Noul(instructions="Is it true?", true="Yes", false="No")
        d = question_to_dict(q1)
        q2 = question_from_dict(d)
        assert q1 == q2

    def test_choice_round_trip(self):
        q1 = Choice(instructions="Pick", criteria={"a": "A", "b": "B"})
        d = question_to_dict(q1)
        q2 = question_from_dict(d)
        assert q1 == q2

    def test_score_round_trip(self):
        q1 = Score(instructions="Rate", levels=["low", "medium", "high"])
        d = question_to_dict(q1)
        q2 = question_from_dict(d)
        assert q1 == q2
