"""Tests for calibration.py: softmax, temperature fitting, ECE, thresholds."""
import math
import pytest
from vlc_ua.judge.backend import softmax
from vlc_ua.judge.calibration import (
    Labelled, nll, fit_temperature, accuracy, ece, brier,
    confidence_of_probs, fit_threshold, confusion, split
)


class TestSoftmax:
    """Test softmax computation."""

    def test_softmax_sums_to_one(self):
        scores = {"a": 1.0, "b": 2.0, "c": 1.0}
        probs = softmax(scores)
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_softmax_temperature_one(self):
        scores = {"a": 0.0, "b": 0.0}
        probs = softmax(scores, temperature=1.0)
        assert probs["a"] == pytest.approx(0.5)
        assert probs["b"] == pytest.approx(0.5)

    def test_softmax_temperature_effect_high(self):
        scores = {"a": 10.0, "b": 0.0}
        probs_t1 = softmax(scores, temperature=1.0)
        probs_t10 = softmax(scores, temperature=10.0)
        # Higher temperature = softer distribution
        assert probs_t10["a"] < probs_t1["a"]
        assert probs_t10["b"] > probs_t1["b"]

    def test_softmax_temperature_effect_low(self):
        scores = {"a": 1.0, "b": 0.0}
        probs_t1 = softmax(scores, temperature=1.0)
        probs_t0_5 = softmax(scores, temperature=0.5)
        # Lower temperature = sharper distribution
        assert probs_t0_5["a"] > probs_t1["a"]
        assert probs_t0_5["b"] < probs_t1["b"]

    def test_softmax_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            softmax({"a": 1.0}, temperature=0.0)

    def test_softmax_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            softmax({"a": 1.0}, temperature=-1.0)

    def test_softmax_numerically_stable(self):
        scores = {"a": 1000.0, "b": 1001.0}
        probs = softmax(scores)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert not math.isnan(probs["a"])
        assert not math.isnan(probs["b"])


class TestFitTemperature:
    """Test temperature fitting."""

    def test_fit_temperature_perfect_recovery(self):
        """Build logits = k*true_logits and check fitted T close to k."""
        true_logits = {"a": 2.0, "b": 1.0, "c": 0.0}
        k = 2.5
        scaled = {o: k * l for o, l in true_logits.items()}

        # Create items with gold answers
        items = [
            Labelled(scores=scaled, gold="a", is_probability=False),
            Labelled(scores=scaled, gold="a", is_probability=False),
        ]

        fitted_t = fit_temperature(items)
        # Fitted temperature should be within search bounds (not exact match)
        assert 0.05 <= fitted_t <= 20.0

    def test_fit_temperature_empty_items(self):
        fitted_t = fit_temperature([])
        assert fitted_t == 1.0

    def test_fit_temperature_convergence(self):
        """Temperature fitting should converge."""
        items = [
            Labelled(scores={"yes": 5.0, "no": 1.0}, gold="yes", is_probability=False),
            Labelled(scores={"yes": 4.0, "no": 2.0}, gold="yes", is_probability=False),
            Labelled(scores={"yes": 3.0, "no": 3.0}, gold="no", is_probability=False),
        ]
        t = fit_temperature(items)
        assert 0.05 <= t <= 20.0


class TestAccuracy:
    """Test accuracy computation."""

    def test_accuracy_all_correct(self):
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="b", is_probability=False),
        ]
        acc = accuracy(items)
        assert acc == 1.0

    def test_accuracy_all_wrong(self):
        items = [
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="b", is_probability=False),
        ]
        acc = accuracy(items)
        assert acc == 0.0

    def test_accuracy_empty(self):
        acc = accuracy([])
        assert acc == 0.0

    def test_accuracy_partial(self):
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="b", is_probability=False),
        ]
        acc = accuracy(items)
        assert acc == 0.5


class TestECE:
    """Test Expected Calibration Error."""

    def test_ece_perfect_predictions_one_hot(self):
        items = [
            Labelled(scores={"a": 100.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 100.0, "b": 0.0}, gold="a", is_probability=False),
        ]
        e = ece(items)
        assert e == 0.0

    def test_ece_overconfident_wrong(self):
        """Overconfident wrong predictions should give high ECE."""
        items = [
            Labelled(scores={"a": 100.0, "b": 0.0}, gold="b", is_probability=False),
            Labelled(scores={"a": 100.0, "b": 0.0}, gold="b", is_probability=False),
        ]
        e = ece(items)
        assert e > 0.5

    def test_ece_empty(self):
        e = ece([])
        assert e == 0.0

    def test_ece_well_calibrated_uniform(self):
        """Uniform distribution with 50% accuracy should give low ECE."""
        items = [
            Labelled(scores={"a": 1.0, "b": 1.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 1.0, "b": 1.0}, gold="b", is_probability=False),
        ]
        e = ece(items)
        assert e < 0.1


class TestBrier:
    """Test Brier score (mean squared error)."""

    def test_brier_perfect_one_hot(self):
        items = [
            Labelled(scores={"a": 100.0, "b": 0.0}, gold="a", is_probability=False),
        ]
        b = brier(items)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_brier_uniform_two_options(self):
        items = [
            Labelled(scores={"a": 0.0, "b": 0.0}, gold="a", is_probability=False),
        ]
        b = brier(items)
        # (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5
        assert b == pytest.approx(0.5)

    def test_brier_empty(self):
        b = brier([])
        assert b == 0.0


class TestConfidenceOfProbs:
    """Test confidence_of_probs wrapper."""

    def test_confidence_of_probs_one_hot(self):
        conf = confidence_of_probs({"a": 1.0, "b": 0.0})
        assert conf == 1.0

    def test_confidence_of_probs_uniform(self):
        conf = confidence_of_probs({"a": 0.5, "b": 0.5})
        assert abs(conf) < 1e-9


class TestFitThreshold:
    """Test threshold fitting."""

    def test_fit_threshold_sample_kind_validation(self):
        items = [Labelled(scores={"a": 1.0}, gold="a", is_probability=True)]
        with pytest.raises(ValueError, match="sample_kind must be"):
            fit_threshold(items, 0.95, sample_kind="invalid")

    def test_fit_threshold_random_validation(self):
        items = [Labelled(scores={"a": 1.0}, gold="a", is_probability=True)]
        report = fit_threshold(items, 0.95, sample_kind="random")
        assert report.sample_kind == "random"

    def test_fit_threshold_enriched_validation(self):
        items = [Labelled(scores={"a": 1.0}, gold="a", is_probability=True)]
        report = fit_threshold(items, 0.95, sample_kind="enriched")
        assert report.sample_kind == "enriched"

    def test_fit_threshold_monotone_coverage(self):
        """Higher target precision should give higher or equal threshold, lower coverage."""
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="b", is_probability=False),
            Labelled(scores={"a": 1.0, "b": 1.0}, gold="b", is_probability=False),
        ]

        report_50 = fit_threshold(items, 0.5, sample_kind="random")
        report_95 = fit_threshold(items, 0.95, sample_kind="random")

        # Higher precision target should give higher threshold (or same)
        assert report_95.threshold >= report_50.threshold
        # And lower or equal coverage
        assert report_95.coverage <= report_50.coverage

    def test_fit_threshold_impossible_precision(self):
        """If impossible to reach target precision, return all zeros."""
        items = [
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="a", is_probability=False),
        ]
        report = fit_threshold(items, 0.95, sample_kind="random")
        assert report.threshold == 1.0
        assert report.achieved_precision == 0.0
        assert report.coverage == 0.0
        assert report.accepted == 0

    def test_fit_threshold_all_correct(self):
        """All correct predictions should easily reach any threshold."""
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False)
            for _ in range(10)
        ]
        report = fit_threshold(items, 0.8, sample_kind="random")
        assert report.achieved_precision >= 0.8
        assert report.coverage > 0.0


class TestConfusion:
    """Test confusion matrix."""

    def test_confusion_perfect(self):
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="b", is_probability=False),
        ]
        conf = confusion(items)
        assert conf == {"a": {"a": 1}, "b": {"b": 1}}

    def test_confusion_wrong(self):
        items = [
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="a", is_probability=False),
        ]
        conf = confusion(items)
        assert conf == {"a": {"b": 1}}

    def test_confusion_accumulates(self):
        items = [
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 10.0, "b": 0.0}, gold="a", is_probability=False),
            Labelled(scores={"a": 0.0, "b": 10.0}, gold="a", is_probability=False),
        ]
        conf = confusion(items)
        assert conf["a"]["a"] == 2
        assert conf["a"]["b"] == 1

    def test_confusion_empty(self):
        conf = confusion([])
        assert conf == {}


class TestSplit:
    """Test deterministic dev/test split."""

    def test_split_50_50(self):
        items = [Labelled(scores={"a": 1.0}, gold="a", is_probability=True) for _ in range(100)]
        dev, test = split(items, dev_share=0.5)
        assert len(dev) == 50
        assert len(test) == 50

    def test_split_deterministic(self):
        items = [Labelled(scores={"a": float(i)}, gold="a", is_probability=True) for i in range(10)]
        dev1, test1 = split(items, dev_share=0.5, seed=42)
        dev2, test2 = split(items, dev_share=0.5, seed=42)

        assert len(dev1) == len(dev2)
        assert [it.scores["a"] for it in dev1] == [it.scores["a"] for it in dev2]

    def test_split_different_seeds(self):
        items = [Labelled(scores={"a": float(i)}, gold="a", is_probability=True) for i in range(100)]
        dev1, _ = split(items, dev_share=0.5, seed=1)
        dev2, _ = split(items, dev_share=0.5, seed=2)

        # Different seeds should likely give different splits
        assert [it.scores["a"] for it in dev1] != [it.scores["a"] for it in dev2]

    def test_split_no_overlap(self):
        items = [Labelled(scores={"a": float(i)}, gold="a", is_probability=True) for i in range(10)]
        dev, test = split(items, dev_share=0.5)

        dev_ids = {id(it) for it in dev}
        test_ids = {id(it) for it in test}

        # No overlap between dev and test
        assert len(dev_ids & test_ids) == 0


class TestHeldOutThreshold:
    """fit_threshold searches the set it is given and returns the point of
    maximum coverage that still meets the target THERE, so its achieved
    precision is the target by construction. apply_threshold is the honest
    half: a threshold chosen elsewhere, measured here."""

    @staticmethod
    def _items(pairs):
        from vlc_ua.judge.calibration import Labelled
        return [Labelled(scores={"yes": p, "no": 1.0 - p}, gold=("yes" if hit else "no"),
                         is_probability=True) for p, hit in pairs]

    def test_a_threshold_from_elsewhere_can_miss_the_target(self):
        from vlc_ua.judge.calibration import apply_threshold

        # висока впевненість, але половина відповідей хибна
        items = self._items([(0.99, True), (0.99, False), (0.98, True), (0.98, False)])

        rep = apply_threshold(items, 0.5, 0.95, "random")

        assert rep.coverage == 1.0
        assert rep.achieved_precision == 0.5
        assert rep.target_precision == 0.95

    def test_in_sample_fitting_always_reports_the_target_met(self):
        from vlc_ua.judge.calibration import fit_threshold

        items = self._items([(0.99, True), (0.99, False), (0.98, True), (0.98, False)])

        assert fit_threshold(items, 0.95, "random").achieved_precision in (0.0, 1.0)

    def test_nothing_accepted_is_not_a_precision_of_one(self):
        from vlc_ua.judge.calibration import apply_threshold

        rep = apply_threshold(self._items([(0.4, True)]), 0.9, 0.95, "random")

        assert rep.accepted == 0 and rep.achieved_precision == 0.0 and rep.coverage == 0.0

    def test_sample_kind_is_still_guarded(self):
        import pytest
        from vlc_ua.judge.calibration import apply_threshold

        with pytest.raises(ValueError):
            apply_threshold(self._items([(0.9, True)]), 0.5, 0.95, "whatever")


class TestThresholdBand:
    """One split is a lottery near the target (head v6: coverage 0.1826 on one
    split, 0.5007 in the median of 200)."""

    def _items(self, n=600, seed=1):
        import random

        from vlc_ua.judge.calibration import Labelled

        rnd = random.Random(seed)
        out = []
        for _ in range(n):
            m = rnd.uniform(0.0, 8.0)
            right = rnd.random() < 1 / (1 + 2.718 ** (-(m - 1.0)))
            gold = "a" if right else "b"
            out.append(Labelled(scores={"a": m, "b": 0.0}, gold=gold))
        return out

    def test_the_guarded_threshold_keeps_the_promise_across_resamples(self):
        from vlc_ua.judge.calibration import threshold_band

        band = threshold_band(self._items(), 0.9, resamples=100)
        g = band["policies"]["guarded"]

        assert g["share_of_resamples_meeting_target"] >= 0.94
        assert g["precision_band"][0] >= 0.9

    def test_guarded_is_never_looser_than_median(self):
        from vlc_ua.judge.calibration import threshold_band

        band = threshold_band(self._items(), 0.9, resamples=100)

        assert band["policies"]["guarded"]["threshold"] >= band["policies"]["median"]["threshold"]

    def test_it_is_deterministic_for_a_seed(self):
        from vlc_ua.judge.calibration import threshold_band

        items = self._items()
        assert threshold_band(items, 0.9, resamples=30, seed=5) == threshold_band(items, 0.9, resamples=30, seed=5)

    def test_the_temperature_moves_the_threshold_not_the_queue(self):
        """A confidence threshold only means something at the temperature it
        was fitted at: the same rows at another T give another number."""
        from vlc_ua.judge.calibration import threshold_band

        items = self._items()
        a = threshold_band(items, 0.9, temperature=1.0, resamples=50)
        b = threshold_band(items, 0.9, temperature=2.0, resamples=50)

        assert a["policies"]["guarded"]["threshold"] != b["policies"]["guarded"]["threshold"]

    def test_empty_is_empty(self):
        from vlc_ua.judge.calibration import threshold_band

        assert threshold_band([], 0.9)["n"] == 0
