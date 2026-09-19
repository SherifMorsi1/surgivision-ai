"""Correctness tests for phase-recognition metrics with hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from surgivision.evaluation.metrics import (
    evaluate_video,
    relax_predictions,
    segment_boundaries,
)
from surgivision.evaluation.protocols import (
    RELAXED,
    STRICT,
    evaluate_all_protocols,
    evaluate_benchmark,
)


def test_segment_boundaries_finds_runs():
    labels = np.array([0, 0, 0, 1, 1, 2])
    assert segment_boundaries(labels) == [(0, 3, 0), (3, 5, 1), (5, 6, 2)]


def test_segment_boundaries_handles_single_run():
    assert segment_boundaries(np.array([3, 3, 3])) == [(0, 3, 3)]


def test_perfect_prediction_scores_one():
    targets = [0, 0, 1, 1, 2, 2]
    result = evaluate_video(targets, targets, num_phases=3)
    assert result.accuracy == pytest.approx(1.0)
    assert result.mean_precision == pytest.approx(1.0)
    assert result.mean_recall == pytest.approx(1.0)
    assert result.mean_jaccard == pytest.approx(1.0)


def test_per_phase_scores_match_hand_computation():
    # targets     [0, 0, 1, 1]
    # predictions [0, 1, 1, 1]
    # phase 0 -> TP=1 FP=0 FN=1 : P=1.000 R=0.500 J=0.500
    # phase 1 -> TP=2 FP=1 FN=0 : P=0.667 R=1.000 J=0.667
    result = evaluate_video([0, 0, 1, 1], [0, 1, 1, 1], num_phases=2)
    assert result.accuracy == pytest.approx(0.75)

    phase_zero, phase_one = result.per_phase
    assert phase_zero.precision == pytest.approx(1.0)
    assert phase_zero.recall == pytest.approx(0.5)
    assert phase_zero.f1 == pytest.approx(2 / 3)
    assert phase_zero.jaccard == pytest.approx(0.5)
    assert phase_zero.support == 2

    assert phase_one.precision == pytest.approx(2 / 3)
    assert phase_one.recall == pytest.approx(1.0)
    assert phase_one.f1 == pytest.approx(0.8)
    assert phase_one.jaccard == pytest.approx(2 / 3)

    assert result.mean_precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.mean_jaccard == pytest.approx((0.5 + 2 / 3) / 2)


def test_relaxation_forgives_adjacent_phase_at_a_boundary():
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Late by one frame on each side of the transition.
    predictions = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    relaxed = relax_predictions(targets, predictions, tolerance_frames=1)
    assert np.array_equal(relaxed, targets)

    assert evaluate_video(targets, predictions, 2, tolerance_frames=0).accuracy == pytest.approx(0.8)
    assert evaluate_video(targets, predictions, 2, tolerance_frames=1).accuracy == pytest.approx(1.0)


def test_relaxation_does_not_forgive_a_non_adjacent_phase():
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    # Frame 4 sits at a boundary, but phase 2 is not the neighbouring phase.
    predictions = targets.copy()
    predictions[4] = 2

    relaxed = relax_predictions(targets, predictions, tolerance_frames=2)
    assert relaxed[4] == 2, "a non-adjacent phase must not be forgiven"


def test_relaxation_does_not_forgive_errors_away_from_a_boundary():
    targets = np.array([0] * 10 + [1] * 10)
    predictions = targets.copy()
    predictions[5] = 1  # mid-segment, five frames from the transition

    relaxed = relax_predictions(targets, predictions, tolerance_frames=2)
    assert relaxed[5] == 1, "an error in the segment interior must survive relaxation"


def test_zero_tolerance_is_identity():
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([1, 0, 0, 1])
    assert np.array_equal(relax_predictions(targets, predictions, 0), predictions)


def test_absent_phases_are_excluded_from_averages():
    # Phase 2 never occurs and is never predicted; scoring it as zero would
    # unfairly drag the mean down.
    result = evaluate_video([0, 0, 1, 1], [0, 0, 1, 1], num_phases=3)
    assert [phase.phase for phase in result.per_phase] == [0, 1]
    assert result.mean_precision == pytest.approx(1.0)


def test_benchmark_averages_over_videos_not_pooled_frames():
    # A short perfect video and a long poor one. Averaging over videos gives
    # (1.00 + 0.50) / 2 = 0.75; pooling frames gives 12/20 = 0.60, which lets
    # the long video dominate.
    targets = {"short": [0] * 4, "long": [0] * 16}
    predictions = {"short": [0] * 4, "long": [0] * 8 + [1] * 8}

    result = evaluate_benchmark(targets, predictions, num_phases=2, protocol=STRICT)
    assert result.accuracy == pytest.approx(0.75)
    assert result.num_videos == 2

    pooled = np.mean(np.array([0] * 4 + [0] * 8 + [1] * 8) == 0)
    assert pooled == pytest.approx(0.60)
    assert result.accuracy != pytest.approx(pooled)


def test_benchmark_reports_dispersion_across_videos():
    targets = {"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}
    predictions = {"a": [0, 0, 0, 0], "b": [1, 1, 0, 0]}
    result = evaluate_benchmark(targets, predictions, num_phases=2, protocol=STRICT)
    assert result.accuracy == pytest.approx(0.75)
    assert result.accuracy_std == pytest.approx(0.25)


def test_tolerance_frames_scales_with_fps():
    assert RELAXED.tolerance_frames(1.0) == 10
    assert RELAXED.tolerance_frames(25.0) == 250
    assert STRICT.tolerance_frames(25.0) == 0


def test_all_protocols_reports_both_and_relaxed_is_never_worse():
    targets = {"v1": [0] * 5 + [1] * 5}
    predictions = {"v1": [0] * 4 + [1] * 6}
    results = evaluate_all_protocols(targets, predictions, num_phases=2, fps=1.0)
    assert set(results) == {"strict", "relaxed"}
    assert results["relaxed"].accuracy >= results["strict"].accuracy


def test_missing_predictions_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        evaluate_benchmark({"a": [0], "b": [0]}, {"a": [0]}, num_phases=1)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        evaluate_video([0, 0, 1], [0, 1], num_phases=2)


def test_out_of_range_label_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        evaluate_video([0, 1], [0, 5], num_phases=2)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate_video([], [], num_phases=2)
