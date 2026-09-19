"""Frame-level and video-level metrics for surgical phase recognition.

The metrics here follow the evaluation conventions established for Cholec80 by
Twinanda et al. and are reported per video before being averaged across videos.
That order matters: averaging over pooled frames instead lets long videos
dominate the score and inflates results by several points, which is one of the
reasons published numbers are difficult to compare.

Two protocols are provided. The strict protocol scores every frame exactly. The
relaxed protocol forgives disagreement inside a tolerance window around each
ground-truth phase transition, because the precise moment a phase changes is
not consistently identifiable even between expert annotators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_TOLERANCE_S = 10.0


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Per-phase scores for a single phase index."""

    phase: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "jaccard": round(self.jaccard, 4),
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class VideoMetrics:
    """Scores for one video, plus the per-phase breakdown behind them."""

    video_id: str
    accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_jaccard: float
    per_phase: tuple[PhaseMetrics, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "accuracy": round(self.accuracy, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_jaccard": round(self.mean_jaccard, 4),
            "per_phase": [phase.to_dict() for phase in self.per_phase],
        }


def _as_label_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError(f"{name} must contain integer phase labels")
        array = array.astype(np.int64)
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative phase labels")
    return array.astype(np.int64)


def segment_boundaries(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(start, stop, label)`` runs, where ``stop`` is exclusive."""

    if labels.size == 0:
        return []
    change_points = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change_points))
    stops = np.concatenate((change_points, [labels.size]))
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, stops, strict=True)]


def relax_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    tolerance_frames: int,
) -> np.ndarray:
    """Forgive predictions of an adjacent phase near a ground-truth transition.

    Within ``tolerance_frames`` of the start of a ground-truth segment a
    prediction naming the *previous* phase is accepted, and within the same
    distance of the end a prediction naming the *next* phase is accepted. Such
    frames are rewritten to the ground-truth label so that downstream counting
    is protocol-agnostic. Frames predicted as any other phase are left alone,
    and a corrected frame is never propagated into a neighbouring window.
    """

    if tolerance_frames < 0:
        raise ValueError("Tolerance cannot be negative")
    relaxed = predictions.copy()
    if tolerance_frames == 0:
        return relaxed

    segments = segment_boundaries(targets)
    for position, (start, stop, label) in enumerate(segments):
        window = min(tolerance_frames, stop - start)
        if position > 0:
            previous_label = segments[position - 1][2]
            head = slice(start, start + window)
            relaxed[head] = np.where(predictions[head] == previous_label, label, relaxed[head])
        if position + 1 < len(segments):
            next_label = segments[position + 1][2]
            tail = slice(stop - window, stop)
            relaxed[tail] = np.where(predictions[tail] == next_label, label, relaxed[tail])
    return relaxed


def evaluate_video(
    targets: Any,
    predictions: Any,
    num_phases: int,
    video_id: str = "video",
    tolerance_frames: int = 0,
) -> VideoMetrics:
    """Score one video, optionally under the relaxed boundary protocol.

    Phases absent from both the ground truth and the prediction are excluded
    from the averages rather than scored as zero, which otherwise penalizes a
    model for a phase the video never contained.
    """

    target_array = _as_label_array(targets, "targets")
    prediction_array = _as_label_array(predictions, "predictions")
    if target_array.shape != prediction_array.shape:
        raise ValueError("Targets and predictions must have the same length")
    if num_phases <= 0:
        raise ValueError("num_phases must be greater than zero")
    for name, array in (("targets", target_array), ("predictions", prediction_array)):
        if np.any(array >= num_phases):
            raise ValueError(f"{name} contains a label outside [0, {num_phases - 1}]")

    scored = relax_predictions(target_array, prediction_array, tolerance_frames)
    accuracy = float(np.mean(scored == target_array))

    per_phase: list[PhaseMetrics] = []
    for phase in range(num_phases):
        target_mask = target_array == phase
        predicted_mask = scored == phase
        support = int(target_mask.sum())
        if support == 0 and not predicted_mask.any():
            continue
        true_positive = float(np.sum(target_mask & predicted_mask))
        false_positive = float(np.sum(~target_mask & predicted_mask))
        false_negative = float(np.sum(target_mask & ~predicted_mask))
        precision = true_positive / (true_positive + false_positive) if predicted_mask.any() else 0.0
        recall = true_positive / (true_positive + false_negative) if support else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        union = true_positive + false_positive + false_negative
        jaccard = true_positive / union if union else 0.0
        per_phase.append(
            PhaseMetrics(
                phase=phase,
                precision=precision,
                recall=recall,
                f1=f1,
                jaccard=jaccard,
                support=support,
            )
        )

    def _mean(attribute: str) -> float:
        if not per_phase:
            return 0.0
        return float(np.mean([getattr(item, attribute) for item in per_phase]))

    return VideoMetrics(
        video_id=video_id,
        accuracy=accuracy,
        mean_precision=_mean("precision"),
        mean_recall=_mean("recall"),
        mean_f1=_mean("f1"),
        mean_jaccard=_mean("jaccard"),
        per_phase=tuple(per_phase),
    )
