"""Benchmark-level evaluation protocols and cross-video aggregation.

A protocol fixes the two choices that make published surgical-phase numbers
hard to compare: the boundary tolerance, and whether scores are averaged over
videos or over pooled frames. Both are recorded in the result so a reported
number always carries the protocol that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from surgivision.evaluation.metrics import (
    DEFAULT_TOLERANCE_S,
    VideoMetrics,
    evaluate_video,
)


@dataclass(frozen=True, slots=True)
class Protocol:
    """An evaluation protocol, named so results remain self-describing."""

    name: str
    tolerance_s: float
    description: str

    def tolerance_frames(self, fps: float) -> int:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        return int(round(self.tolerance_s * fps))


STRICT = Protocol(
    name="strict",
    tolerance_s=0.0,
    description="Every frame is scored exactly, with no boundary tolerance.",
)

RELAXED = Protocol(
    name="relaxed",
    tolerance_s=DEFAULT_TOLERANCE_S,
    description=(
        "Predictions of an adjacent phase within 10 seconds of a ground-truth "
        "transition are accepted, following the Cholec80 convention."
    ),
)

PROTOCOLS: dict[str, Protocol] = {STRICT.name: STRICT, RELAXED.name: RELAXED}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregated scores across a set of videos under one protocol."""

    protocol: Protocol
    fps: float
    accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_jaccard: float
    accuracy_std: float
    per_video: tuple[VideoMetrics, ...]

    @property
    def num_videos(self) -> int:
        return len(self.per_video)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.name,
            "tolerance_s": self.protocol.tolerance_s,
            "fps": self.fps,
            "num_videos": self.num_videos,
            "accuracy": round(self.accuracy, 4),
            "accuracy_std": round(self.accuracy_std, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_jaccard": round(self.mean_jaccard, 4),
            "per_video": [video.to_dict() for video in self.per_video],
        }

    def summary_line(self) -> str:
        return (
            f"{self.protocol.name:<8} "
            f"acc {self.accuracy * 100:6.2f} ±{self.accuracy_std * 100:5.2f}  "
            f"prec {self.mean_precision * 100:6.2f}  "
            f"rec {self.mean_recall * 100:6.2f}  "
            f"jacc {self.mean_jaccard * 100:6.2f}  "
            f"(n={self.num_videos})"
        )


def evaluate_benchmark(
    targets_by_video: Mapping[str, Sequence[int]],
    predictions_by_video: Mapping[str, Sequence[int]],
    num_phases: int,
    protocol: Protocol = RELAXED,
    fps: float = 1.0,
) -> BenchmarkResult:
    """Score every video, then average across videos rather than pooled frames."""

    if not targets_by_video:
        raise ValueError("No ground-truth videos were supplied")
    missing = set(targets_by_video) - set(predictions_by_video)
    if missing:
        raise ValueError(f"Predictions are missing for: {sorted(missing)}")

    tolerance = protocol.tolerance_frames(fps)
    per_video = tuple(
        evaluate_video(
            targets_by_video[video_id],
            predictions_by_video[video_id],
            num_phases=num_phases,
            video_id=video_id,
            tolerance_frames=tolerance,
        )
        for video_id in sorted(targets_by_video)
    )

    accuracies = np.array([video.accuracy for video in per_video], dtype=np.float64)
    return BenchmarkResult(
        protocol=protocol,
        fps=fps,
        accuracy=float(accuracies.mean()),
        accuracy_std=float(accuracies.std(ddof=0)),
        mean_precision=float(np.mean([v.mean_precision for v in per_video])),
        mean_recall=float(np.mean([v.mean_recall for v in per_video])),
        mean_f1=float(np.mean([v.mean_f1 for v in per_video])),
        mean_jaccard=float(np.mean([v.mean_jaccard for v in per_video])),
        per_video=per_video,
    )


def evaluate_all_protocols(
    targets_by_video: Mapping[str, Sequence[int]],
    predictions_by_video: Mapping[str, Sequence[int]],
    num_phases: int,
    fps: float = 1.0,
) -> dict[str, BenchmarkResult]:
    """Score under every registered protocol so both numbers are always reported."""

    return {
        name: evaluate_benchmark(
            targets_by_video,
            predictions_by_video,
            num_phases=num_phases,
            protocol=protocol,
            fps=fps,
        )
        for name, protocol in PROTOCOLS.items()
    }
