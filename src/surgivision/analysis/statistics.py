"""Aggregate statistics derived only from observed pipeline outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from surgivision.types import FrameAnalysis, SceneBoundary, VideoMetadata


def calculate_statistics(
    metadata: VideoMetadata,
    frames: list[FrameAnalysis],
    scene_boundaries: list[SceneBoundary],
    runtime_s: float,
    model_device: str,
    instrument_like_labels: tuple[str, ...] = ("knife", "scissors"),
) -> dict[str, Any]:
    detections = [detection for frame in frames for detection in frame.detections]
    counts = Counter(detection.class_name for detection in detections)
    instrument_names = {label.casefold() for label in instrument_like_labels}
    instrument_like_count = sum(
        count for label, count in counts.items() if label.casefold() in instrument_names
    )
    low_quality_count = sum(frame.quality.is_low_quality for frame in frames)
    sampled_count = len(frames)
    average_confidence = (
        sum(detection.confidence for detection in detections) / len(detections)
        if detections
        else 0.0
    )
    return {
        "video_duration_s": round(metadata.duration_s, 4),
        "sampled_frames": sampled_count,
        "sampling_coverage_percent": round(
            sampled_count / metadata.frame_count * 100 if metadata.frame_count else 0.0, 4
        ),
        "total_detections": len(detections),
        "instrument_like_detections": instrument_like_count,
        "detections_by_class": dict(sorted(counts.items())),
        "average_detection_confidence": round(average_confidence, 4),
        "low_quality_frames": low_quality_count,
        "low_quality_percent": round(
            low_quality_count / sampled_count * 100 if sampled_count else 0.0, 4
        ),
        "scene_boundaries": len(scene_boundaries),
        "runtime_s": round(runtime_s, 4),
        "model_device": model_device,
    }
