"""Scene-change detection and timestamped event construction."""

from __future__ import annotations

import numpy as np

from surgivision.types import FrameAnalysis, SceneBoundary


def detect_scene_boundaries(
    frames: list[FrameAnalysis],
    embeddings: np.ndarray | None,
    distance_threshold: float,
) -> list[SceneBoundary]:
    """Mark adjacent sampled frames whose cosine distance exceeds a threshold."""

    if distance_threshold < 0:
        raise ValueError("Scene distance threshold cannot be negative")
    if embeddings is None or len(frames) < 2:
        return []
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(frames):
        raise ValueError("Embedding rows must align with analyzed frames")

    boundaries: list[SceneBoundary] = []
    for index in range(1, len(frames)):
        similarity = float(np.clip(np.dot(values[index - 1], values[index]), -1.0, 1.0))
        distance = 1.0 - similarity
        if distance >= distance_threshold:
            boundaries.append(
                SceneBoundary(
                    timestamp_s=frames[index].sample.timestamp_s,
                    distance=distance,
                    previous_frame_index=frames[index - 1].sample.frame_index,
                    current_frame_index=frames[index].sample.frame_index,
                )
            )
    return boundaries


def build_timeline(
    frames: list[FrameAnalysis],
    scene_boundaries: list[SceneBoundary],
    instrument_like_labels: tuple[str, ...] = ("knife", "scissors"),
) -> list[dict[str, object]]:
    """Build a stable, timestamp-sorted list of observed analysis events."""

    events: list[dict[str, object]] = []
    instrument_names = {label.casefold() for label in instrument_like_labels}
    for frame in frames:
        timestamp = round(frame.sample.timestamp_s, 4)
        events.append(
            {
                "timestamp_s": timestamp,
                "event_type": "sampled_frame",
                "label": f"Frame {frame.sample.frame_index}",
                "value": frame.sample.frame_index,
            }
        )
        if frame.quality.is_low_quality:
            events.append(
                {
                    "timestamp_s": timestamp,
                    "event_type": "low_quality",
                    "label": "Blur threshold exceeded",
                    "value": round(frame.quality.blur_score, 4),
                }
            )
        for detection in frame.detections:
            detection_type = (
                "instrument_like_detection"
                if detection.class_name.casefold() in instrument_names
                else "object_detection"
            )
            events.append(
                {
                    "timestamp_s": timestamp,
                    "event_type": detection_type,
                    "label": detection.class_name,
                    "value": round(detection.confidence, 4),
                }
            )
    for boundary in scene_boundaries:
        events.append(
            {
                "timestamp_s": round(boundary.timestamp_s, 4),
                "event_type": "scene_boundary",
                "label": "Visual scene change",
                "value": round(boundary.distance, 4),
            }
        )
    return sorted(events, key=lambda event: (float(event["timestamp_s"]), str(event["event_type"])))
