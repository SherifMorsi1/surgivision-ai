"""Typed data structures used throughout the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    filename: str
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    codec: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 4),
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
        }


@dataclass(slots=True)
class SampledFrame:
    frame_index: int
    timestamp_s: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class QualityMeasurement:
    timestamp_s: float
    blur_score: float
    is_low_quality: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "blur_score": round(self.blur_score, 4),
            "is_low_quality": self.is_low_quality,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    timestamp_s: float
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 2) for value in self.bbox],
        }


@dataclass(frozen=True, slots=True)
class SceneBoundary:
    timestamp_s: float
    distance: float
    previous_frame_index: int
    current_frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "distance": round(self.distance, 4),
            "previous_frame_index": self.previous_frame_index,
            "current_frame_index": self.current_frame_index,
        }


@dataclass(slots=True)
class FrameAnalysis:
    sample: SampledFrame
    quality: QualityMeasurement
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.sample.frame_index,
            "timestamp_s": round(self.sample.timestamp_s, 4),
            "quality": self.quality.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
        }


@dataclass(slots=True)
class AnalysisResult:
    analysis_id: str
    metadata: VideoMetadata
    frames: list[FrameAnalysis]
    scene_boundaries: list[SceneBoundary]
    statistics: dict[str, Any]
    timeline: list[dict[str, Any]]
    embeddings: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "metadata": self.metadata.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "scene_boundaries": [boundary.to_dict() for boundary in self.scene_boundaries],
            "statistics": self.statistics,
            "timeline": self.timeline,
        }


@dataclass(frozen=True, slots=True)
class SearchMatch:
    frame_position: int
    frame_index: int
    timestamp_s: float
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_position": self.frame_position,
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 4),
            "similarity": round(self.similarity, 5),
        }
