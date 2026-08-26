"""Environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by the API, dashboard, and analysis pipeline."""

    sampling_interval_s: float = 2.0
    max_sampled_frames: int = 300
    blur_threshold: float = 80.0
    detection_confidence: float = 0.50
    detection_batch_size: int = 4
    embedding_batch_size: int = 16
    scene_distance_threshold: float = 0.18
    max_upload_mb: int = 500
    search_results: int = 6
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    instrument_like_labels: tuple[str, ...] = ("knife", "scissors")

    @classmethod
    def from_env(cls) -> "Settings":
        """Create validated settings from ``SURGIVISION_*`` environment variables."""

        return cls(
            sampling_interval_s=_env_float("SURGIVISION_SAMPLING_INTERVAL", 2.0, 0.1),
            max_sampled_frames=_env_int("SURGIVISION_MAX_FRAMES", 300, 1),
            blur_threshold=_env_float("SURGIVISION_BLUR_THRESHOLD", 80.0, 0.0),
            detection_confidence=_env_float("SURGIVISION_DETECTION_CONFIDENCE", 0.50, 0.0),
            detection_batch_size=_env_int("SURGIVISION_DETECTION_BATCH_SIZE", 4, 1),
            embedding_batch_size=_env_int("SURGIVISION_EMBEDDING_BATCH_SIZE", 16, 1),
            scene_distance_threshold=_env_float("SURGIVISION_SCENE_DISTANCE", 0.18, 0.0),
            max_upload_mb=_env_int("SURGIVISION_MAX_UPLOAD_MB", 500, 1),
            search_results=_env_int("SURGIVISION_SEARCH_RESULTS", 6, 1),
            clip_model=os.getenv("SURGIVISION_CLIP_MODEL", "ViT-B-32"),
            clip_pretrained=os.getenv(
                "SURGIVISION_CLIP_PRETRAINED", "laion2b_s34b_b79k"
            ),
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
