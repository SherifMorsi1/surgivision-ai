"""Timestamp-preserving frame sampling."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from surgivision.config import Settings
from surgivision.exceptions import FrameSamplingError
from surgivision.types import SampledFrame, VideoMetadata

logger = logging.getLogger(__name__)


class FrameSampler:
    """Sample frames at a fixed time interval without decoding every frame."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def sample_indices(
        self,
        metadata: VideoMetadata,
        interval_s: float | None = None,
        max_frames: int | None = None,
    ) -> list[int]:
        interval = self.settings.sampling_interval_s if interval_s is None else interval_s
        limit = self.settings.max_sampled_frames if max_frames is None else max_frames
        if interval <= 0:
            raise ValueError("Sampling interval must be greater than zero")
        if limit <= 0:
            raise ValueError("Maximum sampled frames must be greater than zero")

        step = max(1, round(interval * metadata.fps))
        return list(range(0, metadata.frame_count, step))[:limit]

    def sample(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        interval_s: float | None = None,
        max_frames: int | None = None,
    ) -> list[SampledFrame]:
        indices = self.sample_indices(metadata, interval_s, max_frames)
        capture = cv2.VideoCapture(str(path))
        sampled: list[SampledFrame] = []
        try:
            if not capture.isOpened():
                raise FrameSamplingError("Could not open the video for frame sampling")
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    logger.warning("Skipping undecodable frame %d", frame_index)
                    continue
                sampled.append(
                    SampledFrame(
                        frame_index=frame_index,
                        timestamp_s=frame_index / metadata.fps,
                        image=frame,
                    )
                )
        finally:
            capture.release()

        if not sampled:
            raise FrameSamplingError("No frames could be sampled from the video")
        return sampled
