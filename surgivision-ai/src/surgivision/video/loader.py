"""Video validation and metadata extraction."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from surgivision.config import Settings
from surgivision.exceptions import VideoValidationError
from surgivision.types import VideoMetadata


class VideoLoader:
    """Validate local video files and read container metadata through OpenCV."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def validate_path(self, path: str | Path) -> Path:
        video_path = Path(path)
        if not video_path.exists() or not video_path.is_file():
            raise VideoValidationError(f"Video file does not exist: {video_path}")
        if video_path.suffix.lower() not in self.settings.allowed_extensions:
            supported = ", ".join(self.settings.allowed_extensions)
            raise VideoValidationError(
                f"Unsupported video extension '{video_path.suffix}'. Supported: {supported}"
            )
        if video_path.stat().st_size == 0:
            raise VideoValidationError("Video file is empty")
        if video_path.stat().st_size > self.settings.max_upload_bytes:
            raise VideoValidationError(
                f"Video exceeds the {self.settings.max_upload_mb} MB upload limit"
            )
        return video_path

    def probe(self, path: str | Path) -> VideoMetadata:
        """Decode container metadata and reject files without a usable video stream."""

        video_path = self.validate_path(path)
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise VideoValidationError("OpenCV could not open the video stream")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))

            if not math.isfinite(fps) or fps <= 0:
                raise VideoValidationError("Video reports an invalid frame rate")
            if frame_count <= 0:
                raise VideoValidationError("Video contains no decodable frames")
            if width <= 0 or height <= 0:
                raise VideoValidationError("Video reports an invalid resolution")

            codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4))
            codec = "".join(character for character in codec if character.isprintable()).strip()
            return VideoMetadata(
                filename=video_path.name,
                fps=fps,
                frame_count=frame_count,
                duration_s=frame_count / fps,
                width=width,
                height=height,
                codec=codec or "unknown",
            )
        finally:
            capture.release()
