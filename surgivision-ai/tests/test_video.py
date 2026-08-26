from pathlib import Path

import pytest

from surgivision.config import Settings
from surgivision.exceptions import VideoValidationError
from surgivision.video.loader import VideoLoader


def test_video_metadata_parsing(sample_video: Path) -> None:
    metadata = VideoLoader(Settings()).probe(sample_video)

    assert metadata.filename == "sample.avi"
    assert metadata.fps == pytest.approx(10.0, rel=0.05)
    assert metadata.frame_count == 30
    assert metadata.duration_s == pytest.approx(3.0, rel=0.05)
    assert (metadata.width, metadata.height) == (96, 64)


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError, match="does not exist"):
        VideoLoader(Settings()).probe(tmp_path / "missing.mp4")


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "clip.txt"
    invalid.write_bytes(b"not a video")

    with pytest.raises(VideoValidationError, match="Unsupported video extension"):
        VideoLoader(Settings()).probe(invalid)


def test_invalid_video_stream_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"not a video")

    with pytest.raises(VideoValidationError, match="could not open"):
        VideoLoader(Settings()).probe(invalid)
