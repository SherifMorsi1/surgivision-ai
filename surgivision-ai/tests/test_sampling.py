from pathlib import Path

import pytest

from surgivision.config import Settings
from surgivision.video.loader import VideoLoader
from surgivision.video.sampler import FrameSampler


def test_sample_indices_and_timestamps(sample_video: Path) -> None:
    settings = Settings(max_sampled_frames=20)
    metadata = VideoLoader(settings).probe(sample_video)
    sampler = FrameSampler(settings)

    assert sampler.sample_indices(metadata, interval_s=0.5) == [0, 5, 10, 15, 20, 25]
    frames = sampler.sample(sample_video, metadata, interval_s=0.5)

    assert [frame.frame_index for frame in frames] == [0, 5, 10, 15, 20, 25]
    assert [frame.timestamp_s for frame in frames] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )
    assert all(frame.image.shape == (64, 96, 3) for frame in frames)


def test_sampling_respects_max_frames(sample_video: Path) -> None:
    settings = Settings(max_sampled_frames=3)
    metadata = VideoLoader(settings).probe(sample_video)

    frames = FrameSampler(settings).sample(sample_video, metadata, interval_s=0.1)

    assert len(frames) == 3


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_invalid_sampling_interval(sample_video: Path, interval: float) -> None:
    metadata = VideoLoader(Settings()).probe(sample_video)
    with pytest.raises(ValueError, match="greater than zero"):
        FrameSampler(Settings()).sample_indices(metadata, interval_s=interval)
