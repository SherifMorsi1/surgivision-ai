from pathlib import Path

import numpy as np

from surgivision.config import Settings
from surgivision.models.feature_encoder import normalize_embeddings
from surgivision.pipeline import AnalysisPipeline
from surgivision.types import Detection, SampledFrame


class StubDetector:
    device = "cpu-test"

    def detect(self, frames: list[SampledFrame]) -> list[list[Detection]]:
        return [
            [
                Detection(
                    timestamp_s=frame.timestamp_s,
                    class_name="scissors",
                    confidence=0.8,
                    bbox=(1.0, 2.0, 20.0, 25.0),
                )
            ]
            for frame in frames
        ]


class StubEncoder:
    device = "cpu-test"

    def encode_frames(self, frames: list[SampledFrame]) -> np.ndarray:
        features = np.array(
            [[float(frame.image[:, :, 2].mean()), 255.0] for frame in frames],
            dtype=np.float32,
        )
        return normalize_embeddings(features)

    def encode_text(self, query: str) -> np.ndarray:
        return normalize_embeddings(np.array([1.0, 1.0], dtype=np.float32))


def test_pipeline_with_injected_models(sample_video: Path) -> None:
    settings = Settings(
        max_sampled_frames=10,
        detection_batch_size=2,
        embedding_batch_size=2,
        scene_distance_threshold=0.01,
    )
    pipeline = AnalysisPipeline(
        settings=settings,
        detector=StubDetector(),
        encoder=StubEncoder(),
    )

    result = pipeline.analyze(sample_video, sampling_interval_s=0.5)
    matches = pipeline.search(result, "bright scene", top_k=3)

    assert len(result.frames) == 6
    assert result.embeddings is not None
    assert result.embeddings.shape == (6, 2)
    assert result.statistics["total_detections"] == 6
    assert result.statistics["instrument_like_detections"] == 6
    assert result.statistics["model_device"] == "cpu-test"
    assert len(matches) == 3
    assert all(frame.detections[0].class_name == "scissors" for frame in result.frames)


def test_model_free_pipeline_smoke(sample_video: Path) -> None:
    pipeline = AnalysisPipeline(settings=Settings(max_sampled_frames=4), enable_pretrained=False)

    result = pipeline.analyze(sample_video, sampling_interval_s=1.0)

    assert result.statistics["sampled_frames"] == 3
    assert result.statistics["total_detections"] == 0
    assert result.embeddings is None
