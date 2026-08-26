"""End-to-end orchestration for surgical-video research analysis."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from surgivision.analysis.search import SimilaritySearch
from surgivision.analysis.statistics import calculate_statistics
from surgivision.analysis.timeline import build_timeline, detect_scene_boundaries
from surgivision.config import Settings
from surgivision.models.feature_encoder import CLIPFeatureEncoder, FeatureEncoder
from surgivision.models.instrument_detector import (
    GenericObjectDetector,
    NoOpObjectDetector,
    ObjectDetector,
)
from surgivision.types import AnalysisResult, FrameAnalysis, SearchMatch
from surgivision.video.loader import VideoLoader
from surgivision.video.quality import QualityAnalyzer
from surgivision.video.sampler import FrameSampler

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]


class AnalysisPipeline:
    """Coordinate video decoding, inference, scene analysis, and search caching."""

    def __init__(
        self,
        settings: Settings | None = None,
        detector: ObjectDetector | None = None,
        encoder: FeatureEncoder | None = None,
        *,
        enable_pretrained: bool = True,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.loader = VideoLoader(self.settings)
        self.sampler = FrameSampler(self.settings)
        self.quality = QualityAnalyzer(self.settings.blur_threshold)
        self.detector = detector or (
            GenericObjectDetector(self.settings.detection_confidence)
            if enable_pretrained
            else NoOpObjectDetector()
        )
        self.encoder = encoder or (
            CLIPFeatureEncoder(
                model_name=self.settings.clip_model,
                pretrained=self.settings.clip_pretrained,
            )
            if enable_pretrained
            else None
        )

    @staticmethod
    def _progress(
        callback: ProgressCallback | None, stage: str, current: int, total: int
    ) -> None:
        if callback is not None:
            callback(stage, current, total)

    def analyze(
        self,
        video_path: str | Path,
        sampling_interval_s: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Run the configured analysis and retain embeddings in the returned session."""

        started = time.perf_counter()
        self._progress(progress_callback, "Reading video metadata", 0, 1)
        metadata = self.loader.probe(video_path)
        self._progress(progress_callback, "Sampling frames", 0, 1)
        samples = self.sampler.sample(
            video_path,
            metadata,
            interval_s=sampling_interval_s,
        )

        frames: list[FrameAnalysis] = []
        for index, sample in enumerate(samples, start=1):
            frames.append(FrameAnalysis(sample=sample, quality=self.quality.analyze(sample)))
            self._progress(progress_callback, "Measuring frame quality", index, len(samples))

        detection_batch_size = self.settings.detection_batch_size
        for start_index in range(0, len(samples), detection_batch_size):
            batch = samples[start_index : start_index + detection_batch_size]
            batch_detections = self.detector.detect(batch)
            if len(batch_detections) != len(batch):
                raise RuntimeError("Detector output count does not match its input batch")
            for offset, detections in enumerate(batch_detections):
                frames[start_index + offset].detections = detections
            self._progress(
                progress_callback,
                "Running object detection",
                min(start_index + len(batch), len(samples)),
                len(samples),
            )

        embeddings: np.ndarray | None = None
        if self.encoder is not None:
            embedding_batches: list[np.ndarray] = []
            embedding_batch_size = self.settings.embedding_batch_size
            for start_index in range(0, len(samples), embedding_batch_size):
                batch = samples[start_index : start_index + embedding_batch_size]
                embedding_batches.append(self.encoder.encode_frames(batch))
                self._progress(
                    progress_callback,
                    "Encoding visual features",
                    min(start_index + len(batch), len(samples)),
                    len(samples),
                )
            embeddings = np.concatenate(embedding_batches, axis=0)

        boundaries = detect_scene_boundaries(
            frames,
            embeddings,
            self.settings.scene_distance_threshold,
        )
        elapsed = time.perf_counter() - started
        model_device = getattr(self.encoder, "device", getattr(self.detector, "device", "unknown"))
        statistics = calculate_statistics(
            metadata,
            frames,
            boundaries,
            runtime_s=elapsed,
            model_device=model_device,
            instrument_like_labels=self.settings.instrument_like_labels,
        )
        timeline = build_timeline(frames, boundaries, self.settings.instrument_like_labels)
        self._progress(progress_callback, "Finalizing analysis", 1, 1)
        logger.info(
            "Analysis completed: %s frames in %.2f seconds on %s",
            len(frames),
            elapsed,
            model_device,
        )
        return AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            metadata=metadata,
            frames=frames,
            scene_boundaries=boundaries,
            statistics=statistics,
            timeline=timeline,
            embeddings=embeddings,
        )

    def search(
        self,
        analysis: AnalysisResult,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        if self.encoder is None:
            from surgivision.exceptions import SearchUnavailableError

            raise SearchUnavailableError("Semantic search is disabled for this pipeline")
        return SimilaritySearch(self.encoder).search(
            analysis,
            query,
            top_k=top_k or self.settings.search_results,
        )
