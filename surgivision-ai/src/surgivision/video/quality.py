"""Deterministic frame-quality measurements."""

from __future__ import annotations

import cv2
import numpy as np

from surgivision.types import QualityMeasurement, SampledFrame


class QualityAnalyzer:
    """Flag unusually blurry frames using variance of the Laplacian."""

    def __init__(self, blur_threshold: float = 80.0) -> None:
        if blur_threshold < 0:
            raise ValueError("Blur threshold cannot be negative")
        self.blur_threshold = blur_threshold

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        if image.size == 0:
            raise ValueError("Cannot evaluate an empty image")
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            grayscale = cv2.cvtColor(image, conversion)
        else:
            raise ValueError("Expected a grayscale, BGR, or BGRA image")
        return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    def analyze(self, sample: SampledFrame) -> QualityMeasurement:
        score = self.blur_score(sample.image)
        return QualityMeasurement(
            timestamp_s=sample.timestamp_s,
            blur_score=score,
            is_low_quality=score < self.blur_threshold,
        )
