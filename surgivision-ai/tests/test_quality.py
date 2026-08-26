import cv2
import numpy as np

from surgivision.types import SampledFrame
from surgivision.video.quality import QualityAnalyzer


def test_blur_detection_separates_sharp_and_blurred_frames() -> None:
    checkerboard = (np.indices((128, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
    sharp = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    sharp_score = QualityAnalyzer.blur_score(sharp)
    blurred_score = QualityAnalyzer.blur_score(blurred)
    threshold = (sharp_score + blurred_score) / 2
    analyzer = QualityAnalyzer(threshold)

    sharp_result = analyzer.analyze(SampledFrame(0, 0.0, sharp))
    blurred_result = analyzer.analyze(SampledFrame(1, 0.1, blurred))

    assert sharp_score > blurred_score
    assert sharp_result.is_low_quality is False
    assert blurred_result.is_low_quality is True
    assert blurred_result.timestamp_s == 0.1


def test_empty_image_is_rejected() -> None:
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    try:
        QualityAnalyzer.blur_score(empty)
    except ValueError as exc:
        assert "empty image" in str(exc)
    else:
        raise AssertionError("Expected an empty image to be rejected")
