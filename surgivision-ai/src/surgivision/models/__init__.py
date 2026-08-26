"""Pretrained model adapters."""

from surgivision.models.feature_encoder import CLIPFeatureEncoder, normalize_embeddings
from surgivision.models.instrument_detector import (
    GenericObjectDetector,
    NoOpObjectDetector,
)

__all__ = [
    "CLIPFeatureEncoder",
    "GenericObjectDetector",
    "NoOpObjectDetector",
    "normalize_embeddings",
]
