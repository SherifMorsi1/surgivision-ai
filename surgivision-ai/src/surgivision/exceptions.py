"""Domain-specific exceptions surfaced by the application layers."""


class SurgiVisionError(Exception):
    """Base exception for recoverable application errors."""


class VideoValidationError(SurgiVisionError):
    """Raised when a file is missing, unsupported, or cannot be decoded."""


class FrameSamplingError(SurgiVisionError):
    """Raised when no usable frames can be sampled from a valid video."""


class ModelLoadError(SurgiVisionError):
    """Raised when a configured pretrained model cannot be initialized."""


class SearchUnavailableError(SurgiVisionError):
    """Raised when semantic search is requested without compatible embeddings."""
