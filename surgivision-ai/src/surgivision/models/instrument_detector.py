"""Object-detection interface and Torchvision implementation."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from surgivision.exceptions import ModelLoadError
from surgivision.types import Detection, SampledFrame


def select_torch_device() -> str:
    """Select CUDA, Apple MPS, or CPU in descending priority."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in installation failures
        raise ModelLoadError("PyTorch is required for pretrained inference") from exc
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ObjectDetector(Protocol):
    device: str

    def detect(self, frames: Sequence[SampledFrame]) -> list[list[Detection]]:
        """Return detections aligned with the input frame order."""


class NoOpObjectDetector:
    """Detector used for model-free smoke tests and explicitly disabled inference."""

    device = "disabled"

    def detect(self, frames: Sequence[SampledFrame]) -> list[list[Detection]]:
        return [[] for _ in frames]


class GenericObjectDetector:
    """Lazy COCO-pretrained Faster R-CNN adapter.

    The checkpoint is general-purpose and its class names must not be interpreted as
    validated surgical-instrument predictions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_detections_per_frame: int = 50,
        device: str | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Detection confidence must be between zero and one")
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_frame = max_detections_per_frame
        self.device = device or select_torch_device()
        self._model = None
        self._categories: list[str] = []
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from torchvision.models.detection import (
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                    fasterrcnn_resnet50_fpn_v2,
                )

                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn_v2(weights=weights)
                model.eval().to(self.device)
                self._categories = list(weights.meta["categories"])
                self._model = model
            except Exception as exc:  # pragma: no cover - depends on runtime/network
                raise ModelLoadError(
                    "Could not load the pretrained object detector. "
                    "Check network access for the first model download."
                ) from exc

    def detect(self, frames: Sequence[SampledFrame]) -> list[list[Detection]]:
        self._load()
        if not frames:
            return []

        import torch

        tensors = []
        for sample in frames:
            rgb = np.ascontiguousarray(sample.image[:, :, ::-1])
            tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device))

        with torch.inference_mode():
            outputs = self._model(tensors)

        batch_results: list[list[Detection]] = []
        for sample, output in zip(frames, outputs, strict=True):
            frame_detections: list[Detection] = []
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels, strict=True):
                confidence = float(score)
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(label)
                class_name = (
                    self._categories[class_id]
                    if 0 <= class_id < len(self._categories)
                    else f"class_{class_id}"
                )
                frame_detections.append(
                    Detection(
                        timestamp_s=sample.timestamp_s,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=tuple(float(value) for value in box),
                    )
                )
                if len(frame_detections) >= self.max_detections_per_frame:
                    break
            batch_results.append(frame_detections)
        return batch_results
