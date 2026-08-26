"""CLIP-compatible image and text embedding adapter."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image

from surgivision.exceptions import ModelLoadError
from surgivision.models.instrument_detector import select_torch_device
from surgivision.types import SampledFrame


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix while preserving all-zero rows."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim not in (1, 2):
        raise ValueError("Embeddings must be a vector or matrix")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


class FeatureEncoder(Protocol):
    device: str

    def encode_frames(self, frames: Sequence[SampledFrame]) -> np.ndarray:
        """Return one normalized embedding per frame."""

    def encode_text(self, query: str) -> np.ndarray:
        """Return one normalized embedding for a text query."""


class CLIPFeatureEncoder:
    """Lazy image/text encoder backed by an OpenCLIP checkpoint."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or select_torch_device()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device,
                )
                model.eval()
                self._model = model
                self._preprocess = preprocess
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
            except Exception as exc:  # pragma: no cover - depends on runtime/network
                raise ModelLoadError(
                    "Could not load the pretrained embedding model. "
                    "Check the model identifiers and first-run network access."
                ) from exc

    def encode_frames(self, frames: Sequence[SampledFrame]) -> np.ndarray:
        self._load()
        if not frames:
            return np.empty((0, 0), dtype=np.float32)

        import torch

        images = [
            self._preprocess(Image.fromarray(sample.image[:, :, ::-1])) for sample in frames
        ]
        batch = torch.stack(images).to(self.device)
        with torch.inference_mode():
            encoded = self._model.encode_image(batch)
        return normalize_embeddings(encoded.detach().float().cpu().numpy())

    def encode_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        self._load()

        import torch

        tokens = self._tokenizer([query]).to(self.device)
        with torch.inference_mode():
            encoded = self._model.encode_text(tokens)
        return normalize_embeddings(encoded.detach().float().cpu().numpy())[0]
