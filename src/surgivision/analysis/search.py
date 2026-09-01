"""Cosine-similarity ranking over cached frame embeddings."""

from __future__ import annotations

import numpy as np

from surgivision.exceptions import SearchUnavailableError
from surgivision.models.feature_encoder import FeatureEncoder, normalize_embeddings
from surgivision.types import AnalysisResult, SearchMatch


def rank_by_similarity(
    frame_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return ``(row_index, cosine_similarity)`` pairs in descending order."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    frames = normalize_embeddings(np.asarray(frame_embeddings, dtype=np.float32))
    query = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32))
    if frames.ndim != 2 or query.ndim != 1:
        raise ValueError("Expected a frame matrix and one query vector")
    if frames.shape[1] != query.shape[0]:
        raise ValueError("Frame and query embedding dimensions do not match")
    similarities = frames @ query
    count = min(top_k, similarities.shape[0])
    order = np.argsort(-similarities, kind="stable")[:count]
    return [(int(index), float(similarities[index])) for index in order]


class SimilaritySearch:
    """Search cached video frame embeddings with a compatible text encoder."""

    def __init__(self, encoder: FeatureEncoder) -> None:
        self.encoder = encoder

    def search(self, analysis: AnalysisResult, query: str, top_k: int = 6) -> list[SearchMatch]:
        if analysis.embeddings is None or len(analysis.frames) == 0:
            raise SearchUnavailableError("This analysis does not contain searchable embeddings")
        query_embedding = self.encoder.encode_text(query)
        ranked = rank_by_similarity(analysis.embeddings, query_embedding, top_k)
        return [
            SearchMatch(
                frame_position=position,
                frame_index=analysis.frames[position].sample.frame_index,
                timestamp_s=analysis.frames[position].sample.timestamp_s,
                similarity=similarity,
            )
            for position, similarity in ranked
        ]
