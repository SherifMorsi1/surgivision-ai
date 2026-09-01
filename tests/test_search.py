import numpy as np
import pytest

from surgivision.analysis.search import rank_by_similarity
from surgivision.models.feature_encoder import normalize_embeddings


def test_embedding_normalization() -> None:
    embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = normalize_embeddings(embeddings)

    assert normalized[0] == pytest.approx([0.6, 0.8])
    assert normalized[1] == pytest.approx([0.0, 0.0])
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)


def test_similarity_ranking_is_descending_and_stable() -> None:
    frames = np.array(
        [
            [1.0, 0.0],
            [0.7, 0.7],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    ranked = rank_by_similarity(frames, np.array([1.0, 0.0]), top_k=2)

    assert [index for index, _ in ranked] == [0, 1]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[1][1] == pytest.approx(2**-0.5)


def test_similarity_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions do not match"):
        rank_by_similarity(np.ones((2, 3)), np.ones(4), top_k=1)
