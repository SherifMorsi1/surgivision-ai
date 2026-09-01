import asyncio
from pathlib import Path

import httpx

import app.api as api_module
from surgivision.config import Settings
from surgivision.pipeline import AnalysisPipeline


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_endpoint() -> None:
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_analyze_requires_upload() -> None:
    response = request("POST", "/analyze")

    assert response.status_code == 422


def test_analyze_rejects_unsupported_extension() -> None:
    response = request(
        "POST",
        "/analyze",
        files={"video": ("notes.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported video extension" in response.json()["detail"]


def test_analyze_rejects_invalid_interval() -> None:
    response = request(
        "POST",
        "/analyze?sampling_interval_s=0",
        files={"video": ("clip.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 422


def test_missing_analysis_and_search_return_not_found() -> None:
    analysis_response = request("GET", "/analysis/unknown")
    search_response = request(
        "POST",
        "/search",
        json={"analysis_id": "unknown", "query": "dark frame", "top_k": 3},
    )

    assert analysis_response.status_code == 404
    assert search_response.status_code == 404


def test_search_schema_validation() -> None:
    response = request(
        "POST",
        "/search",
        json={"analysis_id": "session", "query": "query", "top_k": 0},
    )

    assert response.status_code == 422


def test_analyze_and_retrieve_with_lightweight_pipeline(
    sample_video: Path,
    monkeypatch,
) -> None:
    pipeline = AnalysisPipeline(
        settings=Settings(max_sampled_frames=4),
        enable_pretrained=False,
    )
    monkeypatch.setattr(api_module, "get_pipeline", lambda: pipeline)
    api_module.store.clear()

    response = request(
        "POST",
        "/analyze?sampling_interval_s=1",
        files={"video": ("sample.avi", sample_video.read_bytes(), "video/x-msvideo")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["statistics"]["sampled_frames"] == 3

    retrieval = request("GET", f"/analysis/{payload['analysis_id']}")
    assert retrieval.status_code == 200
    assert retrieval.json()["metadata"]["frame_count"] == 30
