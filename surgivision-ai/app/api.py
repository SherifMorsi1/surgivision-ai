"""FastAPI service for process-local video analysis sessions."""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from surgivision import __version__
from surgivision.analysis.store import AnalysisStore
from surgivision.config import Settings
from surgivision.exceptions import SurgiVisionError
from surgivision.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)
settings = Settings.from_env()
store = AnalysisStore(capacity=8)


@lru_cache(maxsize=1)
def get_pipeline() -> AnalysisPipeline:
    """Return one process-wide pipeline so loaded model weights are reused."""

    return AnalysisPipeline(settings=settings)


class SearchRequest(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=6, ge=1, le=50)


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    statistics: dict[str, Any]


class SearchResponse(BaseModel):
    analysis_id: str
    query: str
    matches: list[dict[str, Any]]


app = FastAPI(
    title="SurgiVision AI API",
    version=__version__,
    description="Research-oriented video preprocessing and computer-vision analysis.",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {settings.max_upload_mb} MB limit",
                )
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(
    video: Annotated[UploadFile, File(description="A supported video file")],
    sampling_interval_s: float = 2.0,
) -> AnalyzeResponse:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension '{suffix or 'none'}'",
        )
    if sampling_interval_s < 0.1 or sampling_interval_s > 600:
        raise HTTPException(
            status_code=422,
            detail="sampling_interval_s must be between 0.1 and 600 seconds",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="surgivision-api-") as temporary_directory:
            video_path = Path(temporary_directory) / f"upload{suffix}"
            await _save_upload(video, video_path)
            result = await run_in_threadpool(
                get_pipeline().analyze,
                video_path,
                sampling_interval_s,
            )
        store.put(result)
        return AnalyzeResponse(
            analysis_id=result.analysis_id,
            status="completed",
            statistics=result.statistics,
        )
    except HTTPException:
        raise
    except SurgiVisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail="Video analysis failed") from exc
    finally:
        await video.close()


@app.get("/analysis/{analysis_id}", tags=["analysis"])
def get_analysis(analysis_id: str) -> dict[str, Any]:
    result = store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    return result.to_dict()


@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchRequest) -> SearchResponse:
    result = store.get(request.analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Search query cannot be blank")
    try:
        matches = await run_in_threadpool(
            get_pipeline().search,
            result,
            request.query,
            request.top_k,
        )
    except SurgiVisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SearchResponse(
        analysis_id=request.analysis_id,
        query=request.query,
        matches=[match.to_dict() for match in matches],
    )
