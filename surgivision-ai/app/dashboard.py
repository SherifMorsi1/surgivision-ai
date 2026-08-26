"""Streamlit dashboard for interactive video analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from surgivision.config import Settings
from surgivision.exceptions import SurgiVisionError
from surgivision.pipeline import AnalysisPipeline
from surgivision.types import FrameAnalysis

st.set_page_config(page_title="SurgiVision AI", page_icon="SV", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(settings=Settings.from_env())


def format_timestamp(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"


def annotated_rgb(frame: FrameAnalysis):
    image = frame.sample.image.copy()
    for detection in frame.detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (46, 204, 113), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (46, 204, 113),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def progress_handler(progress_bar, status_box):
    stages = {
        "Reading video metadata": 0.03,
        "Sampling frames": 0.08,
        "Measuring frame quality": 0.25,
        "Running object detection": 0.58,
        "Encoding visual features": 0.90,
        "Finalizing analysis": 1.0,
    }

    def update(stage: str, current: int, total: int) -> None:
        base = stages.get(stage, 0.0)
        fraction = current / total if total else 0.0
        if stage == "Measuring frame quality":
            value = 0.08 + fraction * 0.17
        elif stage == "Running object detection":
            value = 0.25 + fraction * 0.33
        elif stage == "Encoding visual features":
            value = 0.58 + fraction * 0.32
        else:
            value = base
        progress_bar.progress(min(value, 1.0))
        status_box.caption(stage)

    return update


st.title("SurgiVision AI")
st.caption(
    "Research-oriented preprocessing, generic object detection, visual scene analysis, "
    "and semantic frame retrieval for laparoscopic or endoscopic video."
)

with st.sidebar:
    st.header("Analysis settings")
    sampling_interval = st.number_input(
        "Sampling interval (seconds)",
        min_value=0.1,
        max_value=600.0,
        value=2.0,
        step=0.5,
    )
    st.caption("Larger intervals reduce inference time and temporal resolution.")
    st.divider()
    st.subheader("Model scope")
    st.caption(
        "Object labels come from a general COCO checkpoint. Scene boundaries and search "
        "use a general CLIP-compatible encoder. Neither model is clinically validated."
    )

settings = Settings.from_env()
uploaded = st.file_uploader(
    "Upload a procedure video",
    type=[extension.lstrip(".") for extension in settings.allowed_extensions],
    help=f"Maximum configured upload size: {settings.max_upload_mb} MB",
)

if uploaded is not None:
    upload_bytes = uploaded.getvalue()
    if len(upload_bytes) > settings.max_upload_bytes:
        st.error(f"The selected file exceeds the {settings.max_upload_mb} MB limit.")
    else:
        st.video(upload_bytes)
        if st.button("Run analysis", type="primary", use_container_width=True):
            suffix = Path(uploaded.name).suffix.lower()
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="surgivision-dashboard-", suffix=suffix, delete=False
                ) as temporary_file:
                    temporary_file.write(upload_bytes)
                    temporary_path = Path(temporary_file.name)
                result = get_pipeline().analyze(
                    temporary_path,
                    sampling_interval_s=float(sampling_interval),
                    progress_callback=progress_handler(progress_bar, status_box),
                )
                st.session_state["analysis"] = result
                progress_bar.progress(1.0)
                status_box.caption("Analysis complete")
            except SurgiVisionError as exc:
                st.error(str(exc))
            except Exception:
                st.error(
                    "Analysis could not be completed. Check model download access and the "
                    "video codec, then try again."
                )
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("Upload a supported video and run analysis to populate the dashboard.")
    st.stop()

metadata = analysis.metadata
st.subheader("Video summary")
metric_columns = st.columns(5)
metric_columns[0].metric("Duration", format_timestamp(metadata.duration_s))
metric_columns[1].metric("Resolution", f"{metadata.width} × {metadata.height}")
metric_columns[2].metric("Frame rate", f"{metadata.fps:.2f} FPS")
metric_columns[3].metric("Sampled frames", analysis.statistics["sampled_frames"])
metric_columns[4].metric("Runtime", f"{analysis.statistics['runtime_s']:.2f} s")

overview_tab, timeline_tab, frames_tab, search_tab = st.tabs(
    ["Overview", "Timeline", "Sampled frames", "Semantic search"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Detection statistics")
        detection_counts = analysis.statistics["detections_by_class"]
        if detection_counts:
            detection_frame = pd.DataFrame(
                {"Class": list(detection_counts), "Count": list(detection_counts.values())}
            )
            st.plotly_chart(
                px.bar(detection_frame, x="Class", y="Count", color="Class"),
                use_container_width=True,
            )
        else:
            st.info("No objects met the configured confidence threshold.")
    with right:
        st.markdown("#### Quality and scene statistics")
        quality_columns = st.columns(2)
        quality_columns[0].metric(
            "Low-quality sampled frames",
            analysis.statistics["low_quality_frames"],
            f"{analysis.statistics['low_quality_percent']:.1f}%",
        )
        quality_columns[1].metric(
            "Visual scene boundaries", analysis.statistics["scene_boundaries"]
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Timestamp": format_timestamp(frame.sample.timestamp_s),
                        "Blur score": round(frame.quality.blur_score, 2),
                        "Low quality": frame.quality.is_low_quality,
                    }
                    for frame in analysis.frames
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

with timeline_tab:
    event_frame = pd.DataFrame(analysis.timeline)
    visible_events = event_frame[event_frame["event_type"] != "sampled_frame"]
    if visible_events.empty:
        st.info("No detection, quality, or scene-boundary events were observed.")
    else:
        timeline_figure = px.scatter(
            visible_events,
            x="timestamp_s",
            y="event_type",
            color="event_type",
            hover_data=["label", "value"],
            labels={"timestamp_s": "Video time (seconds)", "event_type": "Event"},
        )
        timeline_figure.update_traces(marker={"size": 11})
        st.plotly_chart(timeline_figure, use_container_width=True)
    instrument_events = event_frame[event_frame["event_type"] == "instrument_like_detection"]
    st.markdown("#### Instrument-like label usage")
    if instrument_events.empty:
        st.caption(
            "No COCO labels configured as instrument-like were detected. This is expected for "
            "many surgical videos because the detector is not surgical-specific."
        )
    else:
        st.dataframe(instrument_events, use_container_width=True, hide_index=True)

with frames_tab:
    st.caption("Bounding boxes show general-purpose detector outputs.")
    for row_start in range(0, len(analysis.frames), 3):
        columns = st.columns(3)
        for column, frame in zip(columns, analysis.frames[row_start : row_start + 3]):
            with column:
                st.image(annotated_rgb(frame), use_container_width=True)
                labels = ", ".join(d.class_name for d in frame.detections) or "No detections"
                quality_label = "low quality" if frame.quality.is_low_quality else "quality passed"
                st.caption(
                    f"{format_timestamp(frame.sample.timestamp_s)} · {quality_label} · {labels}"
                )

with search_tab:
    query = st.text_input(
        "Describe a visual moment",
        help="Examples: instrument visible, close-up scene, or dark frame.",
    )
    top_k = st.slider("Results", min_value=1, max_value=12, value=6)
    if st.button("Search frames", disabled=not query.strip()):
        try:
            matches = get_pipeline().search(analysis, query, top_k)
            for row_start in range(0, len(matches), 3):
                columns = st.columns(3)
                for column, match in zip(columns, matches[row_start : row_start + 3]):
                    frame = analysis.frames[match.frame_position]
                    with column:
                        st.image(annotated_rgb(frame), use_container_width=True)
                        st.caption(
                            f"{format_timestamp(match.timestamp_s)} · similarity "
                            f"{match.similarity:.3f}"
                        )
        except (SurgiVisionError, ValueError) as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "Research and educational software only. Not a medical device and not for diagnosis, "
    "treatment, surgical guidance, or patient-care decisions."
)
