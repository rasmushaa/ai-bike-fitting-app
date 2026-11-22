"""The single-page Streamlit interface."""

import hashlib
import logging
import threading
from pathlib import Path

import streamlit as st

from .analysis import FPS_OPTIONS, OVERLAY_VERSION, AnalysisError, compile_video, process_video
from .player import player_html
from .model import ModelError, load_model

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading the YOLO pose model…")
def get_model_resource():
    return load_model(), threading.Lock()


def main() -> None:
    st.set_page_config(page_title="AI Bike Fitting", page_icon="🚲", layout="centered")
    st.title("AI Bike Fitting")
    st.write("Upload a side-on video of one rider on a stationary bike to see their pose and joint angles.")
    upload = st.file_uploader(
        "Cycling video", type=["mp4"],
        help="MP4 only · up to 100 MB · first 30 seconds analyzed",
    )
    with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
        side = st.radio("Side facing the camera", ["Right", "Left"], horizontal=True,
                        help="Choose the rider's anatomical side, rather than the side of the screen.").lower()
        fps = st.selectbox("Processing FPS", FPS_OPTIONS, index=0,
                           help="Higher FPS analyzes more frames and takes longer.")
        analyze = st.button("Analyze video", type="primary", disabled=upload is None, key="analyze")
    video = upload.getvalue() if upload is not None else None
    fingerprint = (OVERLAY_VERSION, hashlib.sha256(video).hexdigest(), side, fps) if video is not None else None
    if st.session_state.get("input_fingerprint") != fingerprint:
        st.session_state.input_fingerprint = fingerprint
        st.session_state.analysis_result = None
        st.session_state.video_export = None

    if analyze:
        st.session_state.analysis_result = None
        st.session_state.video_export = None
        progress = st.progress(0.0, text="Preparing video…")
        try:
            model, lock = get_model_resource()
            with st.spinner("Analyzing your cycling video…"):
                result = process_video(
                    video, Path(upload.name).suffix, side, model, lock,
                    fps=fps,
                    progress=lambda value: progress.progress(value, text=f"Estimating pose… {value:.0%}"),
                )
            st.session_state.analysis_result = result
        except (AnalysisError, ModelError) as exc:
            logger.exception("Video analysis failed")
            st.error(str(exc))
        except Exception:
            logger.exception("Unexpected video analysis failure")
            st.error("Analysis failed. Please retry or try another video. Server logs contain the details.")
        finally:
            progress.empty()

    result = st.session_state.get("analysis_result")
    if result is not None:
        st.iframe(player_html(result.frames, result.fps), height="content", alt="Cycling analysis frame player")
        st.caption(f"{result.side.title()} side · {result.duration_s:.1f} seconds · {len(result.samples)} frames · {result.fps} FPS")
        if st.button("Download Video", key="compile"):
            try:
                with st.spinner("Compiling analysis frames into an MP4…"):
                    st.session_state.video_export = compile_video(result)
            except AnalysisError as exc:
                logger.exception("Video compilation failed")
                st.error(str(exc))
        exported = st.session_state.get("video_export")
        if exported is not None:
            st.video(exported, format="video/mp4", alt="Compiled cycling analysis with pose, traces, and distributions")
            st.download_button("Download MP4", exported, "bike-fit-analysis.mp4", "video/mp4", on_click="ignore")
