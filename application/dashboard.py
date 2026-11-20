"""DetectObject — Light & Clean App (Image Detection, Video Detection, Live Webcam, Object Tracking, Model Information)."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.components.live_webcam_ui import render_live_webcam
from application.components.theme import apply_theme
from application.components.widgets import (
    render_class_pills,
    render_detection_table,
    render_page_head,
)
from configuration.settings import AppConfig, detect_compute_device, load_config
from detection.image_detector import ImageDetector
from detection.live_api import LiveInferenceServer
from detection.object_counter import CountingConfig
from detection.video_detector import VideoDetector
from models.detector import ObjectDetector, OpenVocabularyDetector

st.set_page_config(
    page_title="DetectObject",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = [
    "Image Detection",
    "Video Detection",
    "Live Webcam",
    "Object Tracking",
    "Model Information",
]


@st.cache_resource(show_spinner="Loading model...")
def get_cached_detector(weights_path: str | None, device: str) -> ObjectDetector:
    return ObjectDetector(weights_path=weights_path, device=device)


@st.cache_resource(show_spinner="Loading open-vocabulary model...")
def get_cached_open_vocab_detector(device: str) -> OpenVocabularyDetector:
    return OpenVocabularyDetector(device=device)


@st.cache_resource
def get_live_inference_server() -> LiveInferenceServer:
    cfg = load_config()
    det = get_cached_detector(None, cfg.get("model", "device", default="auto"))
    server = LiveInferenceServer(detector=det, config=cfg)
    server.start()
    return server


def discover_available_weights() -> list[Path]:
    weights_dir = PROJECT_ROOT / "models" / "weights"
    return sorted(p for p in weights_dir.glob("*.pt") if p.is_file()) if weights_dir.exists() else []


def get_active_detector(config: AppConfig) -> ObjectDetector:
    weights_choice = st.session_state.get("selected_weights_path")
    device_choice = config.get("model", "device", default="auto")
    return get_cached_detector(weights_choice, device_choice)


# ==============================================================================
# 1. IMAGE DETECTION
# ==============================================================================
def page_image_detection(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Image Detection", "Upload JPG, PNG, or WEBP images to detect and count objects.")

    c1, c2, c3 = st.columns([2.3, 1.3, 0.8])
    with c1:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with c2:
        test_dir = PROJECT_ROOT / "datasets" / "processed" / "images" / "test"
        samples = sorted(test_dir.glob("*.jpg"))[:12] if test_dir.exists() else []
        sample_choice = st.selectbox(
            "Sample",
            options=[None] + samples,
            format_func=lambda p: "Try a sample image..." if p is None else p.name,
            label_visibility="collapsed",
        )
    with c3:
        with st.popover("Options", width="stretch"):
            conf = st.slider("Confidence", 0.05, 0.95, float(config.get("model", "confidence_threshold", default=0.35)), 0.05, key="img_conf")
            iou = st.slider("IoU", 0.10, 0.95, float(config.get("model", "iou_threshold", default=0.45)), 0.05, key="img_iou")
            imgsz = st.select_slider("Resolution (px)", options=[320, 416, 480, 640], value=480, key="img_imgsz")
            st.divider()
            open_vocab = st.toggle("Text Prompts (YOLO-World)", value=False, key="img_ov")
            prompt_text = st.text_input(
                "Prompts",
                value="person, laptop, smartphone, water bottle, chair, dog, cat",
                disabled=not open_vocab,
            )

    image_bgr = None
    source_name = "image"
    if uploaded is not None:
        try:
            image_bgr = ImageDetector.load_image_from_bytes(uploaded.getvalue(), uploaded.name)
            source_name = Path(uploaded.name).stem
        except ValueError as exc:
            st.error(str(exc))
            return
    elif sample_choice is not None:
        image_bgr = ImageDetector.load_image_from_path(sample_choice)
        source_name = sample_choice.stem

    if image_bgr is None:
        st.markdown('<div class="do-empty"><p>Upload an image or select a sample above to detect objects.</p></div>', unsafe_allow_html=True)
        return

    pipeline = ImageDetector(detector=detector, config=config)
    with st.spinner("Detecting..."):
        try:
            if open_vocab:
                prompts = [p.strip() for p in prompt_text.split(",") if p.strip()]
                out = pipeline.detect_open_vocab(
                    image_bgr,
                    get_cached_open_vocab_detector(detector.device),
                    prompts=prompts,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    save_output=True,
                )
            else:
                out = pipeline.detect_array(image_bgr, conf=conf, iou=iou, imgsz=imgsz, save_output=True, output_stem=f"{source_name}_detected")
        except Exception as exc:
            st.error(str(exc))
            return

    res = out.result
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Objects", res.total_count)
    m2.metric("Classes", len(res.class_counts))
    m3.metric("Latency", f"{res.inference_time_ms:.1f} ms")

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.caption("Original")
        st.image(out.original_rgb, width="stretch")
    with col_b:
        st.caption("Detected")
        st.image(out.annotated_rgb, width="stretch")

    render_class_pills(res.class_counts)

    d1, d2, _ = st.columns([1, 1, 2.5])
    d1.download_button("Download PNG", data=out.to_png_bytes(), file_name=f"{source_name}_detected.png", mime="image/png", width="stretch")
    d2.download_button("Download JSON", data=json.dumps(res.to_dict(), indent=2), file_name=f"{source_name}_detections.json", mime="application/json", width="stretch")

    if res.detections:
        with st.expander(f"Detections ({len(res.detections)})", expanded=False):
            render_detection_table(res.detections)


# ==============================================================================
# 2. VIDEO DETECTION
# ==============================================================================
def page_video_detection(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Video Detection", "Upload MP4, AVI, MOV, MKV, or WEBM videos for frame-by-frame detection and MP4 export.")

    v1, v2 = st.columns([3.0, 0.8])
    with v1:
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"], label_visibility="collapsed")
    with v2:
        with st.popover("Options", width="stretch"):
            conf = st.slider("Confidence", 0.05, 0.95, float(config.get("model", "confidence_threshold", default=0.35)), 0.05, key="vid_conf")
            iou = st.slider("IoU", 0.10, 0.95, float(config.get("model", "iou_threshold", default=0.45)), 0.05, key="vid_iou")
            imgsz = st.select_slider("Resolution (px)", options=[320, 416, 480, 640], value=320, key="vid_imgsz")
            frame_skip = st.selectbox("Frame Skip", [1, 2, 3, 4], index=0, key="vid_skip")

    if uploaded is not None:
        if st.button("Process Video", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            pbar = st.progress(0.0, text="Processing...")
            pipeline = VideoDetector(detector=detector, config=config)

            def _cb(curr: int, total: int, fps: float, snap: Any) -> None:
                pbar.progress(min(1.0, curr / max(1, total)), text=f"Frame {curr}/{total} · {fps:.1f} FPS · {snap.frame_total} objects")

            try:
                out = pipeline.process_video(
                    video_path=tmp_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    frame_skip=frame_skip,
                    tracking_enabled=False,
                    progress_callback=_cb,
                )
                st.session_state["last_video_output"] = out
                pbar.empty()
            except Exception as exc:
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)
    elif st.session_state.get("last_video_output") is None:
        st.markdown('<div class="do-empty"><p>Upload a video file above to start frame-by-frame object detection.</p></div>', unsafe_allow_html=True)

    out = st.session_state.get("last_video_output")
    if out is not None and Path(out.output_path).exists():
        k1, k2, k3 = st.columns(3)
        k1.metric("Frames Processed", f"{out.processed_frames} / {out.total_frames}")
        k2.metric("Avg FPS", f"{out.avg_processing_fps:.1f}")
        k3.metric("Peak Objects", out.peak_frame_objects)

        video_bytes = Path(out.output_path).read_bytes()
        st.video(video_bytes)
        render_class_pills(out.cumulative_class_detections)
        st.download_button("Download Processed Video (MP4)", data=video_bytes, file_name=Path(out.output_path).name, mime="video/mp4")


# ==============================================================================
# 3. LIVE WEBCAM
# ==============================================================================
def page_live_webcam(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Live Webcam", "Real-time object detection, tracking, counting, and screenshot capture from your camera.")

    tab_stream, tab_cloud = st.tabs(["Real-Time Stream (30 FPS)", "Cloud Snapshot Camera"])

    with tab_stream:
        server = get_live_inference_server()
        server.set_detector(detector)

        render_live_webcam(
            api_base_url=server.base_url,
            default_conf=float(config.get("model", "confidence_threshold", default=0.35)),
            default_iou=float(config.get("model", "iou_threshold", default=0.45)),
            default_imgsz=int(config.get("webcam", "inference_imgsz", default=320)),
            default_tracker=str(config.get("tracking", "tracker_type", default="bytetrack")),
        )

    with tab_cloud:
        st.caption(
            "Capture live photos directly from your browser. "
            "Works across all cloud hosting platforms (Streamlit Community Cloud, Hugging Face Spaces) without requiring secondary network ports."
        )
        cam_photo = st.camera_input("Take Camera Snapshot", key="cloud_webcam_input", label_visibility="collapsed")
        if cam_photo is not None:
            pipeline = ImageDetector(detector=detector, config=config)
            image_bgr = ImageDetector.load_image_from_bytes(cam_photo.getvalue(), "webcam_snap.jpg")
            conf = float(config.get("model", "confidence_threshold", default=0.35))
            iou = float(config.get("model", "iou_threshold", default=0.45))
            res = pipeline.detect_array(image_bgr, conf=conf, iou=iou, imgsz=320, save_output=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("Objects in Frame", len(res.detections))
            k2.metric("Latency", f"{res.inference_ms:.1f} ms")
            k3.metric("FPS Equivalent", f"{1000.0 / max(1.0, res.inference_ms):.1f}")

            st.image(res.annotated_rgb, channels="RGB", width="stretch")
            render_class_pills(res.class_counts)
            st.download_button(
                "Download Annotated Snapshot (PNG)",
                data=res.to_png_bytes(),
                file_name=f"webcam_snapshot_{int(time.time())}.png",
                mime="image/png",
            )


# ==============================================================================
# 4. OBJECT TRACKING
# ==============================================================================
def page_object_tracking(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Object Tracking", "Track unique object IDs across video frames with ByteTrack or BoT-SORT, counting lines, and ROI zones.")

    t1, t2 = st.columns([3.0, 0.8])
    with t1:
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"], key="trk_upload", label_visibility="collapsed")
    with t2:
        with st.popover("Options", width="stretch"):
            tracker_type = st.selectbox("Tracker", ["bytetrack", "botsort"], index=0, key="trk_algo")
            conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05, key="trk_conf")
            imgsz = st.select_slider("Resolution (px)", options=[320, 416, 480, 640], value=320, key="trk_imgsz")
            show_trails = st.toggle("Trails", value=True, key="trk_trails")
            st.divider()
            line_enabled = st.toggle("Counting Line", value=True, key="trk_line")
            line_orientation = st.radio("Orientation", ["horizontal", "vertical"], horizontal=True, disabled=not line_enabled, key="trk_lo")
            line_pos = st.slider("Line Position", 0.1, 0.9, 0.5, 0.05, disabled=not line_enabled, key="trk_lp")
            st.divider()
            roi_enabled = st.toggle("ROI Zone", value=False, key="trk_roi")
            roi_x = st.slider("ROI X", 0.0, 1.0, (0.15, 0.85), 0.05, disabled=not roi_enabled, key="trk_rx")
            roi_y = st.slider("ROI Y", 0.0, 1.0, (0.15, 0.85), 0.05, disabled=not roi_enabled, key="trk_ry")

    if uploaded is not None:
        if st.button("Run Tracking", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            pbar = st.progress(0.0, text="Tracking...")
            pipeline = VideoDetector(detector=detector, config=config)
            count_cfg = CountingConfig(
                line_enabled=line_enabled,
                line_orientation=line_orientation,
                line_position=line_pos,
                roi_enabled=roi_enabled,
                roi_box=(roi_x[0], roi_y[0], roi_x[1], roi_y[1]),
            )

            def _cb(curr: int, total: int, fps: float, snap: Any) -> None:
                pbar.progress(min(1.0, curr / max(1, total)), text=f"Frame {curr}/{total} · Unique IDs: {snap.unique_tracked_total}")

            try:
                out = pipeline.process_video(
                    video_path=tmp_path,
                    conf=conf,
                    imgsz=imgsz,
                    tracking_enabled=True,
                    tracker_type=tracker_type,
                    show_trails=show_trails,
                    counting_config=count_cfg,
                    progress_callback=_cb,
                )
                st.session_state["last_tracking_output"] = out
                pbar.empty()
            except Exception as exc:
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)
    elif st.session_state.get("last_tracking_output") is None:
        st.markdown('<div class="do-empty"><p>Upload a video above to track objects with unique IDs, trails, and zone counting.</p></div>', unsafe_allow_html=True)

    out = st.session_state.get("last_tracking_output")
    if out is not None and Path(out.output_path).exists():
        snap = out.final_count_snapshot
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unique Tracked", out.unique_tracked_total)
        k2.metric("Line In / Out", f"{snap.line_in_count} / {snap.line_out_count}")
        k3.metric("ROI Unique", snap.roi_unique_total)
        k4.metric("Avg FPS", f"{out.avg_processing_fps:.1f}")

        vbytes = Path(out.output_path).read_bytes()
        st.video(vbytes)
        render_class_pills(out.unique_tracked_by_class)
        st.download_button("Download Tracked Video (MP4)", data=vbytes, file_name=Path(out.output_path).name, mime="video/mp4")


# ==============================================================================
# 5. MODEL INFORMATION
# ==============================================================================
def page_model_information(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Model Information", "Active weights, device info, measured test performance, and supported classes.")

    weights = discover_available_weights()
    if weights:
        w1, w2 = st.columns([2.5, 1.0])
        current_idx = next((i for i, p in enumerate(weights) if p.resolve() == detector.weights_path.resolve()), 0)
        with w1:
            chosen = st.selectbox(
                "Model Weights",
                options=weights,
                index=current_idx,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024*1024):.2f} MB)",
            )
        with w2:
            st.write("")
            if st.button("Load Model", type="primary", disabled=(chosen.resolve() == detector.weights_path.resolve()), width="stretch"):
                st.session_state["selected_weights_path"] = str(chosen)
                st.rerun()

    info = detector.model_info
    dev = detect_compute_device("auto")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Model", info.get("weights_name", "N/A"))
    m2.metric("Device", detector.device.upper())
    m3.metric("Parameters", f"{info.get('parameters', 0):,}")
    m4.metric("Classes", str(info.get("num_classes", 0)))

    # Compact measured evaluation summary if available
    eval_path = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        st.caption(f"Measured Independent Test Split ({eval_data.get('split_evaluated', 'test')}) · CUDA Available: {dev['cuda_available']}")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Precision", f"{eval_data['precision']:.3f}")
        e2.metric("Recall", f"{eval_data['recall']:.3f}")
        e3.metric("F1-Score", f"{eval_data['f1_score']:.3f}")
        e4.metric("mAP@0.5", f"{eval_data['map50']:.3f}")
        e5.metric("mAP@0.5:0.95", f"{eval_data['map50_95']:.3f}")

        if eval_data.get("fps_benchmarks"):
            st.dataframe(pd.DataFrame(eval_data["fps_benchmarks"])[["imgsz", "mean_latency_ms", "measured_fps", "device"]], width="stretch", hide_index=True)

    search = st.text_input("Search supported classes", placeholder="e.g. person, laptop, car, bottle...")
    rows = [
        {"ID": cid, "Class": name}
        for cid, name in sorted(detector.class_names.items())
        if not search or search.lower() in name.lower()
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=280)


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    config = load_config()
    apply_theme("light")

    with st.sidebar:
        st.markdown('<div class="do-brand">DetectObject</div>', unsafe_allow_html=True)
        selected_page = st.radio("Navigation", NAV_PAGES, index=0, label_visibility="collapsed")
        st.divider()
        detector = get_active_detector(config)
        st.caption(f"{detector.weights_path.name} · {detector.device.upper()}")

    if selected_page == "Image Detection":
        page_image_detection(config, detector)
    elif selected_page == "Video Detection":
        page_video_detection(config, detector)
    elif selected_page == "Live Webcam":
        page_live_webcam(config, detector)
    elif selected_page == "Object Tracking":
        page_object_tracking(config, detector)
    elif selected_page == "Model Information":
        page_model_information(config, detector)


if __name__ == "__main__":
    main()
