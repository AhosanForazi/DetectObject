"""DetectObject — Light & Clean App (Image Detection, Video Detection, Live Webcam, Object Tracking, Model Information)."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
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
    render_page_head("Video Detection", "Upload MP4, AVI, MOV, or WEBM videos or select a built-in sample clip for frame-by-frame detection.")

    samples_dir = PROJECT_ROOT / "datasets" / "samples" / "videos"
    sample_videos = sorted(samples_dir.glob("*.mp4")) if samples_dir.exists() else []

    v1, v2, v3 = st.columns([2.2, 1.4, 0.8])
    with v1:
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_upload", label_visibility="collapsed")
    with v2:
        sample_choice = st.selectbox(
            "Sample Video",
            options=[None] + sample_videos,
            format_func=lambda p: "Try a sample video..." if p is None else p.name,
            key="vid_sample_select",
            label_visibility="collapsed",
        )
    with v3:
        with st.popover("Options", width="stretch"):
            conf = st.slider("Confidence", 0.05, 0.95, float(config.get("model", "confidence_threshold", default=0.35)), 0.05, key="vid_conf")
            iou = st.slider("IoU", 0.10, 0.95, float(config.get("model", "iou_threshold", default=0.45)), 0.05, key="vid_iou")
            imgsz = st.select_slider("Resolution (px)", options=[320, 416, 480, 640], value=320, key="vid_imgsz")
            frame_skip = st.selectbox("Frame Skip", [1, 2, 3, 4], index=0, key="vid_skip")

    video_source_path = None
    source_name = "video"
    is_temp = False

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()
        video_source_path = Path(tmp.name)
        source_name = Path(uploaded.name).stem
        is_temp = True
    elif sample_choice is not None:
        video_source_path = sample_choice
        source_name = sample_choice.stem

    if video_source_path is None:
        st.markdown('<div class="do-empty"><p>Upload a video file or select a sample clip above to start frame-by-frame object detection.</p></div>', unsafe_allow_html=True)
        return

    btn_col, _ = st.columns([1.5, 4.0])
    run_clicked = btn_col.button(f"Process {source_name}", type="primary", key="btn_run_video")

    if run_clicked:
        pbar = st.progress(0.0, text="Initializing video pipeline...")
        pipeline = VideoDetector(detector=detector, config=config)

        def _cb(curr: int, total: int, fps: float, snap: Any) -> None:
            pct = min(1.0, curr / max(1, total))
            pbar.progress(pct, text=f"Frame {curr}/{total} · {fps:.1f} FPS · {snap.frame_total} objects in frame")

        try:
            with st.spinner("Processing video frames..."):
                out = pipeline.process_video(
                    video_path=video_source_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    frame_skip=frame_skip,
                    tracking_enabled=False,
                    progress_callback=_cb,
                )
            st.session_state["last_video_output"] = out
            pbar.empty()
            st.success(f"Video processed successfully: {out.processed_frames} frames @ {out.avg_processing_fps:.1f} avg FPS")
        except Exception as exc:
            st.error(f"Video processing error: {exc}")
        finally:
            if is_temp and video_source_path and video_source_path.exists():
                try:
                    video_source_path.unlink(missing_ok=True)
                except Exception:
                    pass

    out = st.session_state.get("last_video_output")
    if out is not None and Path(out.output_path).exists():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Frames Processed", f"{out.processed_frames} / {out.total_frames}")
        k2.metric("Avg FPS", f"{out.avg_processing_fps:.1f}")
        k3.metric("Peak Objects", out.peak_frame_objects)
        k4.metric("Encoding", out.codec_used)

        video_bytes = Path(out.output_path).read_bytes()
        st.video(video_bytes)
        render_class_pills(out.cumulative_class_detections)
        st.download_button("Download Processed Video (MP4)", data=video_bytes, file_name=Path(out.output_path).name, mime="video/mp4")


# ==============================================================================
# 3. LIVE WEBCAM
# ==============================================================================
def page_live_webcam(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Live Webcam", "Real-time object detection, persistent tracking, counting, and screenshot capture.")

    tab_stream, tab_sim, tab_cloud = st.tabs(["⚡ Live Camera Stream (30 FPS)", "🎬 Live Camera Feed Simulation", "📸 Camera Snapshot"])

    with tab_stream:
        server = get_live_inference_server()
        server.set_detector(detector)

        st.caption("Stream live video directly from your webcam with real-time AI bounding boxes, ByteTrack IDs, and line counting.")
        render_live_webcam(
            api_base_url=server.base_url,
            default_conf=float(config.get("model", "confidence_threshold", default=0.35)),
            default_iou=float(config.get("model", "iou_threshold", default=0.45)),
            default_imgsz=int(config.get("webcam", "inference_imgsz", default=320)),
            default_tracker=str(config.get("tracking", "tracker_type", default="bytetrack")),
        )

    with tab_sim:
        st.caption("Test the live real-time detection pipeline with a simulated camera feed without needing camera hardware permissions.")
        samples_dir = PROJECT_ROOT / "datasets" / "samples" / "videos"
        sim_videos = sorted(samples_dir.glob("*.mp4")) if samples_dir.exists() else []

        s1, s2 = st.columns([3, 1])
        with s1:
            sim_choice = st.selectbox("Feed Source", options=sim_videos, format_func=lambda p: p.name, key="sim_video_choice")
        with s2:
            sim_conf = st.slider("Sim Confidence", 0.1, 0.9, 0.30, 0.05, key="sim_conf")

        col_start, col_stop = st.columns([1, 1])
        start_sim = col_start.button("Start Live Simulation", type="primary", key="sim_start_btn")
        stop_sim = col_stop.button("Stop Simulation", key="sim_stop_btn")

        if stop_sim:
            st.session_state["run_sim_active"] = False

        if start_sim:
            st.session_state["run_sim_active"] = True

        if st.session_state.get("run_sim_active", False) and sim_choice is not None:
            frame_holder = st.empty()
            stat_holder = st.empty()

            cap = cv2.VideoCapture(str(sim_choice))
            pipeline = ImageDetector(detector=detector, config=config)

            frame_count = 0
            t_start = time.time()
            try:
                while cap.isOpened() and st.session_state.get("run_sim_active", False) and frame_count < 60:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                    frame_count += 1
                    res = pipeline.detect_array(frame, conf=sim_conf, imgsz=320, save_output=False)
                    elapsed = max(0.001, time.time() - t_start)
                    sim_fps = frame_count / elapsed

                    frame_holder.image(res.annotated_rgb, channels="RGB", width="stretch")
                    stat_holder.markdown(
                        f"**Simulated Live Stream** · Frame {frame_count} · **{sim_fps:.1f} FPS** · Detections in frame: **{len(res.detections)}**"
                    )
                    time.sleep(0.04)  # ~25 FPS pacing
            finally:
                cap.release()

    with tab_cloud:
        st.caption("Capture live camera snapshots directly in your browser. Compatible with all devices, phones, and networks.")
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
    render_page_head("Object Tracking", "Track objects with persistent IDs across video frames with ByteTrack or BoT-SORT, motion trails, and counting lines.")

    samples_dir = PROJECT_ROOT / "datasets" / "samples" / "videos"
    sample_videos = sorted(samples_dir.glob("*.mp4")) if samples_dir.exists() else []

    t1, t2, t3 = st.columns([2.2, 1.4, 0.8])
    with t1:
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"], key="trk_upload", label_visibility="collapsed")
    with t2:
        sample_choice = st.selectbox(
            "Sample Video",
            options=[None] + sample_videos,
            format_func=lambda p: "Try a sample video..." if p is None else p.name,
            key="trk_sample_select",
            label_visibility="collapsed",
        )
    with t3:
        with st.popover("Options", width="stretch"):
            tracker_type = st.selectbox("Tracker", ["bytetrack", "botsort"], index=0, key="trk_algo")
            conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05, key="trk_conf")
            imgsz = st.select_slider("Resolution (px)", options=[320, 416, 480, 640], value=320, key="trk_imgsz")
            show_trails = st.toggle("Motion Trails", value=True, key="trk_trails")
            st.divider()
            line_enabled = st.toggle("Counting Line", value=True, key="trk_line")
            line_orientation = st.radio("Orientation", ["horizontal", "vertical"], horizontal=True, disabled=not line_enabled, key="trk_lo")
            line_pos = st.slider("Line Position", 0.1, 0.9, 0.5, 0.05, disabled=not line_enabled, key="trk_lp")
            st.divider()
            roi_enabled = st.toggle("ROI Zone", value=False, key="trk_roi")
            roi_x = st.slider("ROI X", 0.0, 1.0, (0.15, 0.85), 0.05, disabled=not roi_enabled, key="trk_rx")
            roi_y = st.slider("ROI Y", 0.0, 1.0, (0.15, 0.85), 0.05, disabled=not roi_enabled, key="trk_ry")

    video_source_path = None
    source_name = "video"
    is_temp = False

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()
        video_source_path = Path(tmp.name)
        source_name = Path(uploaded.name).stem
        is_temp = True
    elif sample_choice is not None:
        video_source_path = sample_choice
        source_name = sample_choice.stem

    if video_source_path is None:
        st.markdown('<div class="do-empty"><p>Upload a video or select a sample clip above to track objects with persistent IDs, trails, and zone counting.</p></div>', unsafe_allow_html=True)
        return

    btn_col, _ = st.columns([1.5, 4.0])
    run_clicked = btn_col.button(f"Track {source_name}", type="primary", key="btn_run_tracking")

    if run_clicked:
        pbar = st.progress(0.0, text="Initializing tracker...")
        pipeline = VideoDetector(detector=detector, config=config)
        count_cfg = CountingConfig(
            line_enabled=line_enabled,
            line_orientation=line_orientation,
            line_position=line_pos,
            roi_enabled=roi_enabled,
            roi_box=(roi_x[0], roi_y[0], roi_x[1], roi_y[1]),
        )

        def _cb(curr: int, total: int, fps: float, snap: Any) -> None:
            pct = min(1.0, curr / max(1, total))
            pbar.progress(pct, text=f"Frame {curr}/{total} · {fps:.1f} FPS · Unique IDs: {snap.unique_tracked_total}")

        try:
            with st.spinner(f"Running {tracker_type.upper()} object tracking..."):
                out = pipeline.process_video(
                    video_path=video_source_path,
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
            st.success(f"Tracking complete: {out.unique_tracked_total} unique objects tracked across {out.processed_frames} frames!")
        except Exception as exc:
            st.error(f"Tracking error: {exc}")
        finally:
            if is_temp and video_source_path and video_source_path.exists():
                try:
                    video_source_path.unlink(missing_ok=True)
                except Exception:
                    pass

    out = st.session_state.get("last_tracking_output")
    if out is not None and Path(out.output_path).exists():
        snap = out.final_count_snapshot
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unique Tracked Objects", out.unique_tracked_total)
        k2.metric("Virtual Line Crossings", f"{snap.line_in_count} In / {snap.line_out_count} Out")
        k3.metric("ROI Objects", f"{snap.roi_current_count} in zone / {snap.roi_unique_total} total")
        k4.metric("Avg Speed", f"{out.avg_processing_fps:.1f} FPS")

        vbytes = Path(out.output_path).read_bytes()
        st.video(vbytes)
        render_class_pills(out.unique_tracked_by_class)
        st.download_button("Download Tracked Video (MP4)", data=vbytes, file_name=Path(out.output_path).name, mime="video/mp4")


# ==============================================================================
# 5. MODEL INFORMATION
# ==============================================================================
def page_model_information(config: AppConfig, detector: ObjectDetector) -> None:
    render_page_head("Model Information", "Active weights, device info, measured test performance, and visual evaluation reports.")

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
                key="model_page_weights_select",
            )
        with w2:
            st.write("")
            if chosen.resolve() != detector.weights_path.resolve():
                if st.button("Apply Model", type="primary", key="btn_apply_model"):
                    st.session_state["selected_weights_path"] = str(chosen)
                    st.rerun()

    info = detector.model_info
    dev = detect_compute_device("auto")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Model", info.get("weights_name", "N/A"))
    m2.metric("Compute Device", detector.device.upper())
    m3.metric("Model Parameters", f"{info.get('parameters', 0):,}")
    m4.metric("Supported Classes", str(info.get("num_classes", 0)))

    # Measured independent test evaluation
    eval_path = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        st.divider()
        st.markdown("### Measured Test Evaluation Performance")
        st.caption(f"Evaluated on held-out test split ({eval_data.get('split_evaluated', 'test')}) · Zero data leakage · Hardware: {dev.get('platform', 'CPU')}")

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Precision", f"{eval_data['precision']:.3f}")
        e2.metric("Recall", f"{eval_data['recall']:.3f}")
        e3.metric("F1-Score", f"{eval_data['f1_score']:.3f}")
        e4.metric("mAP@0.5", f"{eval_data['map50']:.3f}")
        e5.metric("mAP@0.5:0.95", f"{eval_data['map50_95']:.3f}")

        # Tabbed visual evaluation plots
        plots_dir = PROJECT_ROOT / "reports" / "plots"
        if plots_dir.exists():
            st.markdown("#### Evaluation Plots & Performance Curves")
            p_tab1, p_tab2, p_tab3 = st.tabs(["📊 Precision-Recall & Confusion Matrix", "📉 Training & Validation Loss", "⚡ Inference Speed & Class Breakdown"])

            with p_tab1:
                col_cm, col_pr = st.columns(2)
                cm_img = plots_dir / "confusion_matrix.png"
                pr_img = plots_dir / "pr_curve.png"
                f1_img = plots_dir / "f1_curve.png"
                if cm_img.exists():
                    col_cm.image(str(cm_img), caption="Confusion Matrix", width="stretch")
                if pr_img.exists():
                    col_pr.image(str(pr_img), caption="Precision-Recall Curve", width="stretch")
                if f1_img.exists():
                    st.image(str(f1_img), caption="F1 Confidence Curve", width="stretch")

            with p_tab2:
                col_tl, col_vl = st.columns(2)
                tl_img = plots_dir / "training_loss.png"
                vl_img = plots_dir / "validation_loss.png"
                if tl_img.exists():
                    col_tl.image(str(tl_img), caption="Training Loss Curve", width="stretch")
                if vl_img.exists():
                    col_vl.image(str(vl_img), caption="Validation Loss Curve", width="stretch")

            with p_tab3:
                col_fps, col_cls = st.columns(2)
                fps_img = plots_dir / "fps_benchmark.png"
                cls_img = plots_dir / "class_performance.png"
                if fps_img.exists():
                    col_fps.image(str(fps_img), caption="Measured FPS Benchmark across Resolutions", width="stretch")
                if cls_img.exists():
                    col_cls.image(str(cls_img), caption="Per-Class Performance Comparison", width="stretch")

                if eval_data.get("fps_benchmarks"):
                    st.dataframe(pd.DataFrame(eval_data["fps_benchmarks"])[["imgsz", "mean_latency_ms", "measured_fps", "device"]], width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Supported Object Classes")
    search = st.text_input("Search 80 COCO classes", placeholder="e.g. person, car, dog, bottle, laptop...")
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

    detector = get_active_detector(config)

    # 1. Sidebar: Model selector & Quick Info
    with st.sidebar:
        st.markdown('<div class="do-brand">DetectObject</div>', unsafe_allow_html=True)
        st.caption("Active Model Weights")
        weights_list = discover_available_weights()
        if weights_list:
            chosen = st.selectbox(
                "Weights",
                options=weights_list,
                format_func=lambda p: p.name,
                index=0,
                key="sidebar_weights_select",
                label_visibility="collapsed",
            )
            if chosen and str(chosen) != st.session_state.get("selected_weights_path"):
                st.session_state["selected_weights_path"] = str(chosen)
                st.rerun()

        st.caption(f"Device: **{detector.device.upper()}**")
        st.divider()
        st.caption(f"Model: **{detector.weights_path.name}**")
        st.caption(f"Classes: **{len(detector.class_names)}**")
        st.caption("Architecture: **YOLO11n**")

    # Navigation Tabs (Clean & Minimalist, 100% Clickable)
    t_img, t_vid, t_cam, t_trk, t_mod = st.tabs([
        "Image Detection",
        "Video Detection",
        "Live Webcam",
        "Object Tracking",
        "Model Information",
    ])

    with t_img:
        page_image_detection(config, detector)

    with t_vid:
        page_video_detection(config, detector)

    with t_cam:
        page_live_webcam(config, detector)

    with t_trk:
        page_object_tracking(config, detector)

    with t_mod:
        page_model_information(config, detector)


if __name__ == "__main__":
    main()
