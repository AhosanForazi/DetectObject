"""Minimal light-mode UI helpers for DetectObject."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.detector import Detection


def render_top_menu_bar(nav_pages: list[str], default_page: str, weights_name: str, device: str) -> str:
    """Render a persistent, prominent top navigation menu bar with branding, segmented tabs, and model status."""
    col_brand, col_nav, col_status = st.columns([1.4, 5.2, 1.4], vertical_alignment="center")

    with col_brand:
        st.markdown(
            '<div class="do-brand-title"><span>◎</span> DetectObject</div>',
            unsafe_allow_html=True,
        )

    with col_nav:
        curr_idx = nav_pages.index(default_page) if default_page in nav_pages else 0
        selected = st.radio(
            "Top Navigation Menu",
            nav_pages,
            index=curr_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="top_navbar_radio",
        )

    with col_status:
        st.markdown(
            f'<div style="text-align: right;"><span class="do-nav-badge"><span class="dot"></span> {weights_name} · {device.upper()}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 1px; background: #e2e8f0; margin: 0.5rem 0 1.25rem 0;"></div>', unsafe_allow_html=True)
    return selected


def render_page_head(title: str, subtitle: str) -> None:
    """Render a clean, lightweight page title and 1-line subtitle."""
    st.markdown(
        f"""
        <div class="do-page-head">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_pills(class_counts: dict[str, int], empty_text: str = "No objects detected.") -> None:
    """Render compact class count tags."""
    if not class_counts:
        st.caption(empty_text)
        return
    tags = "".join(f'<span class="do-tag">{cls} <b>{cnt}</b></span>' for cls, cnt in class_counts.items())
    st.markdown(f'<div class="do-pills">{tags}</div>', unsafe_allow_html=True)


def render_detection_table(detections: list[Detection], max_rows: int = 80) -> None:
    """Render a simple table of detections."""
    if not detections:
        st.caption("No detections.")
        return
    rows = []
    for det in detections[:max_rows]:
        row = {"Class": det.class_name, "Confidence": f"{det.confidence * 100:.0f}%"}
        if det.track_id is not None:
            row["Track ID"] = det.track_id
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
