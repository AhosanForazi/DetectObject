"""Minimal light-mode UI helpers for DetectObject."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.detector import Detection


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
