"""Light, clean, minimal CSS styling for DetectObject."""

from __future__ import annotations

import streamlit as st


def apply_theme(theme: str = "light") -> None:
    """Apply a crisp, airy, light-mode minimalist stylesheet."""
    st.markdown(
        """
        <style>
            #MainMenu, footer, header[data-testid="stHeader"] {
                visibility: hidden;
                height: 0;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }
            .stApp {
                background-color: #ffffff;
                color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
            }
            section[data-testid="stSidebar"] {
                background-color: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }
            section[data-testid="stSidebar"] * {
                color: #0f172a;
            }
            .do-brand {
                font-size: 1.05rem;
                font-weight: 600;
                letter-spacing: -0.02em;
                color: #0f172a;
                margin-bottom: 0.75rem;
            }
            .do-page-head {
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 0.75rem;
                margin-bottom: 1.25rem;
            }
            .do-page-head h1 {
                margin: 0;
                font-size: 1.4rem;
                font-weight: 600;
                letter-spacing: -0.02em;
                color: #0f172a;
            }
            .do-page-head p {
                margin: 0.2rem 0 0 0;
                font-size: 0.88rem;
                color: #64748b;
            }
            .do-empty {
                border: 1px dashed #cbd5e1;
                border-radius: 10px;
                background: #f8fafc;
                padding: 3rem 1.5rem;
                text-align: center;
                color: #64748b;
                margin: 0.5rem 0 1rem 0;
            }
            .do-empty p {
                margin: 0;
                font-size: 0.88rem;
            }
            .do-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin: 0.5rem 0;
            }
            .do-tag {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.22rem 0.6rem;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
                background: #f8fafc;
                font-size: 0.82rem;
                color: #0f172a;
            }
            .do-tag b {
                color: #2563eb;
            }
            div[data-testid="stMetric"] {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.65rem 0.85rem;
            }
            div[data-testid="stMetric"] label {
                color: #64748b;
                font-size: 0.78rem;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 8px;
                font-weight: 500;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
