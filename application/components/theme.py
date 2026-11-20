"""Light, clean, minimal CSS styling for DetectObject."""

from __future__ import annotations

import streamlit as st


def apply_theme(theme: str = "light") -> None:
    """Apply a crisp, airy, light-mode minimalist stylesheet."""
    st.markdown(
        """
        <style>
            #MainMenu, footer {
                visibility: hidden;
                height: 0;
            }
            header[data-testid="stHeader"] {
                background-color: transparent !important;
                z-index: 99 !important;
            }
            /* Make sure sidebar toggle button in header is always visible & styled */
            button[data-testid="stSidebarCollapseButton"],
            [data-testid="stHeader"] button,
            button[data-testid="baseButton-header"] {
                visibility: visible !important;
                display: flex !important;
                color: #0f172a !important;
                background-color: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 8px !important;
            }
            .block-container {
                padding-top: 1.0rem;
                padding-bottom: 2rem;
                max-width: 1200px;
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
            /* Top Navigation Navbar */
            .do-brand-title {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 1.15rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: #0f172a;
                white-space: nowrap;
            }
            .do-brand-title span {
                color: #2563eb;
                font-size: 1.3rem;
            }
            .do-nav-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.76rem;
                font-weight: 600;
                color: #475569;
                background: #f1f5f9;
                padding: 0.3rem 0.75rem;
                border-radius: 9999px;
                border: 1px solid #e2e8f0;
                white-space: nowrap;
            }
            .do-nav-badge .dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: #16a34a;
            }
            /* Segmented Top Menu Bar styling for horizontal radio */
            div[data-testid="stRadio"]:has(input[value="Image Detection"]),
            div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]),
            div[role="radiogroup"]:has(input[value="Image Detection"]) {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: wrap !important;
                align-items: center !important;
                gap: 0.35rem !important;
                background: #f1f5f9 !important;
                padding: 0.25rem 0.35rem !important;
                border-radius: 10px !important;
                border: 1px solid #e2e8f0 !important;
                width: fit-content !important;
                margin: 0 auto !important;
            }
            div[role="radiogroup"]:has(input[value="Image Detection"]) label {
                margin: 0 !important;
                padding: 0.38rem 0.85rem !important;
                border-radius: 7px !important;
                background: transparent !important;
                cursor: pointer !important;
                transition: all 0.15s ease-in-out !important;
            }
            div[role="radiogroup"]:has(input[value="Image Detection"]) label:hover {
                background: #e2e8f0 !important;
            }
            /* Hide default radio circle */
            div[role="radiogroup"]:has(input[value="Image Detection"]) div[data-testid="stRadioCircle"],
            div[role="radiogroup"]:has(input[value="Image Detection"]) label > div:first-child {
                display: none !important;
            }
            div[role="radiogroup"]:has(input[value="Image Detection"]) label p {
                font-size: 0.86rem !important;
                font-weight: 500 !important;
                color: #475569 !important;
                margin: 0 !important;
                white-space: nowrap !important;
            }
            /* Active selected item */
            div[role="radiogroup"]:has(input[value="Image Detection"]) label:has(input:checked),
            div[role="radiogroup"]:has(input[value="Image Detection"]) label[data-checked="true"] {
                background: #ffffff !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
            }
            div[role="radiogroup"]:has(input[value="Image Detection"]) label:has(input:checked) p,
            div[role="radiogroup"]:has(input[value="Image Detection"]) label[data-checked="true"] p {
                color: #2563eb !important;
                font-weight: 600 !important;
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
