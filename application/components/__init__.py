"""Light, clean UI components for DetectObject."""

from application.components.live_webcam_ui import render_live_webcam
from application.components.theme import apply_theme
from application.components.widgets import (
    render_class_pills,
    render_detection_table,
    render_page_head,
)

__all__ = [
    "apply_theme",
    "render_class_pills",
    "render_detection_table",
    "render_live_webcam",
    "render_page_head",
]
