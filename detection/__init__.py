"""Detection pipelines for images, videos, live webcams, counting, and visualization."""

from detection.image_detector import ImageDetectionOutput, ImageDetector
from detection.object_counter import CountingConfig, CountSnapshot, ObjectCounter
from detection.video_detector import VideoDetectionOutput, VideoDetector
from detection.visualizer import DetectionVisualizer
from detection.webcam_detector import WebcamDetector, WebcamFramePacket, enumerate_cameras

__all__ = [
    "CountingConfig",
    "CountSnapshot",
    "DetectionVisualizer",
    "ImageDetectionOutput",
    "ImageDetector",
    "ObjectCounter",
    "VideoDetectionOutput",
    "VideoDetector",
    "WebcamDetector",
    "WebcamFramePacket",
    "enumerate_cameras",
]
