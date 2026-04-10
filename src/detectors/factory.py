from __future__ import annotations

from src.config import DetectorConfig
from src.detectors.base import BaseDetector
from src.detectors.yolo26 import Yolo26Detector
from src.detectors.yolov8 import YoloV8Detector


def build_detector(config: DetectorConfig) -> BaseDetector:
    family = config.family.lower()
    if family == "yolov8":
        return YoloV8Detector(config)
    if family == "yolo26":
        return Yolo26Detector(config)

    raise ValueError(
        f"Unsupported detector family '{config.family}'. "
        "Supported families: yolov8, yolo26."
    )
