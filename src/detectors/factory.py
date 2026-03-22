from __future__ import annotations

from src.config import DetectorConfig
from src.detectors.base import BaseDetector
from src.detectors.yolov8 import YoloV8Detector


def build_detector(config: DetectorConfig) -> BaseDetector:
    family = config.family.lower()
    if family != "yolov8":
        raise ValueError(
            f"Unsupported detector family '{config.family}'. "
            "The diploma comparison pipeline expects the YOLOv8 family."
        )

    return YoloV8Detector(config)
