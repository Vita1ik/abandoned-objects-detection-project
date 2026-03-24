from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DetectorConfig:
    family: str = "yolov8"
    variant: str = "baseline"
    weights_path: str | None = None
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    tracker: str = "bytetrack.yaml"
    persist_tracking: bool = True
    device: str = "cpu"
    target_classes: list[str] = field(
        default_factory=lambda: ["person", "backpack", "handbag", "suitcase"]
    )


@dataclass(slots=True)
class LogicConfig:
    abandon_threshold: int = 5
    dist_threshold: int = 160
    fix_confirm_threshold: float = 0.7


@dataclass(slots=True)
class PreprocessingConfig:
    enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    brightness_beta: int = 8
    contrast_alpha: float = 1.08
    gamma: float = 0.95


@dataclass(slots=True)
class RuntimeConfig:
    detect_every_n_frames: int = 2


@dataclass(slots=True)
class AppConfig:
    video_source: int | str = 0
    window_title: str = "Abandoned Object System - MSc Thesis Project"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig(
        video_source=raw.get("video_source", 0),
        window_title=raw.get(
            "window_title", "Abandoned Object System - MSc Thesis Project"
        ),
        detector=_parse_detector_config(raw.get("detector", {})),
        logic=_parse_logic_config(raw.get("logic", {})),
        preprocessing=_parse_preprocessing_config(raw.get("preprocessing", {})),
        runtime=_parse_runtime_config(raw.get("runtime", {})),
    )


def _parse_detector_config(raw: dict[str, Any]) -> DetectorConfig:
    return DetectorConfig(
        family=raw.get("family", "yolov8"),
        variant=raw.get("variant", "baseline"),
        weights_path=raw.get("weights_path"),
        confidence=float(raw.get("confidence", 0.25)),
        iou=float(raw.get("iou", 0.45)),
        image_size=int(raw.get("image_size", 640)),
        tracker=raw.get("tracker", "bytetrack.yaml"),
        persist_tracking=bool(raw.get("persist_tracking", True)),
        device=raw.get("device", "cpu"),
        target_classes=list(
            raw.get(
                "target_classes",
                ["person", "backpack", "handbag", "suitcase"],
            )
        ),
    )


def _parse_logic_config(raw: dict[str, Any]) -> LogicConfig:
    return LogicConfig(
        abandon_threshold=int(raw.get("abandon_threshold", 5)),
        dist_threshold=int(raw.get("dist_threshold", 160)),
        fix_confirm_threshold=float(raw.get("fix_confirm_threshold", 0.7)),
    )


def _parse_preprocessing_config(raw: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        enabled=bool(raw.get("enabled", True)),
        clahe_clip_limit=float(raw.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(raw.get("clahe_tile_grid_size", 8)),
        brightness_beta=int(raw.get("brightness_beta", 8)),
        contrast_alpha=float(raw.get("contrast_alpha", 1.08)),
        gamma=float(raw.get("gamma", 0.95)),
    )


def _parse_runtime_config(raw: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        detect_every_n_frames=max(1, int(raw.get("detect_every_n_frames", 2))),
    )
