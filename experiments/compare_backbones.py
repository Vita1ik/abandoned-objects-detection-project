from __future__ import annotations

import csv
import time
from pathlib import Path

import yaml

from src.config import DetectorConfig
from src.detectors.factory import build_detector


MODEL_CONFIGS = [
    Path("configs/models/yolov8_baseline.yaml"),
    Path("configs/models/yolov8_mobilenetv3.yaml"),
    Path("configs/models/yolov8_mobilenetv4.yaml"),
]


def main() -> None:
    output_path = Path("logs/experiments/backbone_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_config_path in MODEL_CONFIGS:
        rows.append(evaluate_variant(model_config_path))

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "variant",
                "weights_path",
                "status",
                "load_time_sec",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison scaffold saved to {output_path}")


def evaluate_variant(model_config_path: Path) -> dict[str, object]:
    with model_config_path.open("r", encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh) or {}

    start = time.perf_counter()
    try:
        build_detector(
            DetectorConfig(
                family=model_config["family"],
                variant=model_config["variant"],
                weights_path=model_config["weights_path"],
            )
        )
        status = "ready"
    except FileNotFoundError:
        status = "weights_missing"
    except Exception as exc:  # pragma: no cover - helper for manual experiments
        status = f"error: {exc}"

    return {
        "model": model_config["name"],
        "variant": model_config["variant"],
        "weights_path": model_config["weights_path"],
        "status": status,
        "load_time_sec": round(time.perf_counter() - start, 4),
        "notes": model_config.get("notes", ""),
    }


if __name__ == "__main__":
    main()
