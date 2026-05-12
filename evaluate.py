from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import load_app_config
from src.training.custom_backbones import register_ultralytics_custom_backbones


DEFAULT_BASELINE_DESCRIPTOR = Path("configs/models/yolov8_baseline.yaml")
DEFAULT_MOBILENETV3_DESCRIPTOR = Path("configs/models/yolov8_mobilenetv3.yaml")
DEFAULT_MOBILENETV4_DESCRIPTOR = Path("configs/models/yolov8_mobilenetv4.yaml")
DEFAULT_DATA_PATH = Path("data/main_dataset/data.yaml")
DEFAULT_OUTPUT_DIR = Path("runs/eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 baseline and the current configured model on one dataset."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the runtime config used for default device and image size.",
    )
    parser.add_argument(
        "--baseline-descriptor",
        default=str(DEFAULT_BASELINE_DESCRIPTOR),
        help="Path to the baseline model descriptor YAML.",
    )
    parser.add_argument(
        "--mobilenetv3-descriptor",
        default=str(DEFAULT_MOBILENETV3_DESCRIPTOR),
        help="Path to the MobileNetV3 model descriptor YAML.",
    )
    parser.add_argument(
        "--mobilenetv4-descriptor",
        default=str(DEFAULT_MOBILENETV4_DESCRIPTOR),
        help="Path to the MobileNetV4 model descriptor YAML.",
    )
    parser.add_argument(
        "--baseline-weights",
        default=None,
        help="Override path to baseline weights.",
    )
    parser.add_argument(
        "--mobilenetv3-weights",
        default=None,
        help="Override path to MobileNetV3 weights.",
    )
    parser.add_argument(
        "--mobilenetv4-weights",
        default=None,
        help="Override path to MobileNetV4 weights.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate. Use the same split for both models.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override evaluation image size for both models.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Evaluation device, e.g. cpu, 0, 0,1. Defaults to config.detector.device.",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where Ultralytics validation runs and summary JSON will be stored.",
    )
    parser.add_argument(
        "--name",
        default="baseline_vs_current",
        help="Run name prefix for the generated evaluation artifacts.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_model_source(
    descriptor: dict[str, Any],
    override_weights: str | None,
    label: str,
) -> str:
    model_source = override_weights or descriptor.get("weights_path")
    if not model_source:
        raise SystemExit(
            f"{label} has no weights_path configured. "
            "Pass an explicit weights path via the command line."
        )
    return str(model_source)


def load_yolo(model_source: str, variant: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise SystemExit(
            "Ultralytics is not installed in the current Python environment. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    register_ultralytics_custom_backbones(variant)
    return YOLO(model_source)


def evaluate_model(
    *,
    label: str,
    model_source: str,
    variant: str,
    data_path: str,
    split: str,
    imgsz: int,
    device: str,
    project: str,
    run_name: str,
) -> dict[str, Any]:
    model = load_yolo(model_source, variant)
    metrics = model.val(
        data=data_path,
        split=split,
        imgsz=imgsz,
        device=device,
        project=project,
        name=run_name,
        verbose=False,
        plots=False,
    )
    return {
        "label": label,
        "variant": variant,
        "model_source": model_source,
        "imgsz": imgsz,
        "device": device,
        "split": split,
        "metrics": extract_metrics(metrics),
    }


def extract_metrics(metrics) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "precision": _safe_float(getattr(box, "mp", None)),
        "recall": _safe_float(getattr(box, "mr", None)),
        "mAP50": _safe_float(getattr(box, "map50", None)),
        "mAP50_95": _safe_float(getattr(box, "map", None)),
        "preprocess_ms": _safe_float(speed.get("preprocess")),
        "inference_ms": _safe_float(speed.get("inference")),
        "loss_ms": _safe_float(speed.get("loss")),
        "postprocess_ms": _safe_float(speed.get("postprocess")),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def save_summary(
    *,
    output_dir: Path,
    run_name: str,
    data_path: str,
    split: str,
    imgsz: int,
    device: str,
    results: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "split": split,
        "imgsz": imgsz,
        "device": device,
        "results": results,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def print_summary(results: list[dict[str, Any]], summary_path: Path) -> None:
    print()
    print("Evaluation summary")
    print("=" * 80)
    header = (
        f"{'model':<18}"
        f"{'precision':>12}"
        f"{'recall':>10}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
        f"{'infer ms':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['label']:<18}"
            f"{format_metric(metrics['precision']):>12}"
            f"{format_metric(metrics['recall']):>10}"
            f"{format_metric(metrics['mAP50']):>10}"
            f"{format_metric(metrics['mAP50_95']):>12}"
            f"{format_metric(metrics['inference_ms']):>11}"
        )
    print("-" * len(header))
    print(f"Saved JSON summary to: {summary_path}")


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML was not found: {data_path}")

    config = load_app_config(args.config)
    baseline_descriptor = load_yaml(args.baseline_descriptor)
    mobilenetv3_descriptor = load_yaml(args.mobilenetv3_descriptor)
    mobilenetv4_descriptor = load_yaml(args.mobilenetv4_descriptor)

    imgsz = args.imgsz or int(config.detector.image_size)
    device = args.device or str(config.detector.device)
    project = str(Path(args.project))

    models = [
        {
            "label": "baseline",
            "variant": str(baseline_descriptor.get("variant", "baseline")),
            "model_source": resolve_model_source(
                baseline_descriptor,
                args.baseline_weights,
                "Baseline model",
            ),
        },
        {
            "label": "mobilenetv3",
            "variant": str(mobilenetv3_descriptor.get("variant", "mobilenetv3")),
            "model_source": resolve_model_source(
                mobilenetv3_descriptor,
                args.mobilenetv3_weights,
                "MobileNetV3 model",
            ),
        },
        {
            "label": "mobilenetv4",
            "variant": str(mobilenetv4_descriptor.get("variant", "mobilenetv4")),
            "model_source": resolve_model_source(
                mobilenetv4_descriptor,
                args.mobilenetv4_weights,
                "MobileNetV4 model",
            ),
        },
    ]

    results = [
        evaluate_model(
            label=model["label"],
            model_source=model["model_source"],
            variant=model["variant"],
            data_path=str(data_path),
            split=args.split,
            imgsz=imgsz,
            device=device,
            project=project,
            run_name=f"{args.name}_{model['label']}",
        )
        for model in models
    ]

    summary_path = save_summary(
        output_dir=Path(project),
        run_name=args.name,
        data_path=str(data_path),
        split=args.split,
        imgsz=imgsz,
        device=device,
        results=results,
    )
    print_summary(results, summary_path)


if __name__ == "__main__":
    main()
