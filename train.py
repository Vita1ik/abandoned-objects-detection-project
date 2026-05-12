from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.training.custom_backbones import register_ultralytics_custom_backbones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or validate a YOLOv8 + MobileNetV3 experiment scaffold."
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/yolov8_mobilenetv3_lite.yaml",
        help="Path to the experiment descriptor YAML.",
    )
    parser.add_argument(
        "--data",
        help="Path to the Ultralytics data.yaml dataset config.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes used for loading training data.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use, e.g. 0.1 for 10%% of the data.",
    )
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--project", default="runs/train", help="Ultralytics project directory.")
    parser.add_argument("--name", default=None, help="Optional Ultralytics run name.")
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation during training. Disable for faster smoke tests on large datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the model and print a short summary without training.",
    )
    return parser.parse_args()


def load_model_descriptor(path: str | Path) -> dict:
    descriptor_path = Path(path)
    with descriptor_path.open("r", encoding="utf-8") as fh:
        descriptor = yaml.safe_load(fh) or {}
    descriptor["_path"] = str(descriptor_path)
    return descriptor


def resolve_model_source(descriptor: dict) -> str:
    architecture_path = descriptor.get("architecture_path")
    bootstrap_weights = descriptor.get("bootstrap_weights")
    weights_path = descriptor.get("weights_path")

    if architecture_path:
        return architecture_path
    if bootstrap_weights:
        return bootstrap_weights
    if weights_path:
        return weights_path
    raise ValueError("Model descriptor must define architecture_path, bootstrap_weights, or weights_path.")


def build_model(model_source: str, descriptor: dict):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise SystemExit(
            "Ultralytics is not installed in the current Python environment. "
            "Install project dependencies first with `pip install -r requirements.txt`."
        ) from exc

    register_ultralytics_custom_backbones(descriptor.get("variant"))
    return YOLO(model_source)


def main() -> None:
    args = parse_args()
    descriptor = load_model_descriptor(args.model_config)
    model_source = resolve_model_source(descriptor)
    model = build_model(model_source, descriptor)

    if args.dry_run:
        print("Dry-run model initialization succeeded.")
        print(f"Experiment descriptor: {descriptor['_path']}")
        print(f"Model source: {model_source}")
        print(f"Variant: {descriptor.get('variant', 'unknown')}")
        print(
            "Next step: run training with a valid --data path once the architecture "
            "build has been verified."
        )
        return

    if not args.data:
        raise SystemExit("Training requires --data path to a valid Ultralytics data.yaml file.")

    imgsz = args.imgsz or int(descriptor.get("default_imgsz", 640))
    batch = args.batch or int(descriptor.get("default_batch", 8))
    run_name = args.name or descriptor.get("name", "mobilenetv3_experiment")
    train_overrides = dict(descriptor.get("train_overrides", {}))

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": args.workers,
        "fraction": args.fraction,
        "device": args.device,
        "project": args.project,
        "name": run_name,
        "val": args.val,
    }
    train_kwargs.update(train_overrides)
    train_kwargs["data"] = args.data
    train_kwargs["epochs"] = args.epochs
    train_kwargs["imgsz"] = imgsz
    train_kwargs["batch"] = batch
    train_kwargs["workers"] = args.workers
    train_kwargs["fraction"] = args.fraction
    train_kwargs["device"] = args.device
    train_kwargs["project"] = args.project
    train_kwargs["name"] = run_name
    train_kwargs["val"] = args.val

    bootstrap_weights = descriptor.get("bootstrap_weights")
    if bootstrap_weights:
        model = model.load(bootstrap_weights)

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
