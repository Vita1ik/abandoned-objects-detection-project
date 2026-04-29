from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import build_model, load_model_descriptor, resolve_model_source


DEFAULT_METRIC = "metrics/mAP50-95(B)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize YOLO training hyperparameters with Optuna."
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/yolov8_mobilenetv3_lite.yaml",
        help="Path to the experiment descriptor YAML.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the Ultralytics data.yaml dataset config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs per trial.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use per trial.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device, e.g. cpu, mps, 0, 0,1.",
    )
    parser.add_argument(
        "--project",
        default="runs/optuna",
        help="Directory for Optuna trial runs.",
    )
    parser.add_argument(
        "--study-name",
        default="yolo_hpo",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///runs/optuna/study.db",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric column from Ultralytics results.csv to maximize.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampling.",
    )
    parser.add_argument(
        "--imgsz-choices",
        nargs="+",
        type=int,
        default=[512],
        help="Candidate image sizes.",
    )
    parser.add_argument(
        "--batch-choices",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Candidate batch sizes.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum detections per image during validation to reduce NMS overhead.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold used during validation/inference.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold used during validation/inference.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers per trial.",
    )
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation during training.",
    )
    return parser.parse_args()


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in the current environment. "
            "Install dependencies first with `pip install -r requirements.txt`."
        ) from exc
    return optuna


def infer_search_profile(model_config_path: str | Path) -> str:
    stem = Path(model_config_path).stem.lower()
    if "mobilenetv4_lite" in stem:
        return "mobilenetv4_lite"
    if "mobilenetv3_lite" in stem:
        return "mobilenetv3_lite"
    return "default"


def suggest_hyperparameters(trial, args: argparse.Namespace, profile: str) -> dict[str, Any]:
    common = {
        "imgsz": trial.suggest_categorical("imgsz", args.imgsz_choices),
        "batch": trial.suggest_categorical("batch", args.batch_choices),
        "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.015),
        "hsv_s": trial.suggest_float("hsv_s", 0.15, 0.5),
        "hsv_v": trial.suggest_float("hsv_v", 0.2, 0.65),
        "degrees": trial.suggest_float("degrees", 0.0, 6.0),
        "translate": trial.suggest_float("translate", 0.02, 0.12),
        "scale": trial.suggest_float("scale", 0.2, 0.6),
        "fliplr": trial.suggest_float("fliplr", 0.0, 0.2),
        "mosaic": trial.suggest_float("mosaic", 0.1, 0.6),
        "mixup": trial.suggest_float("mixup", 0.0, 0.15),
        "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.2),
    }

    if profile == "mobilenetv3_lite":
        return {
            **common,
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "Adam"]),
            "lr0": trial.suggest_float("lr0", 1e-3, 4e-3, log=True),
            "lrf": trial.suggest_float("lrf", 0.05, 0.2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 3e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.84, 0.92),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 2.0, 5.0),
            "box": trial.suggest_float("box", 7.5, 10.0),
            "cls": trial.suggest_float("cls", 0.25, 0.55),
            "dfl": trial.suggest_float("dfl", 1.1, 1.6),
        }

    if profile == "mobilenetv4_lite":
        return {
            **common,
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "Adam"]),
            "lr0": trial.suggest_float("lr0", 7e-4, 3e-3, log=True),
            "lrf": trial.suggest_float("lrf", 0.04, 0.18, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 3e-4, 2e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.85, 0.93),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 1.0, 4.0),
            "box": trial.suggest_float("box", 7.0, 10.5),
            "cls": trial.suggest_float("cls", 0.2, 0.5),
            "dfl": trial.suggest_float("dfl", 1.0, 1.7),
        }

    return {
        **common,
        "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "Adam"]),
        "lr0": trial.suggest_float("lr0", 5e-4, 4e-3, log=True),
        "lrf": trial.suggest_float("lrf", 0.03, 0.2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 3e-4, 3e-3, log=True),
        "momentum": trial.suggest_float("momentum", 0.84, 0.94),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 1.0, 5.0),
        "box": trial.suggest_float("box", 7.0, 10.5),
        "cls": trial.suggest_float("cls", 0.2, 0.6),
        "dfl": trial.suggest_float("dfl", 1.0, 1.8),
    }


def read_best_metric(results_csv: Path, metric_name: str) -> float:
    if not results_csv.exists():
        raise FileNotFoundError(f"Ultralytics results file was not found at '{results_csv}'.")

    lines = [line.strip() for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Ultralytics results file '{results_csv}' does not contain any training rows.")

    headers = [column.strip() for column in lines[0].split(",")]
    if metric_name not in headers:
        available = ", ".join(headers)
        raise ValueError(f"Metric '{metric_name}' was not found in results.csv. Available columns: {available}")

    metric_index = headers.index(metric_name)
    best_value = None
    for row in lines[1:]:
        values = [value.strip() for value in row.split(",")]
        if metric_index >= len(values):
            continue
        current = float(values[metric_index])
        best_value = current if best_value is None else max(best_value, current)

    if best_value is None:
        raise ValueError(f"Metric '{metric_name}' could not be read from '{results_csv}'.")

    return best_value


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    optuna = load_optuna()

    descriptor = load_model_descriptor(args.model_config)
    model_source = resolve_model_source(descriptor)
    search_profile = infer_search_profile(args.model_config)

    project_dir = Path(args.project).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        model = None
        try:
            model = build_model(model_source, descriptor)

            trial_hparams = suggest_hyperparameters(trial, args, search_profile)
            run_name = f"{args.study_name}_trial_{trial.number:03d}"

            train_kwargs = {
                "data": args.data,
                "epochs": args.epochs,
                "fraction": args.fraction,
                "device": args.device,
                "project": str(project_dir),
                "name": run_name,
                "val": args.val,
                "workers": args.workers,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "plots": False,
                "save": True,
                "verbose": False,
                **trial_hparams,
            }

            bootstrap_weights = descriptor.get("bootstrap_weights")
            if bootstrap_weights:
                model = model.load(bootstrap_weights)

            model.train(**train_kwargs)

            save_dir = None
            trainer = getattr(model, "trainer", None)
            if trainer is not None:
                save_dir = getattr(trainer, "save_dir", None)
            if save_dir is None:
                save_dir = project_dir / run_name
            else:
                save_dir = Path(save_dir)

            results_csv = save_dir / "results.csv"
            score = read_best_metric(results_csv, args.metric)
            trial.set_user_attr("run_name", run_name)
            trial.set_user_attr("save_dir", str(save_dir))
            trial.set_user_attr("results_csv", str(results_csv))
            return score
        finally:
            if model is not None:
                del model
            gc.collect()
            try:
                import torch

                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    study.optimize(objective, n_trials=args.trials)

    summary = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_run_name": study.best_trial.user_attrs.get("run_name"),
        "search_profile": search_profile,
        "model_config": args.model_config,
        "data": args.data,
        "epochs_per_trial": args.epochs,
        "trials": args.trials,
    }

    summary_yaml = project_dir / f"{args.study_name}_best.yaml"
    summary_json = project_dir / f"{args.study_name}_best.json"
    ensure_parent_dir(summary_yaml)
    summary_yaml.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nOptuna study finished.")
    print(f"Best {args.metric}: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Summary YAML: {summary_yaml}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
