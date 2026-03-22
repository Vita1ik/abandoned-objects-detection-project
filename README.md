# Abandoned Objects Detection Project

This project is a diploma-oriented computer vision system for abandoned object detection in video streams. It is designed not only as a demo application, but also as an experimental framework for comparing `YOLOv8` with different backbone choices for resource-constrained devices.

The current thesis logic is based on three model groups:

- `YOLOv8s pretrained baseline`
- `YOLOv8 + MobileNetV3`
- `YOLOv8 + MobileNetV4`

The repository includes a unified runtime pipeline, configurable detector selection, abandoned-object logic, visualization utilities, and a scaffold for backbone comparison experiments.

## Objectives

The main goals of this project are:

- detect people and personal items in a video stream;
- track objects across frames;
- determine whether an item is attended, unattended, or abandoned;
- compare lightweight `YOLOv8`-based models for deployment on devices with limited computational resources.

## Project Structure

```text
.
├── main.py
├── config.yaml
├── configs/
│   └── models/
├── experiments/
│   └── compare_backbones.py
├── models/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── domain.py
│   ├── detectors/
│   ├── logic.py
│   └── visualization/
├── tests/
└── README.md
```

### Key components

- `main.py` - application entry point.
- `config.yaml` - runtime configuration for inference.
- `configs/models/` - experiment descriptors for each YOLOv8-based thesis model.
- `src/app.py` - main video-processing loop.
- `src/config.py` - typed application configuration loader.
- `src/detectors/` - detector interface, factory, and YOLOv8 implementation.
- `src/logic.py` - state machine for attended, unattended, and abandoned objects.
- `src/visualization/` - drawing and annotation helpers.
- `experiments/compare_backbones.py` - scaffold for checking experiment readiness and generating a comparison CSV.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `PyYAML` is missing during launch, install it explicitly:

```bash
pip install pyyaml
```

## Running the Application

Start the application with:

```bash
python main.py
```

By default, the project reads settings from `config.yaml`.

Current default runtime configuration:

```yaml
detector:
  family: yolov8
  variant: baseline
  weights_path: yolov8s.pt
```

This means the application uses `yolov8s.pt` as the default pretrained baseline, giving better detection quality while still remaining practical for constrained deployment scenarios.

## Configuration

The main runtime settings are stored in `config.yaml`.

Example:

```yaml
video_source: 0
window_title: "Abandoned Object System - MSc Thesis Project"

detector:
  family: yolov8
  variant: mobilenetv3
  weights_path: artifacts/weights/yolov8_mobilenetv3.pt
  confidence: 0.25
  iou: 0.45
  image_size: 640
  tracker: bytetrack.yaml
  persist_tracking: true
  device: cpu

logic:
  abandon_threshold: 5
  dist_threshold: 160
  fix_confirm_threshold: 0.7
```

### Important parameters

- `video_source` - webcam index or path to a video file.
- `detector.variant` - active model configuration: `baseline`, `mobilenetv3`, or `mobilenetv4`.
- `detector.weights_path` - path to the selected model weights.
- `detector.image_size` - inference image size; larger values usually improve small-object detection but reduce FPS.
- `logic.abandon_threshold` - time in seconds before an unattended object is considered abandoned.
- `logic.dist_threshold` - distance threshold used to associate a nearby person with an item.

## Thesis Model Setup

This project is structured for a fair thesis comparison of lightweight `YOLOv8`-based models.

The recommended setup is:

1. Use `YOLOv8s pretrained` as the official baseline model.
2. Implement and train `YOLOv8 + MobileNetV3`.
3. Implement and train `YOLOv8 + MobileNetV4`.
4. Evaluate all models on the same dataset split and under the same runtime conditions.

This setup is methodologically appropriate because:

- all models remain in the same detector family;
- the baseline is a real official pretrained model from Ultralytics;
- the comparison is aligned with deployment on low-resource devices.

### Recommended evaluation metrics

- `mAP@0.5`
- `mAP@0.5:0.95`
- precision
- recall
- F1-score
- FPS
- latency per frame
- model size
- parameter count
- memory usage

For the abandoned-object task, it is also useful to report:

- abandoned event detection accuracy;
- false alarm rate;
- average time to a correct alarm;
- missed abandoned-object events.

## Experiment Files

Each thesis model has a separate experiment descriptor:

- `configs/models/yolov8_baseline.yaml`
- `configs/models/yolov8_mobilenetv3.yaml`
- `configs/models/yolov8_mobilenetv4.yaml`

These files describe which model variant is used and where its weights are expected to be stored.

## Running the Comparison Scaffold

You can run the experiment scaffold with:

```bash
python experiments/compare_backbones.py
```

The script generates:

```text
logs/experiments/backbone_comparison.csv
```

At the moment, this script validates whether each model configuration is ready to run and records a simple status report. It is intended as a foundation for a more complete evaluation pipeline.

## Weights

The current baseline model is:

- `yolov8s.pt` - official pretrained YOLOv8s weights

For the full thesis comparison, you should additionally prepare custom trained weights for:

- `artifacts/weights/yolov8_mobilenetv3.pt`
- `artifacts/weights/yolov8_mobilenetv4.pt`

If a custom thesis weight file is missing, the detector loader will raise a clear error message.

## Troubleshooting

### `FileNotFoundError` for model weights

This means the path in `config.yaml` does not point to an existing `.pt` file.

Fix:

- verify that the file exists;
- update `detector.weights_path`;
- check that the chosen variant matches the selected weight file.

### `ModuleNotFoundError: No module named 'yaml'`

Install `PyYAML`:

```bash
pip install pyyaml
```

### Camera does not open

Try changing:

```yaml
video_source: 0
```

to another device index such as `1`, or use a video file path instead.

## Future Improvements

- add a full training pipeline for custom YOLOv8 backbone variants;
- add automated benchmark scripts for latency and FPS;
- add quantitative evaluation on a labeled validation dataset;
- add unit tests for detection contracts and abandoned-object logic;
- add result visualization and export for thesis tables and charts.

## Thesis Note

This repository intentionally uses `YOLOv8` as the research base, because it remains highly relevant, is well supported in practice, and is suitable for controlled backbone-level experimentation. The `YOLOv8s` pretrained model is used as the official baseline because it offers a better accuracy-to-efficiency balance for this project.
