# Abandoned Objects Detection Project

This project is a diploma-oriented computer vision system for abandoned object detection in video streams. It is designed not only as a demo application, but also as an experimental framework for comparing `YOLOv8` with different backbones:

- `YOLOv8 + default backbone`
- `YOLOv8 + MobileNetV3 backbone`
- `YOLOv8 + MobileNetV4 backbone`

The repository includes a unified runtime pipeline, configurable detector selection, abandoned-object logic, visualization utilities, and a scaffold for backbone comparison experiments.

## Objectives

The main goals of this project are:

- detect people and personal items in a video stream;
- track objects across frames;
- determine whether an item is attended, unattended, or abandoned;
- compare the performance of `YOLOv8` with different backbone architectures under the same evaluation setup.

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
- `configs/models/` - experiment descriptors for each YOLOv8 variant.
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
  variant: default
  weights_path: yolo11n.pt
```

This means the application will try to run immediately with the local `yolo11n.pt` weights if they are available in the project root.

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
- `detector.variant` - active YOLOv8 backbone variant.
- `detector.weights_path` - path to the trained model weights.
- `logic.abandon_threshold` - time in seconds before an unattended object is considered abandoned.
- `logic.dist_threshold` - distance threshold used to associate a nearby person with an item.

## YOLOv8 Backbone Comparison

This project is structured for a fair thesis comparison of YOLOv8 with different backbones. The recommended experimental setup is:

1. Keep the detector family fixed as `YOLOv8`.
2. Change only the backbone:
   - `default`
   - `MobileNetV3`
   - `MobileNetV4`
3. Train all variants on the same dataset split.
4. Evaluate them under the same conditions.

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

Each YOLOv8 variant has a separate experiment descriptor:

- `configs/models/yolov8_default.yaml`
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

For a full thesis comparison, you should prepare separate trained weights for:

- `artifacts/weights/yolov8_default.pt`
- `artifacts/weights/yolov8_mobilenetv3.pt`
- `artifacts/weights/yolov8_mobilenetv4.pt`

If a custom thesis weight file is missing, the detector loader will raise a clear error message. During development, you can temporarily point `weights_path` to an existing local model such as `yolo11n.pt`.

## Troubleshooting

### `FileNotFoundError` for model weights

This means the path in `config.yaml` does not point to an existing `.pt` file.

Fix:

- verify that the file exists;
- update `detector.weights_path`;
- or use a local fallback model such as `yolo11n.pt`.

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

If this repository is used in an academic thesis, the most methodologically correct approach is to compare models within the same detector family and vary only the backbone. That is why this project is structured around `YOLOv8` variants rather than a mix of unrelated detector families.
