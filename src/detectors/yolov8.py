from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.config import DetectorConfig
from src.detectors.base import BaseDetector
from src.domain import Detection


MODEL_VARIANTS = {
    "default": {
        "description": "Baseline YOLOv8 model with the standard Ultralytics backbone.",
        "weights_path": "artifacts/weights/yolov8_default.pt",
        "thesis_role": "reference baseline",
    },
    "mobilenetv3": {
        "description": "YOLOv8 detector with a MobileNetV3 backbone trained separately for comparison.",
        "weights_path": "artifacts/weights/yolov8_mobilenetv3.pt",
        "thesis_role": "lightweight comparison model",
    },
    "mobilenetv4": {
        "description": "YOLOv8 detector with a MobileNetV4 backbone trained separately for comparison.",
        "weights_path": "artifacts/weights/yolov8_mobilenetv4.pt",
        "thesis_role": "latest mobile backbone comparison model",
    },
}


class YoloV8Detector(BaseDetector):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.variant = config.variant.lower()
        if self.variant not in MODEL_VARIANTS:
            supported = ", ".join(sorted(MODEL_VARIANTS))
            raise ValueError(
                f"Unsupported YOLOv8 variant '{config.variant}'. Supported variants: {supported}"
            )

        weights_path = Path(config.weights_path or MODEL_VARIANTS[self.variant]["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights for variant '{self.variant}' were not found at '{weights_path}'. "
                "Place the trained weights there or set detector.weights_path in config.yaml."
            )

        self.model = YOLO(str(weights_path))
        self.target_classes = set(config.target_classes)

    def detect_and_track(self, frame) -> list[dict]:
        results = self.model.track(
            frame,
            persist=self.config.persist_tracking,
            verbose=False,
            tracker=self.config.tracker,
            conf=self.config.confidence,
            iou=self.config.iou,
        )

        boxes_data = results[0].boxes
        if boxes_data.id is None:
            return []

        boxes = boxes_data.xyxy.cpu().numpy()
        ids = boxes_data.id.cpu().numpy().astype(int).tolist()
        classes = boxes_data.cls.cpu().numpy().astype(int)
        confidences = boxes_data.conf.cpu().numpy()

        detections: list[dict] = []
        for box, obj_id, cls_idx, conf in zip(boxes, ids, classes, confidences):
            class_name = self.model.names[cls_idx]
            if class_name not in self.target_classes:
                continue

            detection = Detection(
                id=int(obj_id),
                box=[float(v) for v in box],
                class_name=class_name,
                confidence=float(conf),
                center=(
                    int((box[0] + box[2]) // 2),
                    int((box[1] + box[3]) // 2),
                ),
            )
            detections.append(detection.as_dict())

        return detections
