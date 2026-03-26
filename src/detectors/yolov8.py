from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

from src.config import DetectorConfig
from src.detectors.base import BaseDetector
from src.domain import Detection
from src.training.custom_backbones import register_ultralytics_custom_backbones


MODEL_VARIANTS = {
    "baseline": {
        "description": "Official pretrained YOLOv8s baseline with the standard Ultralytics backbone.",
        "weights_path": "yolov8s.pt",
        "thesis_role": "balanced pretrained baseline for constrained deployment",
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

        register_ultralytics_custom_backbones(self.variant)
        self.model = YOLO(self._resolve_model_source(config))
        self.target_classes = set(config.target_classes)
        self.target_class_ids = self._resolve_target_class_ids()
        self.class_memory: dict[int, dict[str, object]] = {}
        self.max_inactive_frames = 45

    def detect_and_track(self, frame) -> list[dict]:
        results = self.model.track(
            frame,
            persist=self.config.persist_tracking,
            verbose=False,
            tracker=self.config.tracker,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            classes=self.target_class_ids,
        )

        boxes_data = results[0].boxes
        self._prune_stale_tracks()
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

            stable_class = self._stabilize_class(
                track_id=int(obj_id),
                predicted_class=class_name,
                confidence=float(conf),
            )

            detection = Detection(
                id=int(obj_id),
                box=[float(v) for v in box],
                class_name=stable_class,
                confidence=float(conf),
                center=(
                    int((box[0] + box[2]) // 2),
                    int((box[1] + box[3]) // 2),
                ),
            )
            detections.append(detection.as_dict())

        return detections

    def _resolve_model_source(self, config: DetectorConfig) -> str:
        requested_source = config.weights_path or MODEL_VARIANTS[self.variant]["weights_path"]
        weights_path = Path(requested_source)

        if weights_path.exists():
            return str(weights_path)

        # For the official YOLOv8 baseline, allow Ultralytics to resolve/download
        # standard pretrained model names even if they are not present locally yet.
        if self.variant == "baseline" and requested_source in {
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        }:
            return requested_source

        raise FileNotFoundError(
            f"Model weights for variant '{self.variant}' were not found at '{requested_source}'. "
            "Place the trained weights there or set detector.weights_path in config.yaml."
        )

    def _resolve_target_class_ids(self) -> list[int]:
        names = self.model.names
        if isinstance(names, dict):
            return [idx for idx, name in names.items() if name in self.target_classes]
        return [idx for idx, name in enumerate(names) if name in self.target_classes]

    def _stabilize_class(self, track_id: int, predicted_class: str, confidence: float) -> str:
        state = self.class_memory.setdefault(
            track_id,
            {
                "scores": defaultdict(float),
                "last_seen": 0,
                "stable_class": predicted_class,
            },
        )
        state["last_seen"] = 0
        scores = state["scores"]
        scores[predicted_class] += confidence

        best_class = max(scores, key=scores.get)
        item_classes = [name for name in scores if name != "person"]
        best_item_class = max(item_classes, key=lambda name: scores[name], default=None)

        if best_item_class is not None:
            item_score = scores[best_item_class]
            person_score = scores.get("person", 0.0)

            # For this task, once an object track gathers stronger evidence for a bag-like
            # class, avoid flipping it back to person unless the person evidence is clearly stronger.
            if item_score >= 0.6 and person_score <= item_score * 1.35:
                state["stable_class"] = best_item_class
                return best_item_class

        state["stable_class"] = best_class
        return best_class

    def _prune_stale_tracks(self) -> None:
        stale_ids: list[int] = []
        for track_id, state in self.class_memory.items():
            state["last_seen"] += 1
            if state["last_seen"] > self.max_inactive_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.class_memory[track_id]
