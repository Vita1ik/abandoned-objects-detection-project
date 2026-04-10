from __future__ import annotations

from ultralytics import YOLO

from src.config import DetectorConfig
from src.detectors.base import BaseDetector
from src.domain import Detection


class Yolo26Detector(BaseDetector):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = YOLO(config.weights_path or "yolo26n.pt")
        self.target_classes = set(config.target_classes)
        self.target_class_ids = self._resolve_target_class_ids()

    def detect_and_track(self, frame) -> list[dict]:
        results = self.model.track(
            frame,
            persist=self.config.persist_tracking,
            verbose=False,
            tracker=self.config.tracker,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.config.device,
            classes=self.target_class_ids,
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

    def _resolve_target_class_ids(self) -> list[int]:
        names = self.model.names
        if isinstance(names, dict):
            return [idx for idx, name in names.items() if name in self.target_classes]
        return [idx for idx, name in enumerate(names) if name in self.target_classes]
