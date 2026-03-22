import cv2
import os
import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_model_weight, load_config
from nanodet.data.transform import Pipeline

class ObjectDetector:
    def __init__(self, target_classes=None, conf_thresh=0.3):
        """
        target_classes — список класів, які нас цікавлять
        conf_thresh — поріг впевненості
        """
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, "nanodet-plus-m_416.pth")
        config_path = os.path.join(base_path, "nanodet-plus-m_416.yml")
        
        # Завантажуємо конфігурацію
        load_config(cfg, config_path)
        self.conf_thresh = conf_thresh
        self.target_classes = target_classes

        # Створюємо модель та вантажимо ваги
        self.model = build_model(cfg.model)
        load_model_weight(self.model, model_path, use_ema=True)
        self.model.eval()

        # Трансформації
        self.pipeline = Pipeline(cfg.data.test.pipeline, cfg.data.test.input_size)

        # Імена класів
        self.names = cfg.class_names

    def detect_and_track(self, frame):
        h, w, _ = frame.shape
        data = {"img": frame, "img_info": {"height": h, "width": w}}
        data = self.pipeline(data)
        data = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in data.items()}  # batch dimension

        with torch.no_grad():
            result = self.model.inference(data)  # повертає dict з 'det_bboxes' і 'det_scores'

        detections = []
        det_bboxes = result['det_bboxes'].cpu().numpy()
        det_scores = result['det_scores'].cpu().numpy()

        for bbox, score_list in zip(det_bboxes, det_scores):
            for class_id, score in enumerate(score_list):
                if score < self.conf_thresh:
                    continue

                class_name = self.names[class_id]
                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = bbox
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": class_name,
                    "conf": float(score),
                    "center": (int((x1 + x2) // 2), int((y1 + y2) // 2))
                })
        return detections