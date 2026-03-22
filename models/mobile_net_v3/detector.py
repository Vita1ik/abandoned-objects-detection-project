# import cv2
# import numpy as np
# import os

# class ObjectDetector:
#     def __init__(self):
#         # Завантаження моделі SSD MobileNetV3
#         base_path = os.path.dirname(__file__)  # шлях до поточного файлу detector.py

#         weights_path = os.path.join(base_path, "frozen_inference_graph.pb")
#         config_path = os.path.join(base_path, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

#         print("Weights exists:", os.path.exists(weights_path))
#         print("Config exists:", os.path.exists(config_path))

#         self.net = cv2.dnn_DetectionModel(weights_path, config_path)

#         self.net.setInputSize(320, 320)
#         self.net.setInputScale(1.0 / 127.5)
#         self.net.setInputMean((127.5, 127.5, 127.5))
#         self.net.setInputSwapRB(True)

#         # класи COCO
#         self.classes = {
#             1: "person",
#             27: "backpack",
#             31: "handbag",
#             33: "suitcase"
#         }

#         self.target_classes = ['person', 'backpack', 'handbag', 'suitcase']

#         # simple tracker id
#         self.next_id = 0
#         self.objects = {}

#     def detect_and_track(self, frame):

#         class_ids, confidences, boxes = self.net.detect(
#             frame,
#             confThreshold=0.3,
#             nmsThreshold=0.4
#         )

#         detections = []

#         if len(class_ids) > 0:

#             for class_id, conf, box in zip(class_ids.flatten(), confidences.flatten(), boxes):

#                 if class_id in self.classes:

#                     class_name = self.classes[class_id]

#                     if class_name in self.target_classes:

#                         x, y, w, h = box
#                         center = (int(x + w/2), int(y + h/2))

#                         detections.append({
#                             "id": self.next_id,
#                             "box": [x, y, x+w, y+h],
#                             "class": class_name,
#                             "conf": float(conf),
#                             "center": center
#                         })

#                         self.next_id += 1

#         return 


import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetector:
    def __init__(self, device='cpu'):
        # SSD MobileNetV3 pretrained на COCO
        self.device = torch.device(device)
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # COCO класи, що нас цікавлять
        self.classes = {1: "person", 27: "backpack", 31: "handbag", 33: "suitcase"}
        self.target_classes = set(self.classes.values())

        # DeepSORT трекер
        self.tracker = DeepSort(max_age=30)  # max_age = кількість кадрів для збереження id

    def detect_and_track(self, frame):
        """
        frame: OpenCV BGR кадр
        Повертає detections з bbox, class, conf і стабільним id
        """

        # --- Підготовка кадру ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(self.device)

        # --- Детекція ---
        with torch.no_grad():
            outputs = self.model(img_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()      # [x1, y1, x2, y2]
        labels = outputs['labels'].cpu().numpy()    # COCO class ids
        scores = outputs['scores'].cpu().numpy()    # confidence

        # --- Фільтрація класів ---
        dets_for_tracker = []
        for box, label, score in zip(boxes, labels, scores):
            class_name = self.classes.get(label)
            if class_name is None:
                continue
            if class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = box
            dets_for_tracker.append([x1, y1, x2, y2, float(score), class_name])

        if len(dets_for_tracker) == 0:
            dets_for_tracker = np.empty((0, 6))
        else:
            dets_for_tracker = np.array(dets_for_tracker, dtype=object)

        # --- Трекинг через DeepSORT ---
        online_targets = self.tracker.update_tracks(dets_for_tracker, frame=frame)

        detections = []
        for t in online_targets:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            track_id = t.track_id
            conf = t.det_conf
            class_name = t.get_det_class() or "object"
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            detections.append({
                "id": track_id,
                "box": [x1, y1, x2, y2],
                "class": class_name,
                "conf": float(conf),
                "center": (cx, cy)
            })

        return detections