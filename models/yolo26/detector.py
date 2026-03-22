import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        # Завантажуємо модель (YOLO11 або YOLO26)
        self.model = YOLO("yolo26n.pt")
        # Класи, які нас цікавлять згідно з техзавданням
        self.target_classes = ['person', 'backpack', 'handbag', 'suitcase']

    def detect_and_track(self, frame):
        # Використовуємо вбудований трекер (ByteTrack за замовчуванням)
        results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=0.1, iou=0.5)
        
        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, obj_id, cls_idx, conf in zip(boxes, ids, classes, confidences):
                class_name = self.model.names[cls_idx]
                if class_name in self.target_classes:
                    # ПРИМУСОВА КОНВЕРТАЦІЯ ТИПУ
                    clean_id = int(obj_id) 
                    
                    detections.append({
                        "id": clean_id, 
                        "box": box,
                        "class": class_name,
                        "conf": conf,
                        "center": (int((box[0]+box[2])//2), int((box[1]+box[3])//2))
                    })
        return detections