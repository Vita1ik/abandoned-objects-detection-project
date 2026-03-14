import cv2
from src.detector import ObjectDetector
from src.logic import AbandonedLogic

def main():
    cap = cv2.VideoCapture(0) # або 0 для камери
    detector = ObjectDetector("yolo26n.pt")
    logic = AbandonedLogic(abandon_threshold=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Детекція
        detections = detector.detect_and_track(frame)

        # 2. Аналіз логіки
        obj_states = logic.process_frame_logic(detections)

        # 3. Візуалізація
        for d in detections:
            obj_id = d['id']
            
            # Використовуємо .get(), щоб не було KeyError, якщо ID новий
            obj_info = obj_states.get(obj_id)
            
            if obj_info:
                status = obj_info['status']
                
                # Вибір кольору залежно від статусу
                if status == "abandoned":
                    color = (0, 0, 255) # Червоний
                elif status == "unattended":
                    color = (0, 255, 255) # Жовтий
                else:
                    color = (0, 255, 0) # Зелений
                    
                x1, y1, x2, y2 = map(int, d['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{d['class']} ID:{obj_id} [{status.upper()}]"
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Abandoned Object System - MSc Thesis Project", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()