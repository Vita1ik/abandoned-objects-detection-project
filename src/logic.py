import time
import numpy as np

class AbandonedLogic:
    def __init__(self, abandon_threshold=10, dist_threshold=160, fix_confirm_threshold=0.7):
        self.abandon_threshold = abandon_threshold  # Час до тривоги
        self.dist_threshold = dist_threshold        # Радіус "власності"
        self.fix_confirm_threshold = fix_confirm_threshold # Поріг для Sticky Class
        self.objects_data = {} # База станів об'єктів

    def process_frame_logic(self, detections):
        current_time = time.time()
        
        # 1. Розділяємо об'єкти
        persons = [d for d in detections if d['class'] == 'person']
        items = [d for d in detections if d['class'] != 'person']

        for item in items:
            i_id = item['id']
            i_center = np.array(item['center'])
            i_conf = item['conf']

            print(i_conf)
            # --- ЛОГІКА STICKY CLASSES (Фіксація класу) ---
            if i_id not in self.objects_data:
                self.objects_data[i_id] = {
                    "fixed_class": item['class'],
                    "first_seen": current_time,
                    "last_seen_with_owner": current_time,
                    "status": "attended",
                    "conf": i_conf,
                    "owner_center": None
                }
            else:
                # Якщо ще не зафіксовано, пробуємо зафіксувати зараз
                if self.objects_data[i_id]["conf"] < i_conf:
                    self.objects_data[i_id]["conf"] = i_conf
                    self.objects_data[i_id]["fixed_class"] = item['class']
                
                # Якщо вже є фіксований клас — підміняємо поточний для відображення
                if self.objects_data[i_id]["fixed_class"] is not None:
                    item['class'] = self.objects_data[i_id]["fixed_class"]

            # --- ЛОГІКА ВЛАСНИКА (Відстань та лінії) ---
            found_owner_nearby = False
            nearest_dist = float('inf')
            nearest_p_center = None

            for person in persons:
                p_center = np.array(person['center'])
                dist = np.linalg.norm(i_center - p_center)
                
                if dist < self.dist_threshold:
                    found_owner_nearby = True
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_p_center = person['center']

            # --- ЛОГІКА СТАНІВ ТА ТАЙМЕРІВ ---
            if found_owner_nearby:
                self.objects_data[i_id]["last_seen_with_owner"] = current_time
                self.objects_data[i_id]["status"] = "attended"
                self.objects_data[i_id]["owner_center"] = nearest_p_center
            else:
                # Власника поруч немає
                self.objects_data[i_id]["owner_center"] = None
                time_since_owner = current_time - self.objects_data[i_id]["last_seen_with_owner"]
                
                if time_since_owner > self.abandon_threshold:
                    self.objects_data[i_id]["status"] = "abandoned"
                else:
                    self.objects_data[i_id]["status"] = "unattended"
                    # Додаємо час очікування для виводу на екран
                    self.objects_data[i_id]["wait_time"] = int(self.abandon_threshold - time_since_owner)

        return self.objects_data