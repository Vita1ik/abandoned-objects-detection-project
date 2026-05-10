import time
import numpy as np


class AbandonedLogic:
    def __init__(self, abandon_threshold=10, dist_threshold=160, fix_confirm_threshold=0.7):
        self.abandon_threshold = abandon_threshold  # Час до тривоги
        self.dist_threshold = dist_threshold        # Радіус "власності"
        self.fix_confirm_threshold = fix_confirm_threshold # Поріг для Sticky Class
        self.owner_grace_period = 1.5
        self.person_box_margin = 35
        self.owner_reassign_confirmation = 2.0
        self.item_match_distance = 140
        self.item_track_ttl = 2.5
        self.bag_classes = {"backpack", "handbag", "suitcase", "bag"}
        self.objects_data = {} # База станів об'єктів
        self.next_item_id = 1

    def process_frame_logic(self, detections):
        current_time = time.time()
        self._normalize_detections_in_place(detections)

        # 1. Розділяємо об'єкти
        persons = [d for d in detections if d['class'] == 'person']
        items = [d for d in detections if d['class'] != 'person']
        self._assign_stable_item_ids(items, current_time)

        for item in items:
            i_id = item['id']
            i_center = np.array(item['center'])
            i_conf = item['conf']

            # --- ЛОГІКА STICKY CLASSES (Фіксація класу) ---
            if i_id not in self.objects_data:
                self.objects_data[i_id] = {
                    "fixed_class": item['class'],
                    "first_seen": current_time,
                    "last_seen_with_owner": current_time,
                    "status": "attended",
                    "conf": i_conf,
                    "owner_center": None,
                    "owner_id": None,
                    "candidate_owner_id": None,
                    "candidate_owner_since": None,
                    "last_box": item["box"],
                    "last_center": item["center"],
                    "last_seen": current_time,
                }
            else:
                # Якщо ще не зафіксовано, пробуємо зафіксувати зараз
                if self.objects_data[i_id]["conf"] < i_conf:
                    self.objects_data[i_id]["conf"] = i_conf
                    self.objects_data[i_id]["fixed_class"] = item['class']

                # Якщо вже є фіксований клас — підміняємо поточний для відображення
                if self.objects_data[i_id]["fixed_class"] is not None:
                    item['class'] = self.objects_data[i_id]["fixed_class"]

                self.objects_data[i_id]["last_box"] = item["box"]
                self.objects_data[i_id]["last_center"] = item["center"]
                self.objects_data[i_id]["last_seen"] = current_time

            # --- ЛОГІКА ВЛАСНИКА (Відстань та лінії) ---
            found_owner_nearby = False
            nearest_dist = float('inf')
            nearest_p_center = None
            nearest_owner_id = None

            for person in persons:
                dist = self._person_item_distance(person, item)
                if self._is_owner_candidate(person, item, dist):
                    found_owner_nearby = True
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_p_center = person['center']
                        nearest_owner_id = person["id"]

            # --- ЛОГІКА СТАНІВ ТА ТАЙМЕРІВ ---
            confirmed_owner_id = self._resolve_owner_for_item(
                item_state=self.objects_data[i_id],
                found_owner_nearby=found_owner_nearby,
                nearest_owner_id=nearest_owner_id,
                current_time=current_time,
            )

            if confirmed_owner_id is not None and nearest_p_center is not None:
                self.objects_data[i_id]["last_seen_with_owner"] = current_time
                self.objects_data[i_id]["status"] = "attended"
                self.objects_data[i_id]["owner_center"] = nearest_p_center
                self.objects_data[i_id]["owner_id"] = confirmed_owner_id
                self.objects_data[i_id]["wait_time"] = 0
            else:
                # Власника поруч немає
                self.objects_data[i_id]["owner_center"] = None
                time_since_owner = current_time - self.objects_data[i_id]["last_seen_with_owner"]

                if time_since_owner <= self.owner_grace_period:
                    self.objects_data[i_id]["status"] = "attended"
                    self.objects_data[i_id]["wait_time"] = 0
                elif time_since_owner > self.abandon_threshold:
                    self.objects_data[i_id]["status"] = "abandoned"
                    self.objects_data[i_id]["wait_time"] = 0
                else:
                    self.objects_data[i_id]["status"] = "unattended"
                    # Додаємо час очікування для виводу на екран
                    self.objects_data[i_id]["wait_time"] = int(self.abandon_threshold - time_since_owner)

        return self.objects_data

    def _normalize_detections_in_place(self, detections):
        for detection in detections:
            if detection["class"] in self.bag_classes:
                detection["class"] = "bag"

    def _assign_stable_item_ids(self, items, current_time):
        assigned_existing_ids = set()

        for item in items:
            matched_id = self._match_existing_item(item, current_time, assigned_existing_ids)
            if matched_id is None:
                matched_id = self.next_item_id
                self.next_item_id += 1

            item["id"] = matched_id
            assigned_existing_ids.add(matched_id)

    def _match_existing_item(self, item, current_time, assigned_existing_ids):
        best_match_id = None
        best_score = float("inf")

        for object_id, object_state in self.objects_data.items():
            if object_id in assigned_existing_ids:
                continue
            if current_time - object_state.get("last_seen", 0) > self.item_track_ttl:
                continue

            stored_center = np.array(object_state.get("last_center", item["center"]))
            current_center = np.array(item["center"])
            center_dist = float(np.linalg.norm(current_center - stored_center))
            iou_penalty = 1.0 - self._box_iou(item["box"], object_state.get("last_box", item["box"]))

            if center_dist > self.item_match_distance and iou_penalty > 0.85:
                continue

            score = center_dist + 80.0 * iou_penalty
            if score < best_score:
                best_score = score
                best_match_id = object_id

        return best_match_id

    def _resolve_owner_for_item(self, item_state, found_owner_nearby, nearest_owner_id, current_time):
        if not found_owner_nearby or nearest_owner_id is None:
            item_state["candidate_owner_id"] = None
            item_state["candidate_owner_since"] = None
            return None

        current_owner_id = item_state.get("owner_id")
        time_since_owner = current_time - item_state.get("last_seen_with_owner", current_time)

        # Do not let a passerby become the new owner once the item has already
        # been left unattended. This preserves the abandoned-state timer instead
        # of resetting it whenever another person walks close to the item.
        if (
            current_owner_id is not None
            and current_owner_id != nearest_owner_id
            and time_since_owner > self.owner_grace_period
        ):
            item_state["candidate_owner_id"] = None
            item_state["candidate_owner_since"] = None
            return None

        if current_owner_id is None or current_owner_id == nearest_owner_id:
            item_state["candidate_owner_id"] = None
            item_state["candidate_owner_since"] = None
            return nearest_owner_id

        candidate_owner_id = item_state.get("candidate_owner_id")
        if candidate_owner_id != nearest_owner_id:
            item_state["candidate_owner_id"] = nearest_owner_id
            item_state["candidate_owner_since"] = current_time
            return None

        candidate_owner_since = item_state.get("candidate_owner_since")
        if candidate_owner_since is None:
            item_state["candidate_owner_since"] = current_time
            return None

        if current_time - candidate_owner_since >= self.owner_reassign_confirmation:
            item_state["candidate_owner_id"] = None
            item_state["candidate_owner_since"] = None
            return nearest_owner_id

        return None

    def _is_owner_candidate(self, person, item, dist):
        if self._item_center_inside_expanded_person_box(person, item):
            return True
        return dist < self.dist_threshold

    def _item_center_inside_expanded_person_box(self, person, item):
        px1, py1, px2, py2 = person["box"]
        ix, iy = item["center"]

        px1 -= self.person_box_margin
        py1 -= self.person_box_margin
        px2 += self.person_box_margin
        py2 += self.person_box_margin

        return px1 <= ix <= px2 and py1 <= iy <= py2

    def _person_item_distance(self, person, item):
        p_center = np.array(person['center'])
        i_center = np.array(item['center'])
        center_dist = np.linalg.norm(i_center - p_center)

        px1, py1, px2, py2 = person["box"]
        ix1, iy1, ix2, iy2 = item["box"]

        dx = max(px1 - ix2, 0, ix1 - px2)
        dy = max(py1 - iy2, 0, iy1 - py2)
        edge_dist = float(np.hypot(dx, dy))

        return min(center_dist, edge_dist)

    def _box_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union
