from __future__ import annotations

import cv2


def draw_detections_with_logic(frame, detections, obj_states) -> None:
    for detection in detections:
        obj_id = detection["id"]
        obj_info = obj_states.get(obj_id, {})
        status = obj_info.get("status", "unknown")
        color = _pick_color(status)

        x1, y1, x2, y2 = map(int, detection["box"])
        label = _build_label(detection, status, obj_info)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def _pick_color(status: str) -> tuple[int, int, int]:
    if status == "abandoned":
        return (0, 0, 255)
    if status == "unattended":
        return (0, 255, 255)
    return (0, 255, 0)


def _build_label(detection: dict, status: str, obj_info: dict) -> str:
    wait_time = obj_info.get("wait_time")
    parts = [
        detection["class"],
        f"ID:{detection['id']}",
        f"{detection['conf']:.2f}",
        status.upper(),
    ]
    if wait_time is not None and status == "unattended":
        parts.append(f"{wait_time}s")
    return " | ".join(parts)
