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


def draw_performance_stats(frame, stats: dict[str, float]) -> None:
    lines = [
        f"Display FPS: {stats['display_fps']:.1f}",
        f"Display: {stats['display_ms']:.1f} ms",
        f"Loop: {stats['loop_ms']:.1f} ms",
        f"Detect: {stats['detect_ms']:.1f} ms ({stats['detect_fps']:.1f} FPS)",
    ]

    x = 10
    y = 24
    line_height = 22
    width = 320
    height = 12 + line_height * len(lines)

    cv2.rectangle(frame, (x - 6, y - 18), (x + width, y - 18 + height), (30, 30, 30), -1)
    cv2.rectangle(frame, (x - 6, y - 18), (x + width, y - 18 + height), (120, 120, 120), 1)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
