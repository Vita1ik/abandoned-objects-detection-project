from __future__ import annotations

from collections import deque
from time import perf_counter

import cv2
import numpy as np

from src.config import AppConfig, load_app_config
from src.detectors.factory import build_detector
from src.logic import AbandonedLogic
from src.visualization.draw import draw_detections_with_logic, draw_performance_stats


def run_application(config_path: str = "config.yaml") -> None:
    config = load_app_config(config_path)
    _run_with_config(config)


def _run_with_config(config: AppConfig) -> None:
    cap = cv2.VideoCapture(config.video_source)
    detector = build_detector(config.detector)
    logic = AbandonedLogic(
        abandon_threshold=config.logic.abandon_threshold,
        dist_threshold=config.logic.dist_threshold,
        fix_confirm_threshold=config.logic.fix_confirm_threshold,
    )
    frame_index = 0
    last_detections = []
    frame_times_ms: deque[float] = deque(maxlen=config.runtime.performance_window_size)
    detect_times_ms: deque[float] = deque(maxlen=config.runtime.performance_window_size)

    while cap.isOpened():
        frame_start = perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = _preprocess_frame(frame, config)
        should_detect = frame_index % config.runtime.detect_every_n_frames == 0
        detect_time_ms = 0.0
        if should_detect:
            detect_start = perf_counter()
            last_detections = detector.detect_and_track(processed_frame)
            detect_time_ms = (perf_counter() - detect_start) * 1000.0
            detect_times_ms.append(detect_time_ms)

        detections = [dict(detection) for detection in last_detections]
        obj_states = logic.process_frame_logic(detections)
        draw_detections_with_logic(processed_frame, detections, obj_states)

        frame_time_ms = (perf_counter() - frame_start) * 1000.0
        frame_times_ms.append(frame_time_ms)
        if config.runtime.show_performance_stats:
            draw_performance_stats(
                processed_frame,
                _build_performance_stats(frame_times_ms, detect_times_ms, detect_time_ms),
            )

        cv2.imshow(config.window_title, processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


def _preprocess_frame(frame, config: AppConfig):
    if not config.preprocessing.enabled:
        return frame

    enhanced = frame.copy()

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.preprocessing.clahe_clip_limit,
        tileGridSize=(
            config.preprocessing.clahe_tile_grid_size,
            config.preprocessing.clahe_tile_grid_size,
        ),
    )
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    enhanced = cv2.convertScaleAbs(
        enhanced,
        alpha=config.preprocessing.contrast_alpha,
        beta=config.preprocessing.brightness_beta,
    )

    gamma = max(config.preprocessing.gamma, 0.01)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(enhanced, table)


def _build_performance_stats(
    frame_times_ms: deque[float],
    detect_times_ms: deque[float],
    current_detect_time_ms: float,
) -> dict[str, float]:
    avg_frame_ms = sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else 0.0
    avg_detect_ms = (
        sum(detect_times_ms) / len(detect_times_ms)
        if detect_times_ms
        else current_detect_time_ms
    )
    fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
    return {
        "fps": fps,
        "frame_ms": avg_frame_ms,
        "detect_ms": avg_detect_ms,
    }
