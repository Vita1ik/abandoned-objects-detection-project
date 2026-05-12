from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from time import perf_counter, sleep

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
    cv2.namedWindow(config.window_title, cv2.WINDOW_NORMAL)
    detector = build_detector(config.detector)
    logic = AbandonedLogic(
        abandon_threshold=config.logic.abandon_threshold,
        dist_threshold=config.logic.dist_threshold,
        fix_confirm_threshold=config.logic.fix_confirm_threshold,
    )
    frame_index = 0
    last_detections = []
    loop_times_ms: deque[float] = deque(maxlen=config.runtime.performance_window_size)
    display_intervals_ms: deque[float] = deque(maxlen=config.runtime.performance_window_size)
    detect_times_ms: deque[float] = deque(maxlen=config.runtime.performance_window_size)
    detect_future: Future[tuple[list[dict], float]] | None = None
    target_frame_interval_s = _resolve_target_frame_interval(cap, config.video_source)
    next_frame_deadline = perf_counter()
    last_presented_at: float | None = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        while cap.isOpened():
            if target_frame_interval_s is not None:
                now = perf_counter()
                if now < next_frame_deadline:
                    sleep(next_frame_deadline - now)

            frame_start = perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = _preprocess_frame(frame, config)
            display_size = _resolve_display_size(processed_frame, config)
            cv2.resizeWindow(config.window_title, *display_size)
            detect_time_ms = 0.0

            if detect_future is not None and detect_future.done():
                last_detections, detect_time_ms = detect_future.result()
                detect_times_ms.append(detect_time_ms)
                detect_future = None

            should_detect = frame_index % config.runtime.detect_every_n_frames == 0
            if should_detect and detect_future is None:
                detect_future = executor.submit(
                    _detect_frame,
                    detector,
                    processed_frame.copy(),
                )

            detections = [dict(detection) for detection in last_detections]
            obj_states = logic.process_frame_logic(detections)
            draw_detections_with_logic(processed_frame, detections, obj_states)

            loop_time_ms = (perf_counter() - frame_start) * 1000.0
            loop_times_ms.append(loop_time_ms)
            if config.runtime.show_performance_stats:
                presented_at = perf_counter()
                if last_presented_at is not None:
                    display_intervals_ms.append((presented_at - last_presented_at) * 1000.0)
                last_presented_at = presented_at
                draw_performance_stats(
                    processed_frame,
                    _build_performance_stats(
                        display_intervals_ms,
                        loop_times_ms,
                        detect_times_ms,
                        detect_time_ms,
                    ),
                )

            display_frame = cv2.resize(
                processed_frame,
                display_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(config.window_title, display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_index += 1
            if target_frame_interval_s is not None:
                next_frame_deadline = max(
                    next_frame_deadline + target_frame_interval_s,
                    perf_counter(),
                )

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


def _detect_frame(detector, frame) -> tuple[list[dict], float]:
    detect_start = perf_counter()
    detections = detector.detect_and_track(frame)
    detect_time_ms = (perf_counter() - detect_start) * 1000.0
    return detections, detect_time_ms


def _resolve_target_frame_interval(cap, video_source) -> float | None:
    if isinstance(video_source, int):
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        return None

    return 1.0 / fps


def _resolve_display_size(frame, config: AppConfig) -> tuple[int, int]:
    frame_height, frame_width = frame.shape[:2]
    scale = config.window.scale
    width = max(1, int(frame_width * scale))
    height = max(1, int(frame_height * scale))
    return width, height


def _build_performance_stats(
    display_intervals_ms: deque[float],
    loop_times_ms: deque[float],
    detect_times_ms: deque[float],
    current_detect_time_ms: float,
) -> dict[str, float]:
    avg_display_ms = (
        sum(display_intervals_ms) / len(display_intervals_ms) if display_intervals_ms else 0.0
    )
    avg_loop_ms = sum(loop_times_ms) / len(loop_times_ms) if loop_times_ms else 0.0
    avg_detect_ms = (
        sum(detect_times_ms) / len(detect_times_ms)
        if detect_times_ms
        else current_detect_time_ms
    )
    display_fps = 1000.0 / avg_display_ms if avg_display_ms > 0 else 0.0
    detect_fps = 1000.0 / avg_detect_ms if avg_detect_ms > 0 else 0.0
    return {
        "display_fps": display_fps,
        "display_ms": avg_display_ms,
        "loop_ms": avg_loop_ms,
        "detect_ms": avg_detect_ms,
        "detect_fps": detect_fps,
    }
