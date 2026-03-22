from __future__ import annotations

import cv2

from src.config import AppConfig, load_app_config
from src.detectors.factory import build_detector
from src.logic import AbandonedLogic
from src.visualization.draw import draw_detections_with_logic


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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_and_track(frame)
        obj_states = logic.process_frame_logic(detections)
        draw_detections_with_logic(frame, detections, obj_states)

        cv2.imshow(config.window_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
