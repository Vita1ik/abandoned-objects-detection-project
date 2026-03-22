from src.config import load_app_config


def test_load_app_config_defaults_for_missing_file():
    config = load_app_config("missing-config.yaml")

    assert config.detector.family == "yolov8"
    assert config.detector.variant == "default"
    assert config.logic.abandon_threshold == 5
