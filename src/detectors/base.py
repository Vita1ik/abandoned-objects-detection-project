from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect_and_track(self, frame) -> list[dict]:
        """Return normalized detections with stable IDs."""
