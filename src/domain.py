from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    id: int
    box: list[float]
    class_name: str
    confidence: float
    center: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "box": self.box,
            "class": self.class_name,
            "conf": self.confidence,
            "center": self.center,
        }
