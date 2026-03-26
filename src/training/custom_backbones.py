from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class TimmFeatureBackbone(nn.Module):
    """A timm-backed feature extractor compatible with Ultralytics TorchVision-style YAML wiring."""

    def __init__(
        self,
        model: str,
        weights: str = "DEFAULT",
        unwrap: bool = True,
        truncate: int = 0,
        split: bool = True,
        out_indices: Iterable[int] = (1, 2, 3),
        proj_channels: Iterable[int] | None = None,
    ):
        super().__init__()
        del unwrap, truncate  # kept for TorchVision YAML compatibility

        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment-specific guidance
            raise ImportError(
                "The timm package is required for MobileNetV4 backbones. "
                "Install it with `pip install timm` in the active environment."
            ) from exc

        self.split = split
        self.out_indices = tuple(int(index) for index in out_indices)
        pretrained = str(weights).upper() != "NONE"
        self.m = timm.create_model(
            model,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        feature_channels = list(self.m.feature_info.channels())
        if proj_channels is None:
            proj_channels = feature_channels
        proj_channels = [int(channel) for channel in proj_channels]
        if len(feature_channels) != len(proj_channels):
            raise ValueError(
                "proj_channels must match the number of selected MobileNetV4 feature maps."
            )

        self.projections = nn.ModuleList(
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(feature_channels, proj_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        features = self.m(x)
        projected = [proj(feature) for proj, feature in zip(self.projections, features)]
        if self.split:
            return [x, *projected]
        return projected[-1]


def register_ultralytics_custom_backbones(variant: str | None = None) -> None:
    """Register project-specific backbones for Ultralytics YAML parsing."""

    if (variant or "").lower() != "mobilenetv4":
        return

    import ultralytics.nn.tasks as tasks

    # Reuse the existing TorchVision parse-model branch so custom timm backbones can
    # be wired through YAML without patching Ultralytics internals on disk.
    tasks.TorchVision = TimmFeatureBackbone
    tasks.parse_model.__globals__["TorchVision"] = TimmFeatureBackbone
