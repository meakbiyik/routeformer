"""Backbone models for SAM."""
from torchcache import torchcache

from .config import (
    InverseFormBackboneConfig,
    TimmBackboneConfig,
    VideoBackboneConfig,
    VideoBackboneModule,
)
from .InverseForm import InverseForm
from .TimmBackbone import TimmBackbone


@torchcache(persistent=True)
class SwinV2(TimmBackbone):
    """Placeholder class to separate caches for SwinV2 models."""

    ...


@torchcache(persistent=True)
class DinoV2(TimmBackbone):
    """Placeholder class to separate caches for DinoV2 models."""

    ...


@torchcache(persistent=True)
class Sam(TimmBackbone):
    """Placeholder class to separate caches for SAM models."""

    ...


__all__ = [
    "SwinV2",
    "DinoV2",
    "Sam",
    "VideoBackboneConfig",
    "VideoBackboneModule",
    "TimmBackboneConfig",
    "InverseFormBackboneConfig",
    "InverseForm",
]
