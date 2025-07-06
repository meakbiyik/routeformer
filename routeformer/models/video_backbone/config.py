"""Video backbone config definition."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import lightning as L

from routeformer.utils.config import BaseConfig


@dataclass
class VideoBackboneConfig(BaseConfig):
    """Config for video backbones."""

    cache_dir: str = None
    train_backbone: bool = False
    backbone_minibatch_size: int = 4
    # Module hash for torchcache to override the default hash
    # optionally prevent minor changes in the module from invalidating the cache
    torchcache_enabled: bool = True
    torchcache_persistent_module_hash: str = None
    torchcache_max_persistent_cache_size: int = 200e9
    torchcache_max_memory_cache_size: int = 20e9

    def __post_init__(self):
        if self.torchcache_enabled and self.train_backbone:
            raise ValueError("torchcache_enabled and train_backbone cannot both be True.")


@dataclass
class TimmBackboneConfig(VideoBackboneConfig):
    """Config for Timm-based backbone."""

    pad_to_square: bool = True
    model_type: str = None


@dataclass
class InverseFormBackboneConfig(VideoBackboneConfig):
    """Config for InverseForm-based backbone."""

    download_model: bool = False
    model_path: str = None


class VideoBackboneModule(ABC, L.LightningModule):
    """Abstract class for video backbones."""

    @property
    @abstractmethod
    def output_feature_shape(self) -> tuple:
        """Shape of the output feature map of the backbone (CxHxW)."""
        ...
