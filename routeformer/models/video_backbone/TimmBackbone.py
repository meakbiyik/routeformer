"""Timm-based vision backbone for Routeformer."""
from pathlib import Path
from typing import Optional

import timm
import torch
from torch import Tensor, nn
from torchvision import transforms

from .config import TimmBackboneConfig, VideoBackboneModule


class TimmBackbone(VideoBackboneModule):
    """Timm-based vision backbones for Routeformer.

    Takes a batch of images as input and outputs a batch of feature vectors.
    """

    def __init__(
        self,
        configs: Optional[TimmBackboneConfig] = None,
    ):
        """Initialize the model.

        Parameters
        ----------
        configs : TimmBackboneConfig, optional
            Configurations for the model, by default None. May contain the following:
                model_type : str
                    Type of the model
                train_backbone : bool
                    Whether to train the backbone.
                backbone_minibatch_size : int
                    Batch size for the backbone to process the images.
        """
        super().__init__()
        self.configs = configs
        self.minibatch_size = configs.backbone_minibatch_size
        self.train_backbone = configs.train_backbone

        self.Backbone: nn.Module = timm.create_model(
            configs.model_type,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        data_config = timm.data.resolve_model_data_config(self.Backbone)
        self.transforms: transforms.Compose = timm.data.create_transform(
            **data_config,
            is_training=False,
        )

        if not self.train_backbone:
            self.Backbone.eval()
            self.Backbone.requires_grad_(False)
        else:
            self.train_transforms = transforms.Compose([
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
            ])

        # we remove transforms.ToTensor() from the transforms pipeline
        # because images are already tensors
        self.transforms.transforms = [
            t
            for t in self.transforms.transforms
            if not isinstance(t, transforms.ToTensor)
        ]

        self.outputs_bhwc = (
            "swin" in configs.model_type and "_cr" not in configs.model_type
        )
        self.outputs_bpc = (
            "dino" in configs.model_type
        )  # DINO outputs BxPxC, where P is the patch count

        # Let's just push a random image through the model to get the output shape
        # of the feature vector
        random_image = self.transforms(torch.rand(1, 3, 224, 224))
        output_shape = self.Backbone(random_image).shape[1:]
        if self.outputs_bhwc:
            output_shape = output_shape[::-1]
        elif self.outputs_bpc:
            output_shape = (output_shape[1], output_shape[0], 1)
        self._output_feature_shape = output_shape

        for k, v in configs.__dict__.items():
            if k.startswith("torchcache_") and v is not None:
                setattr(self, k, v)

        # Override torchcache's persistent_cache_dir manually
        candidate_cache_dir = configs.cache_dir
        if candidate_cache_dir is not None:
            self.torchcache_persistent_cache_dir = (
                Path(candidate_cache_dir) / f"torchcache_{configs.model_type}"
            )

    @property
    def output_feature_shape(self) -> tuple:
        """Shape of the output feature map of the backbone (CxHxW)."""
        return self._output_feature_shape

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        images : Tensor
            Batch of images with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Batch of feature vectors with shape (B, C_f, H_f, W_f),
            where C_f is the channel dimension of the feature vector.
        """
        self.Backbone.requires_grad_(self.current_epoch > 10)

        image_count = images.shape[0]
        embeddings = torch.empty(
            (image_count, *self.output_feature_shape),
            device=images.device,
            dtype=images.dtype,
        )

        with torch.autocast("cuda" if images.is_cuda else "cpu"):
            # split the batch into minibatches of size self.batch_size and process them
            # separately, concatenating the results at the end.
            for i in range(0, image_count, self.minibatch_size):
                embeddings[i : i + self.minibatch_size] = self._forward(
                    images[i : i + self.minibatch_size],
                )

        # Re-arrange the batch dimensions
        # Cast it just in case to the same dtype as the input, who knows
        # what happens in the torch.autocast context manager
        embeddings = embeddings.reshape(-1, *self.output_feature_shape).to(
            dtype=images.dtype
        )

        return embeddings

    def _forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        images : Tensor
            Batch of images with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Batch of feature vectors with shape (B, C_f, H_f, W_f),
            where C_f is the channel dimension of the feature vector.
        """
        if self.configs.pad_to_square:
            max_dim = max(images.shape[2:])
            pad_height = max_dim - images.shape[2]
            pad_width = max_dim - images.shape[3]
            images = torch.nn.functional.pad(
                images,
                (0, pad_width, 0, pad_height),
                mode="constant",
                value=0,
            )
        images = self.transforms(images)
        if self.train_backbone and self.training:
            images = self.train_transforms(images)
        out = self.Backbone(images).to(dtype=images.dtype)

        if self.outputs_bpc:
            out = out.permute(0, 2, 1).unsqueeze(-1)
        elif self.outputs_bhwc:
            out = out.permute(0, 3, 1, 2)

        return out
