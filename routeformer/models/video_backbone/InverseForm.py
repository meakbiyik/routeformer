"""InverseForm backbone from Qualcomm research."""
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import InverseFormBackboneConfig, VideoBackboneModule
from .inverse_form_layers.config import assert_and_infer_cfg
from .inverse_form_layers.lighthrnet import LightHighResolutionNet


class InverseForm(VideoBackboneModule):
    """InverseForm backbone.

    Takes a batch of images as input and outputs a batch of feature vectors.
    """

    def __init__(
        self,
        configs: InverseFormBackboneConfig,
    ):
        """Initialize the model.

        Parameters
        ----------
        configs : InverseFormBackboneConfig
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
        self.model_path = configs.model_path
        self.download_model = configs.download_model

        model_path = Path(configs.model_path)

        if self.download_model and not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                "https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hr16s_4k_slim.pth",  # noqa: E501
                model_path,
            )

        assert_and_infer_cfg(
            result_dir=None,
            global_rank=None,
            apex=False,
            syncbn=False,
            arch="lighthrnet.HRNet16",
            hrnet_base=16,
            fp16=False,
            has_edge=True,
        )

        backbone_state_dict = torch.load(model_path, map_location=self.device)
        full_model: nn.Module = LightHighResolutionNet(0)
        self.load_model(full_model, backbone_state_dict)
        self._Backbone = full_model.backbone
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.Backbone = lambda x: self.adaptive_pool(self._Backbone(x)[-1])

        if not self.train_backbone:
            self._Backbone.eval()
            self._Backbone.requires_grad_(False)
        else:
            self._Backbone.train()
            self._Backbone.requires_grad_(False)
            self._Backbone.stage4.requires_grad_(True)

        # Let's just push a random image through the model to get the output shape
        # of the feature vector
        random_image = torch.rand(1, 3, 224, 224)
        output_shape = self.Backbone(random_image).shape[1:]
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

    @staticmethod
    def load_model(net, pretrained):
        """Load the pretrained model.

        Parameters
        ----------
        net : nn.Module
            The model to load the pretrained weights into.
        pretrained : dict
            The pretrained weights.
        """
        pretrained_dict = pretrained["state_dict"]
        model_dict = net.state_dict()
        updated_model_dict = {}
        lookup_table = {}
        for k_model, v_model in model_dict.items():
            if k_model.startswith("model") or k_model.startswith("module"):
                k_updated = ".".join(k_model.split(".")[1:])
                if k_updated.startswith("backbone"):
                    k_updated = ".".join(k_updated.split(".")[1:])

                lookup_table[k_updated] = k_model
                updated_model_dict[k_updated] = k_model
            else:
                lookup_table[k_model] = k_model
                updated_model_dict[k_model] = k_model

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("model") or k.startswith("modules") or k.startswith("module"):
                k = ".".join(k.split(".")[1:])
            if k.startswith("backbone"):
                k = ".".join(k.split(".")[1:])

            if k in updated_model_dict.keys() and model_dict[lookup_table[k]].shape == v.shape:
                updated_pretrained_dict[updated_model_dict[k]] = v

        model_dict.update(updated_pretrained_dict)
        net.load_state_dict(model_dict)
        return net

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
        images = images.to(torch.float32)
        image_count = images.shape[0]
        embeddings = torch.empty(
            (image_count, *self.output_feature_shape),
            device=images.device,
            dtype=images.dtype,
        )
        # split the batch into minibatches of size self.batch_size and process them
        # separately, concatenating the results at the end.
        for i in range(0, image_count, self.minibatch_size):
            embeddings[i : i + self.minibatch_size] = self.Backbone(
                images[i : i + self.minibatch_size],
            )

        # Re-arrange the batch dimensions
        # Cast it just in case to the same dtype as the input, who knows
        # what happens in the torch.autocast context manager
        embeddings = embeddings.reshape(-1, *self.output_feature_shape).to(dtype=images.dtype)

        return embeddings


if __name__ == "__main__":
    config = InverseFormBackboneConfig(download_model=True, model_path="if.pth")
    model = InverseForm(config)
