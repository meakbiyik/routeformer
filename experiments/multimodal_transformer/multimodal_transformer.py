from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from routeformer.models.config import RouteformerConfig
from routeformer.models.cross_modal_transformer import PerceiveEncoder
from routeformer.models.gps_backbone import Transformer
from routeformer.models.gps_backbone.config import GPSBackboneConfig
from routeformer.models.video_backbone import SwinV2, VideoBackboneModule
from routeformer.models.video_backbone.config import TimmBackboneConfig
from routeformer.utils.filter import median_downsampler


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ToDtype(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


class MultiModalTransformer(nn.Module):
    """A simple multi-modal transformer that takes in scene, video and gaze and outputs a single modality."""

    def __init__(
        self,
        configs: RouteformerConfig,
        video_backbone: Optional[Type[VideoBackboneModule]] = None,
    ):
        super().__init__()
        self.configs = configs

        self.video_backbone = video_backbone(configs=self.configs.video_backbone_config)
        self.frame_encoder = PerceiveEncoder(
            in_channels=self.video_backbone.output_feature_shape[0],
            out_len=1,
            out_channels=self.configs.image_embedding_size,
            n_heads=self.configs.encoder_heads,
            layers=self.configs.encoder_layers,
            dropout=self.configs.feature_dropout,
            d_ff=self.configs.encoder_d_ff,
        )

        self.motion_linear = nn.Linear(2, configs.encoder_hidden_size)

        self.gaze_linear = nn.Linear(2, configs.encoder_hidden_size)

        gps_backbone_config = self.configs.gps_backbone_config.copy()
        gps_backbone_config._enc_in = configs.encoder_hidden_size * 5
        gps_backbone_config._c_out = 2

        self.transformer = Transformer(configs=gps_backbone_config)

    def forward(self, batch, eval=False):
        """ """
        gps = batch["gps"].to(torch.float32)
        motion_vector = gps[:, 1:, :] - gps[:, :-1, :]
        motions = F.pad(motion_vector, (0, 0, 1, 0))
        motion_feats = self.motion_linear(motions)

        left = batch["left_video"]
        right = batch.get("right_video", left)
        left_feats = self._forward_single_video(left)
        right_feats = self._forward_single_video(right)
        scene_feats = torch.cat([left_feats, right_feats], dim=2)

        gaze_video = batch["front_video"]
        gaze_video_feats = self._forward_single_video(gaze_video)

        raw_gaze = batch["gaze"].to(torch.float32)  # B x longer_than_seq_len x 2
        gazes = median_downsampler(
            raw_gaze, self.configs.gps_backbone_config.seq_len
        )  # we also use median downsampling as with GIMO
        gaze_feats = self.gaze_linear(gazes)

        feats = torch.cat(
            [motion_feats, scene_feats, gaze_video_feats, gaze_feats], axis=2
        )

        output = self.transformer(feats)

        last_input_gps = gps[:, -1:, :]
        future_gps_positions = last_input_gps + torch.cumsum(output, dim=1)

        return future_gps_positions

    def _forward_single_video(self, video):
        batch_size = video.shape[0]
        video = video.flatten(0, 1)
        gps_backbone_dtype = self.motion_linear.parameters().__next__().dtype
        video_features = self.video_backbone(video).to(gps_backbone_dtype)

        # The output of the video backbone is a 2D feature map, with shape
        # (B, C, H, W), we first convert it to (B, H*W, C) and then
        # apply an encoder to get the features.
        video_features = video_features.permute(0, 2, 3, 1).reshape(
            video_features.shape[0], -1, video_features.shape[1]
        )
        video_features = torch.cat(
            [
                video_features,
                -torch.ones_like(video_features)[:, :1, :],
            ],
            dim=1,
        )
        video_features = self.frame_encoder(video_features)
        video_features = video_features.view(
            batch_size, -1, self.configs.image_embedding_size
        ).to(gps_backbone_dtype)

        return video_features

if __name__ == "__main__":
    ROUTEFORMER_CONFIG = RouteformerConfig(
        encoder_hidden_size=16,
        batch_size=2,
        gps_backbone_config=GPSBackboneConfig(
            # Fixed parameters - dictated by dataset
            seq_len=4,
            label_len=4,
            pred_len=2,
        ),
        video_backbone_config=TimmBackboneConfig(
            model_type="swinv2_tiny_window16_256.ms_in1k",
            train_backbone=False,
            backbone_minibatch_size=2,
            pad_to_square=True,
        ),
    )

    model = MultiModalTransformer(ROUTEFORMER_CONFIG, SwinV2)
    print(model)

    batch = {
        "left_video": torch.randn(2, 4, 3, 128, 128),
        "right_video": torch.randn(2, 4, 3, 128, 128),
        "front_video": torch.randn(2, 4, 3, 128, 128),
        "gaze": torch.randn(2, 4 * 5, 2),
        "gps": torch.randn(2, 4, 2),
    }
    output = model(batch)
    print(output.shape)
