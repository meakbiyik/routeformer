from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from routeformer.models.config import RouteformerConfig
from routeformer.models.cross_modal_transformer import PerceiveEncoder as BetterPerceiveEncoder
from routeformer.models.gps_backbone.config import GPSBackboneConfig
from routeformer.models.video_backbone import SwinV2, VideoBackboneModule
from routeformer.models.video_backbone.config import TimmBackboneConfig
from routeformer.utils.filter import median_downsampler

from .base_cross_model import *


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


class AdaptedGIMO(nn.Module):
    """GIMO model adapted for Routeformer dataset.

    Adaptations:
    - Removed pointnet and other 3D scene processing
    - Used the existing video backbone as Routeformer and MultiModalNet for scene processing
    - Replaced config with RouteformerConfig and aligned parameters with the Routeformer framework
    - Align prediction length with RouteformerConfig
    - replaced scene_global_feats with the last frame of the scene_feats
    - replaced motion_scene_feats with fov scene feats from the gaze video
    - Gaze dimension is 2 instead of 3

    """

    def __init__(
        self,
        configs: RouteformerConfig,
        video_backbone: Optional[Type[VideoBackboneModule]] = None,
    ):
        super().__init__()
        self.configs = configs

        self.video_backbone = video_backbone(configs=self.configs.video_backbone_config)
        self.frame_encoder = BetterPerceiveEncoder(
            in_channels=self.video_backbone.output_feature_shape[0],
            out_len=1,
            out_channels=self.configs.image_embedding_size,
            n_heads=self.configs.encoder_heads,
            layers=self.configs.encoder_layers,
            dropout=self.configs.feature_dropout,
            d_ff=self.configs.encoder_d_ff,
        )

        self.motion_linear = nn.Linear(2, configs.encoder_hidden_size)
        input_len = configs.gps_backbone_config.seq_len
        output_len = configs.gps_backbone_config.pred_len
        self.motion_encoder = PerceiveEncoder(
            n_input_channels=2 * configs.encoder_hidden_size,
            n_latent=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            n_self_att_heads=configs.encoder_heads,
            n_self_att_layers=configs.encoder_layers,
            dropout=configs.feature_dropout,
        )
        self.motion_decoder = PerceiveDecoder(
            n_query_channels=configs.encoder_hidden_size,
            n_query=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            dropout=configs.feature_dropout,
        )  # (bs, out_len, motion_latent_d)

        self.motion_scene_decoder = PerceiveDecoder(
            n_query_channels=configs.encoder_hidden_size,
            n_query=input_len,
            n_latent_channels=2 * configs.encoder_hidden_size,
            dropout=configs.feature_dropout,
        )
        self.gaze_scene_decoder = PerceiveDecoder(
            n_query_channels=configs.encoder_hidden_size,
            n_query=input_len,
            n_latent_channels=configs.encoder_hidden_size,
            dropout=configs.feature_dropout,
        )

        # if self.use_gaze:
        self.gaze_linear = nn.Linear(2, configs.encoder_hidden_size)
        self.gaze_encoder = PerceiveEncoder(
            n_input_channels=configs.encoder_hidden_size,
            n_latent=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            n_self_att_heads=configs.encoder_heads,
            n_self_att_layers=configs.encoder_layers,
            dropout=configs.feature_dropout,
        )  # (bs, out_len, gaze_latent_d)
        self.gaze_motion_decoder = PerceiveDecoder(
            n_query_channels=configs.encoder_hidden_size,
            n_query=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            dropout=configs.feature_dropout,
        )  # (bs, out_len, gaze_latent_d)
        self.motion_gaze_decoder = PerceiveDecoder(
            n_query_channels=configs.encoder_hidden_size,
            n_query=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            dropout=configs.feature_dropout,
        )  # (bs, out_len, gaze_latent_d)
        # self.cross_modal_gaze_decoder = PerceiveDecoder(n_query_channels=configs.motion_latent_dim,
        #                                            n_query=output_len,
        #                                            n_latent_channels=configs.gaze_latent_dim,
        #                                            dropout=configs.dropout)  # (bs, out_len, gaze_latent_d)

        embedding_dim = 4 * configs.encoder_hidden_size
        self.embedding_layer = PositionwiseFeedForward(embedding_dim, embedding_dim)
        self.output_encoder = PerceiveEncoder(
            n_input_channels=embedding_dim,
            n_latent=output_len,
            n_latent_channels=configs.encoder_hidden_size,
            n_self_att_heads=configs.encoder_heads,
            n_self_att_layers=configs.encoder_layers,
            dropout=configs.feature_dropout,
        )  # (bs, out_len, gaze_latent_d)
        self.outputlayer = nn.Linear(configs.encoder_hidden_size, 2)

    def forward(self, batch, eval=False):
        """ """
        gps = batch["gps"].to(torch.float32)
        motion_vector = gps[:, 1:, :] - gps[:, :-1, :]
        motions = F.pad(motion_vector, (0, 0, 1, 0))

        left = batch["left_video"]
        right = batch.get("right_video", left)
        left_feats, right_feats = self._forward_single_video(left), self._forward_single_video(
            right
        )
        scene_feats = torch.cat([left_feats, right_feats], dim=2)
        scene_global_feats = scene_feats[:, -1:, :].repeat(
            1, self.configs.gps_backbone_config.pred_len, 1
        )

        motion_feats = self.motion_linear(motions)
        motion_scene_feats = self.motion_scene_decoder(motion_feats, scene_feats)
        motion_feats = torch.cat([motion_feats, motion_scene_feats], dim=2)
        motion_embedding = self.motion_encoder(motion_feats)

        front = batch["front_video"]
        raw_gaze = batch["gaze"].to(torch.float32)  # B x longer_than_seq_len x 2
        gazes = median_downsampler(raw_gaze, self.configs.gps_backbone_config.seq_len)
        front_feats = self._forward_single_video(front)
        gaze_embedding = self.gaze_linear(gazes)
        gaze_embedding = self.gaze_scene_decoder(gaze_embedding, front_feats)
        gaze_embedding = self.gaze_encoder(gaze_embedding)

        gaze_motion_embedding = self.gaze_motion_decoder(gaze_embedding, motion_embedding)
        motion_gaze_embedding = self.motion_gaze_decoder(motion_embedding, gaze_embedding)

        cross_modal_embedding = torch.cat(
            [scene_global_feats, gaze_motion_embedding, motion_gaze_embedding],
            dim=2,
        )

        cross_modal_embedding = self.embedding_layer(cross_modal_embedding)
        cross_modal_embedding = self.output_encoder(cross_modal_embedding)
        output = self.outputlayer(cross_modal_embedding)

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
        video_features = video_features.view(batch_size, -1, self.configs.image_embedding_size).to(
            gps_backbone_dtype
        )

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

    model = AdaptedGIMO(ROUTEFORMER_CONFIG, SwinV2)
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
