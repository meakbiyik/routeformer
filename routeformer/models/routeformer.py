"""Implements Routeformer model."""

from typing import Optional, Type

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from routeformer.io.dataset import Data
from routeformer.models.cross_modal_transformer import PerceiveDecoder, PerceiveEncoder
from routeformer.utils.filter import median_downsampler
from routeformer.utils.vector import estimate_angle_and_norm, rotate

from .config import RouteformerConfig
from .gps_backbone import Informer
from .video_backbone import VideoBackboneModule


class Routeformer(L.LightningModule):
    """Routeformer model that predicts the future trajectory of a vehicle."""

    def __init__(
        self,
        configs: RouteformerConfig,
        gps_backbone: Optional[Type[L.LightningModule]] = Informer,
        video_backbone: Optional[Type[VideoBackboneModule]] = None,
    ):
        """Initialize the model.

        Parameters
        ----------
        configs : RouteformerConfig
            Configurations for the model.
        gps_backbone : Optional[Type[L.LightningModule]], optional
            GPS backbone, by default Informer.
        video_backbone : Optional[Type[VideoBackboneModule]], optional
            Video backbone, by default None.
        """
        super().__init__()

        self.configs = configs.copy()
        self.with_video = (
            self.configs.with_video
            if self.configs.with_video is not None
            else video_backbone is not None
        )
        self.with_scene = self.configs.with_scene
        self.with_gaze = self.configs.with_gaze

        if not self.with_video and self.with_gaze:
            raise ValueError(
                "Current gaze backbone requires a video backbone, but video backbone is not provided."  # noqa: E501
            )

        if self.with_video:
            self.video_backbone = video_backbone(
                configs=self.configs.video_backbone_config
            )

            self.frame_encoder = PerceiveEncoder(
                in_channels=self.video_backbone.output_feature_shape[0],
                out_len=1,
                out_channels=self.configs.image_embedding_size,
                n_heads=self.configs.encoder_heads,
                layers=self.configs.encoder_layers,
                d_ff=self.configs.encoder_d_ff,
                dropout=self.configs.feature_dropout,
            )

            # Side encoding is used to distinguish the left, right and gaze views
            # in the self-attention layer. -1, 0, 1 are used to encode the left,
            # right and gaze views, respectively.
            self.left_video_embedding = nn.Parameter(
                torch.randn(1, 1, self.configs.image_embedding_size, device=self.device)
            )
            self.right_video_embedding = nn.Parameter(
                torch.randn(1, 1, self.configs.image_embedding_size, device=self.device)
            )
            self.gaze_video_embedding = nn.Parameter(
                torch.randn(1, 1, self.configs.image_embedding_size, device=self.device)
            )
            self.video_output_embedding = nn.Parameter(
                torch.randn(1, 1, self.configs.image_embedding_size, device=self.device)
            )

            self.video_encoder = PerceiveEncoder(
                in_channels=self.configs.image_embedding_size,
                out_len=self.configs.gps_backbone_config.seq_len,
                out_channels=self.configs.encoder_hidden_size,
                n_heads=self.configs.encoder_heads,
                layers=self.configs.encoder_layers,
                d_ff=self.configs.encoder_d_ff,
                dropout=self.configs.feature_dropout,
            )

            if self.with_gaze:
                self.gaze_encoder = PerceiveEncoder(
                    in_channels=2,
                    out_len=self.configs.gps_backbone_config.seq_len,
                    out_channels=self.configs.encoder_hidden_size,
                    n_heads=self.configs.encoder_heads,
                    layers=self.configs.encoder_layers,
                    d_ff=self.configs.encoder_d_ff,
                    dropout=self.configs.feature_dropout,
                )
                self.gaze_video_decoder = PerceiveDecoder(
                    query_channels=self.configs.encoder_hidden_size,
                    value_channels=self.configs.encoder_hidden_size,
                    out_channels=self.configs.encoder_hidden_size,
                    out_len=self.configs.gps_backbone_config.seq_len,
                    dropout=self.configs.feature_dropout,
                    d_ff=self.configs.encoder_d_ff,
                    n_heads=self.configs.cross_modal_decoder_heads,
                    layers=self.configs.cross_modal_decoder_layers,
                    mix=False,
                )

        self.gps_backbone = gps_backbone(configs=self.configs.gps_backbone_config)

        self.view_dropout = self.configs.view_dropout
        self.motion_noise = self.configs.motion_noise
        self.gaze_dropout = self.configs.gaze_dropout
        self.feature_dropout = self.configs.feature_dropout

    def forward(
        self,
        batch: Data,
        target_batch: Optional[Data] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        batch : Data
            Batch of data. A dictionary that may include the following keys:
            - gps: GPS coordinates, shape (B, T, 2).
            - left_video: Left video, shape (B, T, C, H, W).
            - right_video: Right video, shape (B, T, C, H, W).
            - gaze: Gaze coordinates, shape (B, T, 2).
            - front_video: Front video, shape (B, T, C, H, W).
        eval : boolean, optional
            Autoregressively predict the result

        Returns
        -------
        Tensor
            Predicted future GPS coordinates, shape (B, T, 2).
        Tensor, optional
            Predicted future visual features, shape (B, T, C).
        """
        (motion_dynamics, visual_features) = self.preprocess_batch(batch)

        input_gps = batch["gps"]
        last_input_gps = input_gps[:, -1:, :]
        eval = not self.training

        if not (eval and self.configs.autoregressive):
            output, _ = self._forward(motion_dynamics, visual_features)
            (
                _,  # future_motion_vector
                future_gps_positions,
                future_visual_features,
            ) = self.postprocess_batch(last_input_gps, output)

        else:
            # predict the output one input at a time
            # to autoregressively predict the result
            # the input is the last output
            outputs = []
            current_time_length = 0
            step_size = self.configs.autoregressive_step_size
            pred_len = self.gps_backbone.pred_len
            self.gps_backbone.pred_len = step_size
            while current_time_length < pred_len:
                data_dtype = motion_dynamics.dtype
                output, _ = self._forward(motion_dynamics, visual_features)
                (
                    future_motion_vector,
                    future_gps_positions,
                    future_visual_features,
                ) = self.postprocess_batch(last_input_gps, output)
                outputs.append((future_gps_positions, future_visual_features))
                motion_dynamics = torch.cat(
                    [motion_dynamics[:, step_size:], future_motion_vector], dim=1
                ).to(data_dtype)
                last_input_gps = future_gps_positions[:, -1:, :]
                visual_features = torch.cat(
                    [visual_features[:, step_size:], future_visual_features], dim=1
                ).to(data_dtype)
                current_time_length += step_size

            self.gps_backbone.pred_len = pred_len

            future_gps_positions = torch.cat([output[0] for output in outputs], dim=1)[
                :, :pred_len
            ]
            if self.with_video:
                future_visual_features = torch.cat(
                    [output[1] for output in outputs], dim=1
                )[:, :pred_len]

        if self.configs.dense_prediction:
            return (future_gps_positions, future_visual_features)

        return future_gps_positions

    def _forward(
        self,
        motion_dynamics,
        visual_features,
    ):
        # enrich the motion dynamics with angle, norm (speed) and acceleration
        angle, norm = estimate_angle_and_norm(motion_dynamics)
        if self.configs.rotate_motion:
            origin_angles = angle[:, -1:, :]
        else:
            origin_angles = angle[:, :1, :]
        normalized_angles = angle - origin_angles
        # angles are between -pi and pi, normalize them to [-1, 1]
        normalized_angles = normalized_angles / torch.pi
        acceleration = norm[:, 1:, :] - norm[:, :-1, :]
        acceleration = F.pad(acceleration, (0, 0, 1, 0))
        if self.configs.rotate_motion:
            motion_dynamics = rotate(motion_dynamics, -origin_angles)
    
        motion_dynamics = torch.cat([motion_dynamics, normalized_angles, norm, acceleration], dim=-1)

        inputs = [motion_dynamics]

        if self.with_video:
            inputs.append(visual_features)
        
        if self.configs._only_motion:
            inputs[-1] = torch.zeros_like(inputs[-1])

        input = torch.cat(inputs, dim=-1)

        attention = None
        if self.configs.output_attention:
            output, attention = self.gps_backbone(input)
        else:
            output = self.gps_backbone(input)

        if self.configs.decoder_mode == "recursive":
            if self.configs.dense_prediction:
                output = output + input[:, -1:, :]
            else:
                output = output + input[:, -1:, :2]
        
        if self.configs.rotate_motion:
            output[:, :, :2] = rotate(output[:, :, :2], origin_angles)

        return output, attention

    def preprocess_batch(  # noqa: C901 # FIXME: refactor, remove unnecessary branches
        self,
        batch,
        training: bool = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess the batch.

        Parameters
        ----------
        batch : Data
            Batch of data. See forward() for details.
        training : bool, optional
            Whether the model is in training mode, by default None.
            If None, it is set to self.training.

        Returns
        -------
        Tensor
            GPS dynamics, shape (B, T, 2).
        Tensor
            Visual features, shape (B, T, C).
        """
        if training is None:
            training = self.training

        gps = batch["gps"].to(torch.float32)

        if self.motion_noise > 0.0 and self.training:
            gps = gps + torch.randn_like(gps) * self.motion_noise

        motion_vector = gps[:, 1:, :] - gps[:, :-1, :]

        if self.configs.normalize_motion:
            motion_vector = (
                motion_vector - self.configs.motion_mean
            ) / self.configs.motion_std

        motion_dynamics = motion_vector
        # pad the dynamics with zeros in the temporal dimension
        # so that video features can be aligned with the dynamics.
        motion_dynamics = F.pad(motion_dynamics, (0, 0, 1, 0))
        visual_features = []
        if self.with_video and self.with_scene:
            left_features, right_features = self._forward_video(batch, training)
            visual_features.extend([left_features, right_features])

        if self.with_gaze:
            drop_gaze = False
            if self.gaze_dropout > 0.0 and training:
                drop_gaze = torch.rand(1) < self.gaze_dropout

            if drop_gaze:
                gaze_features = torch.zeros(
                    batch["front_video"].shape[0],
                    batch["front_video"].shape[1],
                    self.configs.image_embedding_size,
                    dtype=motion_dynamics.dtype,
                    device=motion_dynamics.device,
                )
            else:
                gaze_positions = batch["gaze"].to(torch.float32)
                gaze_video_features = self._forward_gaze_video(batch, training)
                input_sequence_length = gaze_video_features.shape[1]
                gaze_positions = median_downsampler(
                    gaze_positions, self.configs.gps_backbone_config.seq_len
                )  # we also use median downsampling as with GIMO
                gaze_positions = self.gaze_encoder(gaze_positions)
                gaze_features = self.gaze_video_decoder(
                    gaze_video_features,
                    gaze_positions,  # Q: gaze_positions, K,V: gaze_video_features
                )
                # Ensure that the lengths of the gaze and video features are the same
                # as for dense prediction, we might use this method to output
                # the target features, which has a shorter sequence length
                # than the input features.
                gaze_features = gaze_features[:, :input_sequence_length]

            visual_features.append(gaze_features)
        
        if self.with_video:
            # visual features are a list of [left, right, gaze], gaze optional
            # let's add the embeddings to them before allowing self attention
            if self.with_scene:
                visual_features[0] = visual_features[0] + self.left_video_embedding
                visual_features[1] = visual_features[1] + self.right_video_embedding
            if self.with_gaze:
                visual_features[-1] = visual_features[-1] + self.gaze_video_embedding
            visual_features = torch.cat(
                [
                    *visual_features,
                    torch.zeros_like(visual_features[-1]) + self.video_output_embedding,
                ],
                dim=1,
            )
            visual_features = self.video_encoder(visual_features)

        return motion_dynamics, visual_features

    def postprocess_batch(self, last_input_gps, output):
        """Postprocess the output of the model.

        Parameters
        ----------
        last_input_gps : Tensor
            Last input GPS coordinates, shape (B, T, 2).
        output : Tensor
            Output of the model. Shape (B, T, 2), or (B, T, [2 or 3]*C) if dense_prediction.

        Returns
        -------
        Tensor
            Predicted future GPS coordinates, shape (B, T, 2).
        Tensor
            Predicted future visual features, shape (B, T, C).
        """
        future_motion_vector = output[:, :, :2]
        if self.configs.normalize_motion:
            future_motion_vector = (
                future_motion_vector * self.configs.motion_std
            ) + self.configs.motion_mean
        # integrate the motion vectors to get the future GPS coordinates
        future_gps_positions = last_input_gps + torch.cumsum(
            future_motion_vector, dim=1
        )
        future_gps_positions = future_gps_positions.to(last_input_gps.dtype)
        output = output[:, :, 2:]

        future_visual_features = None
        if self.with_video and self.configs.dense_prediction:
            assert output.shape[-1] >= self.configs.image_embedding_size, (
                "Output shape for left/right vid. must be at least "
                f"{self.configs.image_embedding_size}, but is "
                f"{output.shape}."
            )
            future_visual_features = output[:, :, : self.configs.image_embedding_size]
            output = output[:, :, self.configs.image_embedding_size :]
    
        assert (
            output.shape[-1] == 0
        ), f"Output should be empty at this point, but is {output.shape}."  # noqa: E501

        return (
            future_motion_vector,
            future_gps_positions,
            future_visual_features,
        )

    def _forward_video(self, batch, training: bool):
        if training is None:
            training = self.training
        left = batch["left_video"]
        right = batch.get("right_video", left)
        # apply view dropout, which is completely dropping out
        # one of the left or right views.
        drop_left, drop_right = False, False
        if self.view_dropout > 0.0 and training:
            drop_one_view = torch.rand(1) < self.view_dropout
            drop_left = drop_one_view and torch.rand(1) < 0.5
            drop_right = (drop_one_view and not drop_left) or "right_video" not in batch
        else:
            drop_right = "right_video" not in batch

        # if necessary, reduce the fps to configs.video_fps,
        # which is a divisor of configs.output_fps
        # though always include the last frame
        relative_fps = self.configs.output_fps // self.configs.video_fps
        assert relative_fps > 0, "Video FPS must be a divisor of the output FPS"
        original_video_length = left.shape[1]
        video_indices = torch.arange(left.shape[1] - 1, 0, -relative_fps).long()
        video_indices = torch.flip(video_indices, dims=[0])
        left = left[:, video_indices]
        right = right[:, video_indices]

        batch_size = left.shape[0]
        left = left.flatten(0, 1)
        right = right.flatten(0, 1)

        right_features = self._forward_single_video(right, drop_right, training)
        left_features = self._forward_single_video(left, drop_left, training)

        gps_backbone_dtype = self.gps_backbone.parameters().__next__().dtype
        left_features = left_features.view(batch_size, -1, left_features.shape[-1]).to(
            gps_backbone_dtype
        )
        right_features = right_features.view(
            batch_size, -1, right_features.shape[-1]
        ).to(gps_backbone_dtype)

        # Now, we will create a zeros tensor with the same shape as the original
        # video, and fill it with the features.
        # This is necessary because the video features are aligned with the
        # motion dynamics, which are padded with zeros in the temporal dimension.
        # We will use the original video indices to fill the features.
        full_zero_features = torch.zeros(
            batch_size,
            original_video_length,
            left_features.shape[-1],
            device=left_features.device,
        )
        full_zero_features[:, video_indices] = left_features
        left_features = full_zero_features

        full_zero_features = torch.zeros(
            batch_size,
            original_video_length,
            right_features.shape[-1],
            device=right_features.device,
        )
        full_zero_features[:, video_indices] = right_features
        right_features = full_zero_features

        return left_features, right_features

    def _forward_single_video(self, video, drop: bool, training: bool):
        if drop and training:
            video_features = torch.zeros(
                video.shape[0],
                self.configs.image_embedding_size,
                dtype=video.dtype,
                device=video.device,
            )
        else:
            gps_backbone_dtype = self.gps_backbone.parameters().__next__().dtype
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
                video.shape[0], self.configs.image_embedding_size
            )

        return video_features

    def _forward_gaze_video(self, batch, training: bool):
        if training is None:
            training = self.training

        video_features = None
        video = batch["front_video"]

        # if necessary, reduce the fps to configs.gaze_fps,
        # which is a divisor of configs.output_fps
        # though always include the last frame
        relative_fps = self.configs.output_fps // self.configs.gaze_fps
        assert relative_fps > 0, "Gaze FPS must be a divisor of the output FPS"
        original_video_length = video.shape[1]
        video_indices = torch.arange(video.shape[1] - 1, 0, -relative_fps).long()
        video_indices = torch.flip(video_indices, dims=[0])
        video = video[:, video_indices]

        batch_size = video.shape[0]
        video = video.flatten(0, 1)
        video_features = self._forward_single_video(video, False, training)

        gps_backbone_dtype = self.gps_backbone.parameters().__next__().dtype
        video_features = video_features.view(
            batch_size, -1, video_features.shape[-1]
        ).to(gps_backbone_dtype)

        # Now, we will create a zeros tensor with the same shape as the original
        # video, and fill it with the features.
        # This is necessary because the video features are aligned with the
        # motion dynamics, which are padded with zeros in the temporal dimension.
        # We will use the original video indices to fill the features.
        full_zero_features = torch.zeros(
            batch_size,
            original_video_length,
            video_features.shape[-1],
            device=video_features.device,
        )
        full_zero_features[:, video_indices] = video_features
        video_features = full_zero_features

        return video_features
