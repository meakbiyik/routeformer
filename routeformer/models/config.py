"""Routeformer config definition."""
from dataclasses import dataclass, field
from typing import Literal

from routeformer.models.gps_backbone import GPSBackboneConfig
from routeformer.models.video_backbone import VideoBackboneConfig
from routeformer.utils.config import BaseConfig


@dataclass
class RouteformerConfig(BaseConfig):
    """Config for Routeformer."""

    gps_backbone_config: GPSBackboneConfig
    video_backbone_config: VideoBackboneConfig = None
    # Whether to output the attention of the backbone
    output_attention: bool = False
    # Whether to use video, by default None. If None, it is set to True
    # if video_backbone_config is not None in Routeformer.
    with_video: bool = None
    # Whether to use gaze, by default False.
    with_gaze: bool = False
    with_scene: bool = True
    # Future discount factor, with a dict that changes the discount
    # factor every time the epoch is in the dict's keys.
    # Weighs the future predictions less than the current prediction,
    # with factor discount_factor^i for the i-th point
    discount_factor: dict = field(default_factory=lambda: {0: 0.9})
    # Whether to sum the ground truth with the future predictions
    decoder_mode: Literal["vanilla", "recursive", "smart"] = "vanilla"
    rotate_motion: bool = False
    # Loss epsilon, used in epsilon-insensitive loss
    loss_function: Literal["mse", "mae", "smooth_l1"] = "smooth_l1"
    epsilon: float = None
    visual_epsilon: float = None
    # Predict autoregressively per step size
    autoregressive: bool = False
    autoregressive_step_size: int = 1
    # Whether to predict densely: do not just predict the next GPS dynamics,
    # but predict the video and gaze features for the next frames if available.
    dense_prediction: bool = False
    dense_loss_ratio: float = 0.25
    # FPS of the video to use, must be a divisor of the output_fps
    video_fps: int = 1
    # FPS of the gaze to use, must be a divisor of the output_fps
    gaze_fps: int = 1
    # Encoder parameters for motion, video and gaze
    encoder_hidden_size: int = 64
    encoder_heads: int = 8
    encoder_layers: int = 2
    encoder_d_ff: int = 64
    cross_modal_decoder_heads: int = 8
    cross_modal_decoder_layers: int = 1
    # whether to normalize the motion
    normalize_motion: bool = False
    motion_mean: float = 0.0
    motion_std: float = 1.0
    # Added noise to the positions
    motion_noise: float = 0.0
    # GoPro image feature refiner params
    view_dropout: float = 0.0
    gaze_dropout: float = 0.0
    feature_dropout: float = 0.0
    image_embedding_size: int = 128
    # Training params, not used in model
    lr: float = 5e-4
    wd: float = 0
    optimizer: str = "Adam"
    batch_size: int = 32
    min_pci: float = 0.0
    step_size: int = 1
    epochs: int = 100
    output_fps: int = 5
    gopro_scaling_factor: float = 1.0
    front_scaling_factor: float = 1.0
    num_workers: int = 0
    use_cache: bool = False
    cache_dir: str = None

    # Rebuttal params
    _only_motion: bool = False

    def __post_init__(self, **kwargs):
        """Initialize RouteformerConfig."""
        # Ensure that the video fps is a divisor of the output fps
        assert (
            self.output_fps % self.video_fps == 0
        ), "Video FPS must be a divisor of the output FPS"
        assert self.output_fps % self.gaze_fps == 0, "Gaze FPS must be a divisor of the output FPS"
        self.with_video = (
            self.with_video
            if self.with_video is not None
            else self.video_backbone_config is not None
        )
        if self.with_gaze:
            assert self.with_video, "Gaze backbone requires video backbone to be used"
        # Set the child gps configs' related attributes
        self.gps_backbone_config.output_attention = self.output_attention
        self.gps_backbone_config.with_video = self.with_video
        self.gps_backbone_config.with_gaze = self.with_gaze
        self.gps_backbone_config.dense_prediction = self.dense_prediction
        self.gps_backbone_config.image_embedding_size = self.image_embedding_size
        self.gps_backbone_config.encoder_hidden_size = self.encoder_hidden_size
        self.gps_backbone_config.output_fps = self.output_fps
        self.gps_backbone_config.dense_loss_ratio = self.dense_loss_ratio
        self.gps_backbone_config.discount_factor = self.discount_factor
        self.gps_backbone_config.smart_decoder = self.decoder_mode == "smart"
