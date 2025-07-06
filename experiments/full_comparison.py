"""Demonstrate GEMDataset class by fitting a very simple model."""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

if "SLURM_NTASKS" in os.environ:
    # Remove SLURM env variables to avoid issues with Lightning
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]

import lightning as L
import numpy as np
import torch
import torch.optim as optim
from autobots.autobots import AutoBotAdapted
from gimo.adapted_gimo import AdaptedGIMO
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from multimodal_transformer.multimodal_transformer import MultiModalTransformer
from torch.utils.data import DataLoader

from routeformer import Routeformer, set_logger_config
from routeformer.io.dataset import GEMDataset, Item
from routeformer.io.dataset_dreyeve import DreyeveDataset
from routeformer.losses.future_discounted_mse import FutureDiscountedLoss
from routeformer.models import RouteformerConfig
from routeformer.models.gps_backbone import (
    GPSBackboneConfig,
    Informer,
    LinearBaseline,
    PatchTST,
    StationaryBaseline,
    Transformer,
)
from routeformer.models.gps_backbone.config import PatchTSTBackboneConfig
from routeformer.models.gps_backbone.Linear import DLinear, NLinear
from routeformer.models.video_backbone import (  # DinoV2,; Sam,;; InverseForm,
    InverseFormBackboneConfig,
    SwinV2,
    TimmBackboneConfig,
)
from routeformer.optimizers import LinearWarmupCosineAnnealingLR
from routeformer.score import ade, fde

torch.set_float32_matmul_precision("medium")

PROJECT_DIR = os.getenv("PROJECT_DIR")

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
SLUMR_NODES = os.environ.get("SLURM_NODELIST")
DISCOUNTED_FACTOR = os.getenv("DISCOUNTED_FACTOR", "default")
ENABLE_LEFT_VIDEO_SPLIT = os.getenv("ENABLE_LEFT_VIDEO_SPLIT", "1") == "1"
PREDICT_FROM_LINEAR = os.getenv("PREDICT_FROM_LINEAR", "0") == "1"
LIMIT_TRAIN_BATCHES = float(os.getenv("LIMIT_TRAIN_BATCHES", 1))


if DISCOUNTED_FACTOR == "default":
    DISCOUNTED_FACTOR = {
        0: 0.97,
        # 100: 0.98,
        # 200: 0.99,
    }
else:
    DISCOUNTED_FACTOR = {
        0: 1,
    }

DATASET: Literal["DREYEVE", "Routeformer"] = os.getenv("DATASET", "DREYEVE")
DEBUG = os.getenv("DEBUG", "0") == "1"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).parent))
LOGS_DIR = RESULTS_DIR / "logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_COUNT = torch.cuda.device_count() if DEVICE == "cuda" else 1
EPOCHS = 200
INPUT_LENGTH_SECONDS = 8
TARGET_LENGTH_SECONDS = 6
MIN_PCI = 20
OUTPUT_FPS = 5
VIDEO_FPS = 1
GAZE_FPS = 1
BATCH_SIZE = 16 // DEVICE_COUNT if not DEBUG else 1

USE_PATCHTST_BACKBONE = False

assert DATASET in ["DREYEVE", "Routeformer"], "DATASET must be one of DREYEVE, Routeformer"

USE_CACHE = True
USE_DATASET_CACHE = USE_CACHE
TORCHCACHE_CACHE_DIR = Path(os.getenv("TORCHCACHE_CACHE_DIR"))
ENABLE_PCI_SPLIT = False
NUM_WORKERS = os.getenv("NUM_WORKERS", 24)
VAL_WORKER_RATIO = 3
if DATASET == "DREYEVE":
    STEP_SIZE_SECONDS = 2
    USE_MEMORY_CACHE = False
    TRAIN_MEMORY_CACHE_SIZE = 180e9 // NUM_WORKERS if USE_MEMORY_CACHE else 0
    VAL_MEMORY_CACHE_SIZE = 60e9 // NUM_WORKERS // VAL_WORKER_RATIO if USE_MEMORY_CACHE else 0
    ENABLE_PCI_SPLIT = (
        os.getenv("ENABLE_PCI_SPLIT", "0") == "1"
    )  # Will prepare for pci balanced split
    PCI_SPLIT_N_SAMPLES_PER_BIN = int(
        os.getenv("PCI_SPLIT_N_SAMPLES_PER_BIN", 200)
    )  # Will prepare for pci balanced split
    GOPRO_SCALING_FACTOR = (
        0.4  # from 1.0, (1920, 1080) -> (768, 432), but we split it into two -> (384, 432)  # noqa
    )
    FRONT_SCALING_FACTOR = 1 / 3.0  # from 1.0, (960, 720) -> (320, 240)
    DATASET_DIR = Path(os.getenv("DREYEVE_DATASET_DIR"))
    DATASET_CACHE_DIR = Path(os.getenv("DREYEVE_DATASET_CACHE_DIR"))
    MOTION_MEAN, MOTION_STD = 4.7068373500451, 2.722694545590219
    IRR_QUARTILES = {
        "25%": 26.79,
        "50%": 36.33,
        "75%": 50.77,
        "95%": 78.02,
    }
else:
    NUM_WORKERS = int(NUM_WORKERS * 0.75)
    DATASET_NUM_WORKERS = 0
    STEP_SIZE_SECONDS = 2
    GOPRO_SCALING_FACTOR = 0.1  # from 1.0, (3840, 2160) -> (384, 216)
    FRONT_SCALING_FACTOR = 0.3  # from 1.0, (1088, 1080) -> (326, 324)
    DATASET_DIR = Path(os.getenv("ROUTEFORMER_DATASET_DIR"))
    DATASET_CACHE_DIR = Path(os.getenv("ROUTEFORMER_DATASET_CACHE_DIR"))
    MOTION_MEAN, MOTION_STD = 1.8332362885457094, 0.9090128501056961
    IRR_QUARTILES = {
        "25%": 24.84,
        "50%": 31.27,
        "75%": 41.19,
        "95%": 62.55,
    }

if DEBUG:
    WANDB_MODE = "disabled"
    set_logger_config(logging.DEBUG)
else:
    WANDB_MODE = "online"
    set_logger_config(logging.ERROR)

DESCRIPTION = os.getenv("DESCRIPTION", "train_val")
PROJECT_NAME = f"{DATASET.lower()}_full_comparison"
EXPERIMENT_NAME = f"{DATASET.lower()}_full_{DESCRIPTION}_0.1denseloss_2enc_64_emb_1dec"

LOGS_DIR.mkdir(exist_ok=True, parents=True)
DATASET_DIR.mkdir(exist_ok=True, parents=True)
DATASET_CACHE_DIR.mkdir(exist_ok=True, parents=True)


class ParallelTrainer(L.LightningModule):
    """Train all candidate models in parallel.

    Ensures that all models are trained on the same data, same shuffle,
    and that the same optimizer is used for all models.
    """

    GPS_BACKBONE_CONFIG_PARAMETERS = dict(
        # Fixed parameters - dictated by dataset
        seq_len=INPUT_LENGTH_SECONDS * OUTPUT_FPS,
        label_len=INPUT_LENGTH_SECONDS * OUTPUT_FPS,
        pred_len=TARGET_LENGTH_SECONDS * OUTPUT_FPS,
        embed="timeF",
        freq="m",
        # Hyperparameters
        moving_avg=25,
        factor=4,
        distil=True,
        dropout=0.0,
        activation="relu",
        individual=False,
        d_model=832,
        n_heads=8,
        e_layers=6,
        d_layers=1,
        d_ff=832 * 4,
    )

    GPS_BACKBONE_CONFIG = GPSBackboneConfig(
        **GPS_BACKBONE_CONFIG_PARAMETERS,
    )

    LINEAR_BACKBONE_CONFIG = GPS_BACKBONE_CONFIG.override(
        kernel_size=25,
    )
    PATCHTST_BACKBONE_CONFIG = PatchTSTBackboneConfig(
        **GPS_BACKBONE_CONFIG_PARAMETERS,
        fc_dropout=0.1,
        head_dropout=0.0,
        patch_len_ratio=0.25,
        stride_ratio=0.125,
        padding_patch="end",
        revin=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
    )

    if USE_PATCHTST_BACKBONE:
        GPS_BACKBONE_CONFIG = PATCHTST_BACKBONE_CONFIG

    ROUTEFORMER_CONFIG = RouteformerConfig(
        gps_backbone_config=GPS_BACKBONE_CONFIG,
        lr=1e-5,
        wd=1e-4,
        discount_factor=DISCOUNTED_FACTOR,
        epsilon=1.0,
        visual_epsilon=0.3,
        optimizer="AdamW",
        batch_size=BATCH_SIZE,
        min_pci=MIN_PCI,
        step_size=STEP_SIZE_SECONDS,
        epochs=EPOCHS,
        output_fps=OUTPUT_FPS,
        gopro_scaling_factor=GOPRO_SCALING_FACTOR,
        front_scaling_factor=FRONT_SCALING_FACTOR,
        num_workers=NUM_WORKERS,
        use_cache=USE_CACHE,
        cache_dir=str(DATASET_CACHE_DIR),
        normalize_motion=False,
        rotate_motion=DATASET == "DREYEVE",
        motion_mean=MOTION_MEAN,
        motion_std=MOTION_STD,
        decoder_mode="smart",
    )

    SWINV2_BACKBONE_CONFIG = TimmBackboneConfig(
        model_type="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        torchcache_persistent_module_hash="01b965509f767bc93a4b77aa0b0ee8ebc020ac284aa16b268c06d879288781ed"  # noqa
        + "a"
        if DATASET == "DREYEVE"
        else "",  # noqa
        torchcache_max_memory_cache_size=100e9
        // DEVICE_COUNT,  # This is enough for SwinV2 base, for float16 cache dtype # noqa
        train_backbone=False,
        backbone_minibatch_size=32,
        cache_dir=str(TORCHCACHE_CACHE_DIR),
        pad_to_square=True,
    )
    SWINV2_BACKBONE_CONFIG = SWINV2_BACKBONE_CONFIG.override(
        torchcache_cache_dtype=torch.float16,
    )
    DINOV2_BACKBONE_CONFIG = SWINV2_BACKBONE_CONFIG.override(
        model_type="vit_base_patch14_dinov2.lvd142m",
        torchcache_persistent_module_hash="1d9cd461e0d50406283c6645639e6dbb4d4cb804430d759d88fe1fdb58ae889a",  # noqa
        torchcache_max_memory_cache_size=14e9
        // DEVICE_COUNT,  # we actually need 128GB for DinoV2 for float16 cache dtype, but we have limited RAM # noqa
    )
    SAM_BACKBONE_CONFIG = SWINV2_BACKBONE_CONFIG.override(
        model_type="samvit_base_patch16.sa1b",
        torchcache_persistent_module_hash="7dab742e83a23ddf2e4b52302c09bd8d67354341f4e4e5eda5a10006e8b08ba7",  # noqa
        torchcache_max_memory_cache_size=14e9
        // DEVICE_COUNT,  # we actually need 64GB for SAM for float16 cache dtype, but we have limited RAM # noqa
    )
    INVFORM_BACKBONE_CONFIG = InverseFormBackboneConfig(
        train_backbone=False,
        backbone_minibatch_size=BATCH_SIZE,
        download_model=True,
        model_path=TORCHCACHE_CACHE_DIR / "inverseform" / "hr16s_4k_slim.pth",
    )

    ROUTEFORMER_CONFIG_SWINV2 = ROUTEFORMER_CONFIG.override(
        video_backbone_config=SWINV2_BACKBONE_CONFIG,
        with_video=True,
        video_fps=VIDEO_FPS,
        gaze_fps=GAZE_FPS,
        dense_prediction=True,
        dense_loss_ratio=0.5,
        image_embedding_size=64,
        view_dropout=0.6,
        gaze_dropout=0.2,
        motion_noise=0.0,
        feature_dropout=0.05,
        encoder_hidden_size=64,
        encoder_heads=8,
        encoder_layers=8,
        encoder_d_ff=64 * 4,
        cross_modal_decoder_heads=8,
        cross_modal_decoder_layers=2,
    )

    ROUTEFORMER_CONFIG_DINOV2 = ROUTEFORMER_CONFIG_SWINV2.override(
        video_backbone_config=DINOV2_BACKBONE_CONFIG,
    )
    ROUTEFORMER_CONFIG_SAM = ROUTEFORMER_CONFIG_SWINV2.override(
        video_backbone_config=SAM_BACKBONE_CONFIG,
    )
    ROUTEFORMER_CONFIG_INVFORM = ROUTEFORMER_CONFIG_SWINV2.override(
        video_backbone_config=INVFORM_BACKBONE_CONFIG,
    )

    ROUTEFORMER_CONFIG_SWINV2_GAZE = ROUTEFORMER_CONFIG_SWINV2.override(
        with_gaze=True,
    )
    ROUTEFORMER_CONFIG_DINOV2_GAZE = ROUTEFORMER_CONFIG_DINOV2.override(
        with_gaze=True,
    )
    ROUTEFORMER_CONFIG_INVFORM_GAZE = ROUTEFORMER_CONFIG_INVFORM.override(
        with_gaze=True,
    )

    ROUTEFORMER_CONFIG_AUTOREG = ROUTEFORMER_CONFIG.override(
        autoregressive=True,
    )
    ROUTEFORMER_CONFIG_SWINV2_AUTOREG = ROUTEFORMER_CONFIG_SWINV2.override(
        autoregressive=True,
    )
    ROUTEFORMER_CONFIG_SWINV2_GAZE_AUTOREG = ROUTEFORMER_CONFIG_SWINV2_AUTOREG.override(
        with_gaze=True,
    )

    GIMO_CONFIG_SWINV2 = ROUTEFORMER_CONFIG_SWINV2_GAZE.override(
        dense_prediction=False,
    )
    GIMO_CONFIG_DINOV2 = ROUTEFORMER_CONFIG_DINOV2_GAZE.override(
        dense_prediction=False,
    )
    GIMO_CONFIG_INVFORM = ROUTEFORMER_CONFIG_INVFORM_GAZE.override(
        dense_prediction=False,
    )

    MULTIMODAL_TRANSFORMER_CONFIG_SWINV2 = ROUTEFORMER_CONFIG_SWINV2_GAZE.override(
        dense_prediction=False,
    )
    MULTIMODAL_TRANSFORMER_CONFIG_DINOV2 = ROUTEFORMER_CONFIG_DINOV2_GAZE.override(
        dense_prediction=False,
    )
    MULTIMODAL_TRANSFORMER_CONFIG_INVFORM = ROUTEFORMER_CONFIG_INVFORM_GAZE.override(
        dense_prediction=False,
    )

    ROUTEFORMER_CONFIG_SWINV2_GAZE_WOUT_SCENE = ROUTEFORMER_CONFIG_SWINV2_GAZE.override(
        with_scene=False,
        gaze_dropout=0.0,
    )

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.Routeformer_with_video_with_gaze_swinv2_autoreg_4s = Routeformer(
            gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
            video_backbone=SwinV2,
            configs=self.ROUTEFORMER_CONFIG_SWINV2_GAZE_AUTOREG.override(
                autoregressive_step_size=int(4 * OUTPUT_FPS),
            ),
        )

        self.Routeformer_with_video_with_gaze_swinv2_wout_scene = Routeformer(
            gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
            video_backbone=SwinV2,
            configs=self.ROUTEFORMER_CONFIG_SWINV2_GAZE_WOUT_SCENE,
        )

        self.Routeformer_with_video_with_gaze_swinv2 = Routeformer(
            gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
            video_backbone=SwinV2,
            configs=self.ROUTEFORMER_CONFIG_SWINV2_GAZE,
        )
        # self.Routeformer_with_video_with_gaze_dinov2 = Routeformer(
        #     gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
        #     video_backbone=DinoV2,
        #     configs=self.ROUTEFORMER_CONFIG_DINOV2_GAZE,
        # )
        # self.Routeformer_with_video_with_gaze_invform = Routeformer(
        #     gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
        #     video_backbone=InverseForm,
        #     configs=self.ROUTEFORMER_CONFIG_INVFORM_GAZE,
        # )

        self.AdaptedGIMO_swinv2 = AdaptedGIMO(
            video_backbone=SwinV2,
            configs=self.GIMO_CONFIG_SWINV2,
        )
        # self.AdaptedGIMO_dinov2 = AdaptedGIMO(
        #     video_backbone=DinoV2,
        #     configs=self.GIMO_CONFIG_DINOV2,
        # )
        # self.AdaptedGIMO_invform = AdaptedGIMO(
        #     video_backbone=InverseForm,
        #     configs=self.GIMO_CONFIG_INVFORM,
        # )

        self.MultiModalTransformer_swinv2 = MultiModalTransformer(
            video_backbone=SwinV2,
            configs=self.MULTIMODAL_TRANSFORMER_CONFIG_SWINV2,
        )
        # self.MultiModalTransformer_dinov2 = MultiModalTransformer(
        #     video_backbone=DinoV2,
        #     configs=self.MULTIMODAL_TRANSFORMER_CONFIG_DINOV2,
        # )
        # self.MultiModalTransformer_invform = MultiModalTransformer(
        #     video_backbone=InverseForm,
        #     configs=self.MULTIMODAL_TRANSFORMER_CONFIG_INVFORM,
        # )

        self.Routeformer_with_video_swinv2 = Routeformer(
            gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
            video_backbone=SwinV2,
            configs=self.ROUTEFORMER_CONFIG_SWINV2,
        )
        # self.Routeformer_with_video_dinov2 = Routeformer(
        #     gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
        #     video_backbone=DinoV2,
        #     configs=self.ROUTEFORMER_CONFIG_DINOV2,
        # )
        # self.Routeformer_with_video_sam = Routeformer(
        #     gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
        #     video_backbone=Sam,
        #     configs=self.ROUTEFORMER_CONFIG_SAM,
        # )
        # self.Routeformer_with_video_invform = Routeformer(
        #     gps_backbone=PatchTST if USE_PATCHTST_BACKBONE else Informer,
        #     video_backbone=InverseForm,
        #     configs=self.ROUTEFORMER_CONFIG_INVFORM,
        # )

        self.AutoBotEgo = AutoBotAdapted(
            configs=self.ROUTEFORMER_CONFIG,
        )

        self.Routeformer_without_video_informer = Routeformer(
            gps_backbone=Informer,
            configs=self.ROUTEFORMER_CONFIG,
        )
        self.Routeformer_without_video_patchtst = Routeformer(
            gps_backbone=PatchTST,
            configs=self.ROUTEFORMER_CONFIG.override(
                gps_backbone_config=self.PATCHTST_BACKBONE_CONFIG
            ),
        )
        self.Routeformer_without_video_transformer = Routeformer(
            gps_backbone=Transformer,
            configs=self.ROUTEFORMER_CONFIG,
        )
        self.Routeformer_without_video_dlinear = Routeformer(
            gps_backbone=DLinear,
            configs=self.ROUTEFORMER_CONFIG.override(
                gps_backbone_config=self.LINEAR_BACKBONE_CONFIG
            ),
        )
        self.Routeformer_without_video_nlinear = Routeformer(
            gps_backbone=NLinear,
            configs=self.ROUTEFORMER_CONFIG.override(
                gps_backbone_config=self.LINEAR_BACKBONE_CONFIG
            ),
        )

        self.stationary_baseline = Routeformer(
            gps_backbone=StationaryBaseline,
            configs=self.ROUTEFORMER_CONFIG,
        )
        self.linear_baseline = Routeformer(
            gps_backbone=LinearBaseline,
            configs=self.ROUTEFORMER_CONFIG,
        )

        self.models: dict[str, Routeformer] = {
            # ------------- Autoregressive models
            "Routeformer_with_video_with_gaze_swinv2_autoreg_4s": self.Routeformer_with_video_with_gaze_swinv2_autoreg_4s,  # noqa
            # ------------- Full Routeformer
            # "Routeformer_with_video_with_gaze_invform": self.Routeformer_with_video_with_gaze_invform,  # noqa
            "Routeformer_with_video_with_gaze_swinv2": self.Routeformer_with_video_with_gaze_swinv2,  # noqa
            # "Routeformer_with_video_with_gaze_dinov2": self.Routeformer_with_video_with_gaze_dinov2,  # noqa
            # ------------- Routeformer without scene
            "Routeformer_with_video_with_gaze_swinv2_wout_scene": self.Routeformer_with_video_with_gaze_swinv2_wout_scene,  # noqa
            # ------------- Adapted GIMO
            # "AdaptedGIMO_invform": self.AdaptedGIMO_invform,
            "AdaptedGIMO_swinv2": self.AdaptedGIMO_swinv2,
            # "AdaptedGIMO_dinov2": self.AdaptedGIMO_dinov2,
            # ------------- MultiModal Transformer
            # "MultiModalTransformer_invform": self.MultiModalTransformer_invform,
            "MultiModalTransformer_swinv2": self.MultiModalTransformer_swinv2,
            # "MultiModalTransformer_dinov2": self.MultiModalTransformer_dinov2,
            # ------------- Video-only models
            # "Routeformer_with_video_invform": self.Routeformer_with_video_invform,
            "Routeformer_with_video_swinv2": self.Routeformer_with_video_swinv2,
            # "Routeformer_with_video_dinov2": self.Routeformer_with_video_dinov2,
            # "Routeformer_with_video_sam": self.Routeformer_with_video_sam,
            # ------------- GPS-only models
            "AutoBotEgo": self.AutoBotEgo,
            "Routeformer_without_video_informer": self.Routeformer_without_video_informer,
            # "Routeformer_without_video_patchtst": self.Routeformer_without_video_patchtst,
            # "Routeformer_without_video_transformer": self.Routeformer_without_video_transformer,
            # "Routeformer_without_video_dlinear": self.Routeformer_without_video_dlinear,
            # "Routeformer_without_video_nlinear": self.Routeformer_without_video_nlinear,
            # ------------- Simple baselines
            "stationary_baseline": self.stationary_baseline,
            "linear_baseline": self.linear_baseline,
        }

        self.save_hyperparameters(
            {
                "ROUTEFORMER_CONFIG_SWINV2_GAZE": self.ROUTEFORMER_CONFIG_SWINV2_GAZE,
            }
        )

        self.trajectory_loss = FutureDiscountedLoss(
            self.ROUTEFORMER_CONFIG.discount_factor,
            self.ROUTEFORMER_CONFIG.epsilon,
            loss_function="smooth_l1",
        )
        self.dense_loss = FutureDiscountedLoss(
            self.ROUTEFORMER_CONFIG.discount_factor,
            self.ROUTEFORMER_CONFIG.visual_epsilon,
            loss_function="smooth_l1",
        )
        # # GIMO uses L1 loss
        # self.gimo_loss = torch.nn.L1Loss()
        # self.multimodal_transformer_loss = torch.nn.MSELoss()

        self.gimo_loss = FutureDiscountedLoss(
            self.ROUTEFORMER_CONFIG.discount_factor,
            self.ROUTEFORMER_CONFIG.epsilon,
            loss_function="smooth_l1",
        )
        self.multimodal_transformer_loss = FutureDiscountedLoss(
            self.ROUTEFORMER_CONFIG.discount_factor,
            self.ROUTEFORMER_CONFIG.epsilon,
            loss_function="smooth_l1",
        )

    def training_step(self, batch: Item, batch_idx):
        """Training step."""
        self.maybe_split_video(batch)
        total_loss = 0
        metrics = {}
        for model_name, model in self.models.items():
            input = batch["train"]
            target = batch["target"]
            target_gps = target["gps"].to(torch.float32)
            if "baseline" not in model_name:
                if model.configs.dense_prediction:
                    future_gps, future_visual_features = model(input)
                    _, target_visual_features = model.preprocess_batch(target, training=False)
                    target_visual_features = target_visual_features[
                        :, : future_visual_features.shape[1]
                    ]
                    autoreg_step_size = model.configs.autoregressive_step_size
                    if model.configs.autoregressive:
                        future_gps = future_gps[:, :autoreg_step_size]
                        target_gps = target_gps[:, :autoreg_step_size]
                    trajectory_loss = self.trajectory_loss(future_gps, target_gps)
                    if model.configs.autoregressive:
                        trajectory_loss = trajectory_loss * (
                            model.configs.gps_backbone_config.pred_len / autoreg_step_size
                        )
                    target_visual_features = target_visual_features.detach()
                    if model.configs.autoregressive:
                        future_visual_features = future_visual_features[:, :autoreg_step_size]
                        target_visual_features = target_visual_features[:, :autoreg_step_size]
                    dense_loss = self.dense_loss(future_visual_features, target_visual_features)
                    dense_loss_weight = (
                        model.configs.dense_loss_ratio * trajectory_loss / max(dense_loss, 1e-6)
                    ).detach()
                    # Activate dense loss after 10 epochs
                    if self.current_epoch < 10:
                        dense_loss_weight = 0
                    metrics[f"train_dense_loss_{model_name}"] = dense_loss
                    adjusted_dense_loss = dense_loss_weight * dense_loss
                    loss = trajectory_loss + adjusted_dense_loss
                else:
                    future_gps = model(input)
                    if "gimo" in model_name.lower():
                        trajectory_loss = self.gimo_loss(future_gps, target_gps)
                    elif "multimodal" in model_name.lower():
                        trajectory_loss = self.multimodal_transformer_loss(future_gps, target_gps)
                    else:
                        trajectory_loss = self.trajectory_loss(future_gps, target_gps)
                    loss = trajectory_loss
                total_loss += loss
                metrics[f"train_loss_{model_name}"] = trajectory_loss
                metrics[f"train_ade_{model_name}"] = ade(future_gps, target_gps)
                metrics[f"train_fde_{model_name}"] = fde(future_gps, target_gps)
        # Consider alternatives -> https://github.com/Lightning-AI/lightning/discussions/15266 # noqa
        # Also see -> https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out # noqa
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=BATCH_SIZE,
            rank_zero_only=True,
        )
        return total_loss

    def maybe_split_video(self, data: Item):
        """Split the video into left and right halves for dreyeve."""
        if DATASET == "DREYEVE" and ENABLE_LEFT_VIDEO_SPLIT:
            # now, create a right video with the rightmost 50% of the left video
            left_width = data["train"]["left_video"].shape[4]
            data["train"]["right_video"] = data["train"]["left_video"][
                :, :, :, :, int(0.5 * left_width) :
            ]
            data["target"]["right_video"] = data["target"]["left_video"][
                :, :, :, :, int(0.5 * left_width) :
            ]
            # and cut the left video to the leftmost 50%
            data["train"]["left_video"] = data["train"]["left_video"][
                :, :, :, : int(0.5 * left_width)
            ]
            data["target"]["left_video"] = data["target"]["left_video"][
                :, :, :, : int(0.5 * left_width)
            ]

    def validation_step(self, batch: Item, batch_idx):
        """Run validation step."""
        self.maybe_split_video(batch)
        return self._eval_and_log(batch, "val")

    def test_step(self, batch: Item, batch_idx):
        """Run test step."""
        self.maybe_split_video(batch)
        return self._eval_and_log(batch, "test")

    def report_split(self, prefix, metrics, buckets, losses, ades, fdes, irrs, final_suffix):
        """Report metrics for a given set of pci buckets."""
        avg_losses = []
        avg_ades = []
        avg_fdes = []
        for suffix, bucket in buckets.items():
            data_in_bucket = irrs[bucket]
            if len(data_in_bucket) > 0:
                bucket_losses = losses[bucket]
                bucket_ades = ades[bucket]
                bucket_fdes = fdes[bucket]
                avg_losses.append(bucket_losses.mean())
                avg_ades.append(bucket_ades.mean())
                avg_fdes.append(bucket_fdes.mean())
                metrics.update(
                    {
                        f"{prefix}_loss_{suffix}": bucket_losses.mean(),
                        f"{prefix}_ade_{suffix}": bucket_ades.mean(),
                        f"{prefix}_fde_{suffix}": bucket_fdes.mean(),
                    }
                )
            else:
                # log_dict is not happy when there are no values to log
                # This will cause slightly incorrect metrics to be reported
                # during training, but it's better than nothing, and it
                # will be the same for all models across trainings.
                avg_losses.append(0)
                avg_ades.append(0)
                avg_fdes.append(0)
                metrics.update(
                    {
                        f"{prefix}_loss_{suffix}": torch.tensor(0.0, device=DEVICE),
                        f"{prefix}_ade_{suffix}": torch.tensor(0.0, device=DEVICE),
                        f"{prefix}_fde_{suffix}": torch.tensor(0.0, device=DEVICE),
                    }
                )

        avg_losses = torch.tensor(avg_losses, device=DEVICE)
        avg_ades = torch.tensor(avg_ades, device=DEVICE)
        avg_fdes = torch.tensor(avg_fdes, device=DEVICE)
        metrics.update(
            {
                f"{prefix}_loss_{final_suffix}": avg_losses.mean(),
                f"{prefix}_ade_{final_suffix}": avg_ades.mean(),
                f"{prefix}_fde_{final_suffix}": avg_fdes.mean(),
            }
        )

    def _eval_and_log(self, batch: Item, split: str):
        """Run evaluation step."""
        metrics = {}
        for model_name, model in self.models.items():
            losses, ades, fdes = self._eval_step(model, batch)
            prefix = f"{split}_{model_name}"
            metrics.update(
                {
                    f"{prefix}_loss": losses.mean(),
                    f"{prefix}_ade": ades.mean(),
                    f"{prefix}_fde": fdes.mean(),
                }
            )
            # add the results to the relevant pci bucket
            # that is, one of <25%, 25-50%, 50-75%, 75-95%, >95%
            irrs = batch["pci"]
            buckets = {
                "<25%": irrs < IRR_QUARTILES["25%"],
                "25-50%": (irrs > IRR_QUARTILES["25%"]) & (irrs < IRR_QUARTILES["50%"]),
                "50-75%": (irrs > IRR_QUARTILES["50%"]) & (irrs < IRR_QUARTILES["75%"]),
                "75-95%": (irrs > IRR_QUARTILES["75%"]) & (irrs < IRR_QUARTILES["95%"]),
                ">95%": irrs >= IRR_QUARTILES["95%"],
            }
            self.report_split(prefix, metrics, buckets, losses, ades, fdes, irrs, "avg%")

            irr_buckets = {
                "<20i": irrs < 20,
                "20-40i": (irrs > 20) & (irrs < 40),
                "40-60i": (irrs > 40) & (irrs < 60),
                "60-80i": (irrs > 60) & (irrs < 80),
                ">80i": irrs >= 80,
            }

            self.report_split(prefix, metrics, irr_buckets, losses, ades, fdes, irrs, "avgi")
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=BATCH_SIZE,
        )
        return metrics

    def _eval_step(self, model: Routeformer, batch):
        torch.manual_seed(12345)
        input = batch["train"]
        target_gps = batch["target"]["gps"]
        intermediate_future_gps = []
        for _ in range(5):
            if model.configs.dense_prediction:
                future_gps, _ = model(input)
            else:
                future_gps = model(input)
            intermediate_future_gps.append(future_gps)
        future_gps = torch.stack(intermediate_future_gps).mean(dim=0)
        losses, ades, fdes = [], [], []
        for index in range(future_gps.shape[0]):
            fgps, tgps = future_gps[index : index + 1], target_gps[index : index + 1]
            loss = self.trajectory_loss(fgps, tgps)
            avg_disp_error = ade(fgps, tgps)
            final_disp_error = fde(fgps, tgps)
            losses.append(loss)
            ades.append(avg_disp_error)
            fdes.append(final_disp_error)
        torch.seed()
        losses = torch.stack(losses)
        ades = torch.stack(ades)
        fdes = torch.stack(fdes)
        return losses, ades, fdes

    def configure_optimizers(self):
        """Configure the optimizer and scheduler."""
        video_backbone_params = []
        if (
            self.ROUTEFORMER_CONFIG.video_backbone_config
            and self.ROUTEFORMER_CONFIG.video_backbone_config.train_backbone
        ):
            video_backbone_params = [
                p for name, p in self.named_parameters() if "video_backbone" in name
            ]
        others = [p for name, p in self.named_parameters() if "video_backbone" not in name]
        # see https://github.com/clovaai/AdamP/issues/10#issuecomment-861208964
        # optimizer = optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-1)
        OptClass = getattr(optim, self.ROUTEFORMER_CONFIG.optimizer)
        optimizer = OptClass(
            [
                {"params": others},
                {"params": video_backbone_params, "lr": 1e-6},
            ],
            lr=self.ROUTEFORMER_CONFIG.lr,
            weight_decay=self.ROUTEFORMER_CONFIG.wd,
        )
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=2,
                max_epochs=EPOCHS,
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    dataloaders = {}
    splits = ["train+val", "test"]
    for split in splits:
        if DATASET == "DREYEVE":
            dataset = DreyeveDataset(
                root_dir=DATASET_DIR,
                split=split,
                input_length=INPUT_LENGTH_SECONDS,
                target_length=TARGET_LENGTH_SECONDS,
                step_size=STEP_SIZE_SECONDS,
                min_pci=MIN_PCI if not split.startswith("train") else 0,
                output_fps=OUTPUT_FPS,
                gopro_scaling_factor=GOPRO_SCALING_FACTOR,
                front_scaling_factor=FRONT_SCALING_FACTOR,
                use_cache=USE_DATASET_CACHE,
                use_memory_cache=USE_MEMORY_CACHE,
                max_memory_cache_size=TRAIN_MEMORY_CACHE_SIZE
                if split.startswith("train")
                else VAL_MEMORY_CACHE_SIZE,
                cache_dir=DATASET_CACHE_DIR,
                enable_pci_split=ENABLE_PCI_SPLIT if split.startswith("train") else False,
                pci_split_n_samples_per_bin=PCI_SPLIT_N_SAMPLES_PER_BIN,  # noqa
                max_cache_size=200e9,
            )
        else:
            dataset = GEMDataset(
                DATASET_DIR,
                split=split,
                input_length=INPUT_LENGTH_SECONDS,
                target_length=TARGET_LENGTH_SECONDS,
                step_size=STEP_SIZE_SECONDS,
                min_pci=MIN_PCI if not split.startswith("train") else 0,
                output_fps=OUTPUT_FPS,
                gopro_scaling_factor=GOPRO_SCALING_FACTOR,
                front_scaling_factor=FRONT_SCALING_FACTOR,
                device=DEVICE,
                with_video=True,
                with_gaze=True,
                undistort_videos=True,
                use_cache=USE_CACHE,
                cache_dir=DATASET_CACHE_DIR,
                max_cache_size=300e9,
                num_workers=DATASET_NUM_WORKERS,
                with_gpu_codec=True,
            )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=split.startswith("train") and not ENABLE_PCI_SPLIT,
            pin_memory=True,
            num_workers=NUM_WORKERS // DEVICE_COUNT
            if split.startswith("train")
            else max(NUM_WORKERS // DEVICE_COUNT // VAL_WORKER_RATIO, 1),
            persistent_workers=NUM_WORKERS > 0,
        )

    wandb_logger = WandbLogger(
        name=EXPERIMENT_NAME,
        project=PROJECT_NAME,
        mode=WANDB_MODE,
        config={
            "MIN_PCI": MIN_PCI,
            "STEP_SIZE_SECONDS": STEP_SIZE_SECONDS,
            "OUTPUT_FPS": OUTPUT_FPS,
            "NUM_WORKERS": NUM_WORKERS,
            "ENABLE_PCI_SPLIT": ENABLE_PCI_SPLIT,
            "BATCH_SIZE": BATCH_SIZE,
            "DATASET": DATASET,
            "NUM_GPUS": torch.cuda.device_count(),
            "SLURM_JOB_ID": SLURM_JOB_ID,
            "SLURM_NODELIST": SLUMR_NODES,
            "PREDICT_FROM_LINEAR": PREDICT_FROM_LINEAR,
            "LIMIT_TRAIN_BATCHES": LIMIT_TRAIN_BATCHES,
        },
        save_dir=LOGS_DIR,
    )

    parallel_trainer = ParallelTrainer()
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend="nccl")
    checkpoints = []
    for model_name, model in parallel_trainer.models.items():
        if (
            "baseline" in model_name
            or "Routeformer_with_video_with_gaze" not in model_name
            or "wout_scene" in model_name
        ):
            continue
        checkpoint_dir = DATASET_CACHE_DIR / "checkpoints" / EXPERIMENT_NAME / model_name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        score_name = f"val_{model_name}_ade"
        file_name = f"{EXPERIMENT_NAME}-{{epoch:02d}}-{{{score_name}:.2f}}_"
        # prepend file name with training start time up to minutes
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-") + file_name
        checkpoints.extend(
            [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=file_name,
                    monitor=score_name,
                    mode="min",
                    save_top_k=1,
                    save_weights_only=False,
                    every_n_epochs=1,
                ),
            ]
        )
    callbacks = checkpoints
    # callbacks.append(
    #     EarlyStopping(
    #         monitor="val_Routeformer_with_video_with_gaze_swinv2_ade",
    #         patience=20,
    #         mode="min",
    #     )
    # )
    trainer = L.Trainer(
        default_root_dir=LOGS_DIR,
        accelerator="gpu" if DEVICE == "cuda" else "auto",
        devices="auto",
        max_epochs=EPOCHS,
        callbacks=callbacks,
        gradient_clip_algorithm="norm",
        gradient_clip_val=2.5,
        num_sanity_val_steps=1,
        strategy=strategy if not DEBUG and DEVICE_COUNT > 1 else "auto",
        logger=wandb_logger,
        check_val_every_n_epoch=2,
        max_steps=10 if DEBUG else -1,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
    )
    trainer.fit(
        parallel_trainer,
        train_dataloaders=dataloaders[splits[0]],
        val_dataloaders=dataloaders[splits[1]],
    )
