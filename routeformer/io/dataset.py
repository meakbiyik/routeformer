"""Dataloader for Routeformer dataset."""
import atexit
import datetime
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import pickle
import platform
import re
import shutil
import subprocess
import tempfile
import warnings
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, TypedDict, Union

import av
import cv2
import kornia.feature as KF
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zstd
from csaps import csaps
from gopro2gpx import fourCC, gpmf, gpshelper
from pyproj import Transformer
from scipy import interpolate
from torchvision.io.video import _read_from_stream

from routeformer.io.file_methods import load_object, load_pldata_file
from routeformer.io.gaze import detect_fixations
from routeformer.io.image_stitcher import ImageStitcher
from routeformer.score import estimate_pci

logger = logging.getLogger(__name__)


class Data(TypedDict):
    """GoPro and EyeTracker data for a single step."""

    stitched_video: torch.Tensor
    left_video: torch.Tensor
    right_video: torch.Tensor
    left_audio: torch.Tensor
    right_audio: torch.Tensor
    gps: torch.Tensor
    front_video: torch.Tensor
    front_audio: torch.Tensor
    gaze: torch.Tensor


class Item(TypedDict):
    """Item outputted by the dataset."""

    train: Data
    target: Data
    pci: float


class _EyeTrackerFront(TypedDict):
    """EyeTracker front video data."""

    video: Path
    intrinsics: Path
    time: Path


class _EyeTrackerGaze(TypedDict):
    """EyeTracker gaze data."""

    gaze: Path
    time: Path


def _set_num_workers(func):
    """Set the number of threads/workers as decorator.

    Cannot be used to decorate a generator method (e.g., __iter__)
    """

    def wrapper(self: "GEMDataset", *args, **kwargs):
        logger.debug("Setting number of threads to %d", self.num_workers)
        prev_num_workers = torch.get_num_threads()
        torch.set_num_threads(self.num_workers)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.set_num_threads(prev_num_workers)

    return wrapper


class GEMDataset(torch.utils.data.Dataset):
    """Routeformer Dataset.

    Data is in the following format in the root directory (leaf nodes
    are real examples from the dataset):

    - root
        - 01GoPro
            - 001
                - left
                    - GH010008.MP4
                    - GH010008_1635783105983.mp4
                    - GH020008.MP4
                    - GH020008_1635783105983.mp4
                - right
                    - GH010009.MP4
                    - GH010009_1636279816768.mp4
                    - GH020009.MP4
                    - GH020009_1636279816768.mp4
                    - GH030009.MP4
                    - GH030009_1636279816768.mp4
                    - GH040009_1636279619167.mp4
            - 002
                - ...
            - ...
        - 02EyeTracker
            - 001
                - PI left v1_sae_log_1.bin.bin
                - PI right v1_sae_log_1.bin.bin
                - event.time
                - event.txt
                - event_timestamps.npy
                - extimu ps1.raw
                - extimu ps1.time
                - extimu ps1_timestamps.npy
                - eye0.intrinsics
                - eye0.mp4
                - eye0.time
                - eye0_lookup.npy
                - eye0_timestamps.npy
                - eye1.intrinsics
                - eye1.mp4
                - eye1.time
                - eye1_lookup.npy
                - eye1_timestamps.npy
                - gaze ps1.raw
                - gaze ps1.time
                - gaze ps1_timestamps.npy
                - gaze.pldata
                - gaze_200hz.raw
                - gaze_200hz.time
                - gaze_200hz_timestamps.npy
                - gaze_right ps1.raw
                - gaze_right ps1.time
                - gaze_right ps1_timestamps.npy
                - gaze_timestamps.npy
                - info.invisible.json
                - info.player.json
                - square_marker_cache
                - surface_definitions_v01
                - template.json
                - wearer.json
                - world.intrinsics
                - world.mp4
                - world.time
                - world_lookup.npy
                - world_timestamps.npy
                - worn ps1.raw
                - worn_200hz.raw
            - 002
                - ...
            - ...
        - 03CorrectedGPS
            - 001
                - GH010008_PD1.csv
                - GH020008_2.csv
                - GH030008_3_L.csv


    The tasks of this class are to:
    - Align left and right video feeds in 01GoPro by finding the files that are
        concurrently recorded
    - Extract GPS data from both video feeds
    - Align these two data sources by time, and interpolate the GPS data to match
        the video frames
    - Align EyeTracker data with the video frames
    - Return a dictionary of the above data for each video frame
    """

    GPS_STREAM_HANDLER = "GoPro MET"
    VIDEO_FPS = 30
    AUDIO_FPS = 48000
    GAZE_FPS = 200
    # For some reason, the gaze data for subject 009 & 010 is at 76 FPS
    ALTERNATIVE_GAZE_FPS = 76
    GAZE_RESOLUTION = 1088, 1080
    LEFT_VIDEO_CAMERA_INTRINSICS = np.array(
        [
            [1710.426021931798, 0, 1884.2289110824929],
            [0, 836.09803935562263, 1176.4416598639007],
            [0, 0, 1],
        ]
    )
    LEFT_VIDEO_DISTORTION_COEFFICIENTS = np.array(
        [
            -0.031747058681490734,
            0.0030000759331449784,
            0.044056989783113468,
            -0.0026995745434254055,
        ]
    )
    RIGHT_VIDEO_CAMERA_INTRINSICS = np.array(
        [
            [1710.426021931798, 0, 1884.2289110824929],
            [0, 836.09803935562263, 1176.4416598639007],
            [0, 0, 1],
        ]
    )
    RIGHT_VIDEO_DISTORTION_COEFFICIENTS = np.array(
        [
            -0.031747058681490734,
            0.0030000759331449784,
            0.044056989783113468,
            -0.0026995745434254055,
        ]
    )

    DATA_SPLIT = {
        "train": [
            "001",
            "003",
            "005",
            "006",
            "007",
            "010",
        ],
        "val": [
            "002",
            "004",
        ],
        "train+val": [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "010",
        ],
        "test": [
            "008",
            "009",
        ],
    }

    def __init__(
        self,
        root: str | Path = "/data/routeformer",
        split: Literal["train", "val", "train+val", "test"] | List[str] = "train",
        input_length: float = 8,
        target_length: float = 6,
        step_size: float = 2,
        avoid_overlap: bool = False,
        min_pci: float = 20.0,
        max_pci: float = None,
        output_fps: float = 5,
        crop_videos: bool = True,
        undistort_videos: bool = True,
        stitch_videos: bool = False,
        gopro_scaling_factor: float = 1.0,
        front_scaling_factor: float = 1.0,
        frame_transform: Optional[Callable] = None,
        video_transform: Optional[Callable] = None,
        output_format: str = "TCHW",
        num_workers: int = 1,
        with_video: bool = True,
        with_audio: bool = False,
        with_gaze: bool = True,
        mask_nonfixations: bool = False,
        dilution_threshold: float = 500.0,
        use_cache: bool = False,
        cache_dir: str | Path | None = None,
        max_cache_size: int = int(10e9),
        device: str = "cpu",
        with_gpu_codec: bool = False,
    ):
        """Initialize GEMDataset.

        Parameters
        ----------
        root : str
            Root directory of the dataset.
        split : str or list of str
            Split of the dataset, one of "train", "val", "train+val" or
            "test", or a list of subject IDs ("001",...,"009"), by default "train"
        input_length : int, optional
            Number of seconds of data to provide as input, by default 8
        target_length : int, optional
            Number of seconds of data to predict, by default 6
        step_size : float, optional
            Number of seconds to step between each sample, by default 2
        avoid_overlap : bool, optional
            Whether to avoid overlapping samples, by default True. The behavior of
            this parameter is as follows:
            - If True and step_size is equal or larger than input_length,
                then this parameter is ignored.
            - If True, and one of min_pci or max_pci is
                specified, then the first sample that satisfies the pci
                constraints is selected, and the parsin continues after the input
                length of that sample.
            - If True, and neither min_pci nor max_pci is
                specified, a warning is raised and the behavior is the same as if
                avoid_overlap was False.
            - If False (default), then overlapping samples are allowed, and the parsing
                continues after the step size.
        min_pci : float, optional
            Minimum pci of the trajectories to use, by default 20.0
        max_pci : float, optional
            Maximum pci of the trajectories to use, by default None
        output_fps : float, optional
            Number of frames per second to output, one of 1, 2, 3, 5, 10, 15 or 30,
            by default 5
        crop_videos : bool, optional
            Whether to crop the unnecessary parts from GoPro videos, by default True
        undistort_videos : bool, optional
            Whether to undistort the video frames, by default False
        stitch_videos : bool, optional
            Whether to stitch all videos together, by default False
        gopro_scaling_factor : float, optional
            Scaling factor to apply to the GoPro video frames, by default 1.0
        front_scaling_factor : float, optional
            Scaling factor to apply to the frontal video from gaze, by default 1.0
        frame_transform : Optional[Callable], optional
            Transform to apply to each video frame, by default None. Must return
            a tensor with the same shape as the input, or broadcastable to the
            same shape.
        video_transform : Optional[Callable], optional
            Transform to apply to the whole video, by default None
        output_format : str, optional
            Format of the output video, either "THWC" or "TCHW", by default "TCHW"
        num_workers : int, optional
            Number of threads/workers to use for loading data, by default 1.
        with_video : bool, optional
            Whether to include video data in the output, by default True
        with_audio : bool, optional
            Whether to include audio data in the output, by default False
        with_gaze : bool, optional
            Whether to include gaze data in the output, by default True
        mask_nonfixations : bool, optional
            Whether to mask non-fixation data in the gaze data, by default False.
            If True, then the gaze data is masked with -1s for non-fixation data.
        dilution_threshold : float, optional
            Maximum dilution of precision to include in the GPS data, by default 500.0
        use_cache : bool, optional
            Whether to cache the GPS data, by default False
        cache_dir : str or Path, optional
            Directory to use for caching, by default None. If None, then a temporary
            directory is used.
        max_cache_size : int, optional
            Maximum size of the cache in bytes, by default 10e9 (10 GB)
        device : str, optional
            Device to load the data on, by default "cpu"
        with_gpu_codec : bool, optional
            Whether to use GPU-accelerated video decoding, by default False
        """
        self.root: Path = Path(root)
        self.split = split if isinstance(split, list) else self.DATA_SPLIT[split]
        self.input_length = input_length
        self.target_length = target_length
        self.step_size = step_size
        self.avoid_overlap = avoid_overlap
        self.min_pci = min_pci
        self.max_pci = max_pci
        self.output_fps = output_fps
        self.crop_videos = crop_videos
        self.undistort_videos = undistort_videos
        self.stitch_videos = stitch_videos
        self.gopro_scaling_factor = gopro_scaling_factor
        self.front_scaling_factor = front_scaling_factor
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.num_workers = num_workers
        self.with_video = with_video
        self.with_audio = with_audio
        self.with_gaze = with_gaze
        self.mask_nonfixations = mask_nonfixations
        self.dilution_threshold = dilution_threshold
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) / "routeformer_dataset" if cache_dir is not None else None
        self.max_cache_size = max_cache_size
        self.device = device
        self.with_gpu_codec = with_gpu_codec

        if self.use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.output_format = output_format.upper()
        if self.output_format not in ("THWC", "TCHW"):
            logger.error(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
            raise ValueError(
                f"output_format should be either 'THWC' or 'TCHW', got {output_format}."
            )

        if self.output_fps not in (1, 2, 3, 5, 10, 15, 30):
            logger.error(
                "output_fps should be one of 1, 2, 3, 5, 10, 15 or 30, " f"got {output_fps}."
            )
            raise ValueError(
                "output_fps should be one of 1, 2, 3, 5, 10, 15 or 30, " f"got {output_fps}."
            )

        if (
            self.avoid_overlap
            and self.step_size <= self.input_length
            and (self.min_pci is None and self.max_pci is None)
        ):
            warnings.warn(
                "avoid_overlap is True, but step_size is equal or "
                "larger than input_length. avoid_overlap will be ignored."
            )
            logger.warning(
                "avoid_overlap is True, but step_size is equal or "
                "larger than input_length. avoid_overlap will be ignored."
            )
            self.avoid_overlap = False

        if self.num_workers == 0:
            self.num_workers = 1

        if self.stitch_videos:
            self._initialize_stitcher()

        self.input_video_frame_count = int(self.input_length * self.output_fps)
        self.target_video_frame_count = int(self.target_length * self.output_fps)
        self.input_audio_frame_count = int(self.input_length * self.AUDIO_FPS)
        self.target_audio_frame_count = int(self.target_length * self.AUDIO_FPS)
        self.input_gaze_frame_count = int(self.input_length * self.GAZE_FPS)
        self.target_gaze_frame_count = int(self.target_length * self.GAZE_FPS)
        self.alternative_input_gaze_frame_count = int(self.input_length * self.ALTERNATIVE_GAZE_FPS)
        self.alternative_target_gaze_frame_count = int(
            self.target_length * self.ALTERNATIVE_GAZE_FPS
        )

        logger.info(f"Dataset root: {self.root}")
        logger.debug(f"Input length: {self.input_length}")
        logger.debug(f"Target length: {self.target_length}")
        logger.debug(f"Input frame count: {self.input_video_frame_count}")
        logger.debug(f"Target frame count: {self.target_video_frame_count}")
        logger.debug(f"Input audio frame count: {self.input_audio_frame_count}")
        logger.debug(f"Target audio frame count: {self.target_audio_frame_count}")
        logger.debug(f"Input gaze frame count: {self.input_gaze_frame_count}")
        logger.debug(f"Target gaze frame count: {self.target_gaze_frame_count}")
        logger.debug(f"Step size: {self.step_size}")
        logger.debug(f"Min pci: {self.min_pci}")
        logger.debug(f"Max pci: {self.max_pci}")
        logger.debug(f"Output FPS: {self.output_fps}")
        logger.debug(f"Crop videos: {self.crop_videos}")
        logger.debug(f"Undistort videos: {self.undistort_videos}")
        logger.debug(f"Stitch videos: {self.stitch_videos}")
        logger.debug(f"GoPro scaling factor: {self.gopro_scaling_factor}")
        logger.debug(f"World scaling factor: {self.front_scaling_factor}")
        logger.debug(f"Frame transform: {self.frame_transform}")
        logger.debug(f"Video transform: {self.video_transform}")
        logger.debug(f"Output format: {self.output_format}")
        logger.debug(f"Number of workers: {self.num_workers}")
        logger.debug(f"Include video: {self.with_video}")
        logger.debug(f"Include audio: {self.with_audio}")
        logger.debug(f"Include gaze: {self.with_gaze}")
        logger.debug(f"Output fixations: {self.mask_nonfixations}")
        logger.debug(f"Dilution threshold: {self.dilution_threshold}")
        logger.debug(f"Cache: {self.use_cache}")
        logger.debug(f"Cache directory: {self.cache_dir}")
        logger.debug(f"Maximum cache size: {self.max_cache_size}")
        logger.debug(f"Device: {self.device}")

        # Gather subjects
        self.subjects = self._gather_subjects()

        logger.info(f"Number of subjects: {len(self.subjects)}")
        logger.debug(f"Subjects: {self.subjects}")

        # Filter subjects according to the split
        self.subjects = [s for s in self.subjects if s in self.split]

        logger.info(f"Number of subjects in split: {len(self.subjects)}")
        logger.debug(f"Subjects in split: {self.subjects}")

        if len(self.subjects) != len(self.split):
            logger.warning(
                f"Number of subjects in split ({len(self.subjects)}) "
                f"does not match the desired number ({len(self.split)})."
            )
            logger.info(f"Subjects in split: {self.subjects}")
            logger.info(f"Expected: {self.split}")

        # For 01GoPro, gather the left and right video files and match
        # them for each subject
        self.left_samples, self.right_samples = self._gather_gopro_samples()
        # Gather the EyeTracker data for each subject
        self.video_samples, self.gaze_samples = self._gather_eyetracker_samples()
        # Gather the corrected GPS data for each subject
        self.corrected_gps_samples = self._gather_corrected_gps_samples()

        logger.debug(f"Left samples: {self.left_samples}")
        logger.debug(f"Right samples: {self.right_samples}")
        logger.debug(f"Video samples: {self.video_samples}")
        logger.debug(f"Gaze samples: {self.gaze_samples}")
        logger.debug(f"Corrected GPS samples: {self.corrected_gps_samples}")

        # Extract the sample info for each subject
        # so that we know the useable duration of each sample per subject
        self.subject_sample_metadatas = self._gather_subject_sample_metadatas()
        # Cache for corrected GPS coordinates to ensure we only interpolate once
        self.corrected_gps_cache = {}
        self.gaze_data_cache = {}

        # A helper to return the sample info for a given index
        # Used internally. Please refer to get_with_info() method.
        self._return_info = False
        self._coordinate_transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

        # Create an indexer over the dataset
        self._indexer = self._create_indexer()
        self._faulty_samples = set()
        self._faulty_sample_replacer = np.random.default_rng(42)

        if self.use_cache:
            self._cache_size = 0
            if self.cache_dir is None:
                self.cache_dir = Path(tempfile.mkdtemp())
                atexit.register(lambda: shutil.rmtree(self.cache_dir))
            logger.info(f"Using cache at {self.cache_dir}")

        logger.info("Dataset initialized")
        logger.debug(f"Number of samples: {len(self)}")

    def _initialize_stitcher(self):
        """Initialize the ImageStitcher."""
        logger.info("Initializing LoFTR-based stitcher")
        matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.stitcher = ImageStitcher(matcher, estimator="ransac").to(self.device)

    def _gather_subjects(self) -> List[str]:
        """Gather all subjects in the dataset.

        For each subdir 01GoPro, 02EyeTracker, etc, check the subdirs for
        the subject names 001, 002, etc. and ensure that they are the same
        across all subdirs.

        Returns
        -------
        List[str]
            List of subject names.
        """
        logger.info(f"Gathering subjects in {self.root}")
        subjects = []
        for subdir in self.root.iterdir():
            if not subdir.is_dir():
                logger.warning(f"Skipping non-directory {subdir} in gathering subjects")
                continue
            subjects.append([s.name for s in subdir.iterdir()])
        common_subjects = set.intersection(*map(set, subjects)) if subjects else []

        if len(common_subjects) == 0:
            logger.error(f"No subjects found in {self.root}")
            raise ValueError(f"No subjects found in {self.root}")

        if not all(len(s) == len(common_subjects) for s in subjects):
            missing_subjects = set.difference(*map(set, subjects))
            logger.warning(f"Number of subjects not the same across subdirs in {self.root}")
            logger.info(f"Missing subjects: {missing_subjects}")

        return sorted(common_subjects)

    def _gather_gopro_samples(
        self,
    ) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        """Gather all samples for 01GoPro.

        Returns
        -------
        Tuple[Dict[str,List[Path]], Dict[str,List[Path]]]]
            Tuple of left and right video files for each subject.
        """
        logger.info(f"Gathering samples in {self.root / '01GoPro'}")
        left = {}
        right = {}
        for subject in self.subjects:
            left_subject = sorted(
                (self.root / "01GoPro" / subject / "left").glob("*.MP4")
            ) + sorted((self.root / "01GoPro" / subject / "links").glob("*.MP4"))
            right_subject = sorted(
                (self.root / "01GoPro" / subject / "right").glob("*.MP4")
            ) + sorted((self.root / "01GoPro" / subject / "rechts").glob("*.MP4"))

            logger.info(f"Filtering samples for subject {subject} in 01GoPro")
            logger.debug(f"Left: {left_subject}")
            logger.debug(f"Right: {right_subject}")

            left[subject], right[subject] = self._filter_gopro_samples(left_subject, right_subject)

            if len(left_subject) != len(right_subject):
                logger.warning(
                    f"Number of left and right video files not the same "
                    f"for subject {subject} in 01GoPro"
                )
                logger.info(f"Left: {left_subject}")
                logger.info(f"Right: {right_subject}")

            if len(left[subject]) == 0:
                logger.warning(f"No matching video files for subject {subject} in 01GoPro")

        return left, right

    def _filter_gopro_samples(
        self, left: List[Path], right: List[Path]
    ) -> Tuple[List[Path], List[Path]]:
        """Find matching video files for each side and discard the rest.

        Matching video files are those that have the same four letters,
        e.g. "GH010008.MP4" on the left and "GH010009.MP4" on the right.

        We ignore videos with the longer names, e.g. "GH010001_1636279619167.mp4".
        """
        left, right = left.copy(), right.copy()
        left_filtered = []
        right_filtered = []
        for lpath in left:
            for ridx, rpath in enumerate(right):
                if (
                    lpath.stem[:4] == rpath.stem[:4]
                    and len(lpath.stem) < 10
                    and len(rpath.stem) < 10
                ):
                    logger.info(f"Found matching video files: {lpath} and {rpath}")
                    left_filtered.append(lpath)
                    right_filtered.append(rpath)
                    right.pop(ridx)
                    break

        # report the discarded videos
        left_discarded = set(left) - set(left_filtered)
        right_discarded = set(right) - set(right_filtered)
        if len(left_discarded) > 0 or len(right_discarded) > 0:
            logger.warning("Discarding videos in 01GoPro")

        if len(left_discarded) > 0:
            logger.info(f"Discarded left videos: {left_discarded}")
        if len(right_discarded) > 0:
            logger.info(f"Discarded right videos: {right_discarded}")

        return left_filtered, right_filtered

    def _gather_eyetracker_samples(
        self,
    ) -> Tuple[Dict[str, _EyeTrackerFront], Dict[str, _EyeTrackerGaze]]:
        """Gather all samples for 02EyeTracker.

        These are the samples from Pupil Labs eye tracker, aka the
        Pupil Invisible. See https://docs.pupil-labs.com/developer/core/recording-format/. # noqa: E501

        Returns
        -------
        Dict[str,_EyeTrackerFront]
            Video files for each subject.
        Dict[str,_EyeTrackerGaze]
            Gaze files for each subject.
        """
        logger.info(f"Gathering samples in {self.root / '02EyeTracker'}")
        videos = {}
        gaze = {}
        # video files are named "world.mp4" with intrinsics in "world.intrinsics" and timestamps in "world_timestamps.npy" # noqa: E501
        # gaze files are in "gaze.pldata" with timestamps in "gaze_timestamps.npy"
        for subject in self.subjects:
            videos[subject] = {
                "video": self.root / "02EyeTracker" / subject / "world.mp4",
                "intrinsics": self.root
                / "02EyeTracker"
                / subject
                / "world.intrinsics",  # noqa: E501
                "time": self.root / "02EyeTracker" / subject / "world_timestamps.npy",  # noqa: E501
            }
            gaze[subject] = {
                "gaze": self.root / "02EyeTracker" / subject / "gaze.pldata",
                "time": self.root / "02EyeTracker" / subject / "gaze_timestamps.npy",  # noqa: E501
            }

            # ensure all files exist
            for video in videos[subject].values():
                # Handle the case in 009 - where the video is named "world_001.mp4" and timestamps are in "world_001_timestamps.npy" # noqa: E501
                if not video.exists() and subject == "009":
                    videos[subject]["video"] = (
                        self.root / "02EyeTracker" / subject / "world_001.mp4"
                    )
                    videos[subject]["time"] = (
                        self.root / "02EyeTracker" / subject / "world_001_timestamps.npy"
                    )

                if not video.exists():
                    logger.warning(f"Video file {video} does not exist")

            for gaze_file in gaze[subject].values():
                if not gaze_file.exists():
                    logger.warning(f"Gaze file {gaze_file} does not exist")

        return videos, gaze

    def _gather_corrected_gps_samples(
        self,
    ) -> Dict[str, List[Path]]:
        """Gather all samples for 03CorrectedGPS.

        These are hand-corrected GPS samples, based on GoPro videos.

        Returns
        -------
        Dict[str,List[Path]]
            CSV files per each video of each subject.
        """
        logger.info(f"Gathering samples in {self.root / '03CorrectedGPS'}")
        samples = {}
        for subject in self.subjects:
            candidate_samples = sorted((self.root / "03CorrectedGPS" / subject).glob("*.csv"))

            logger.info(
                f"Found {len(candidate_samples)} candidate samples " f"for subject {subject}"
            )

            samples[subject] = []
            for sample in candidate_samples:
                # Check if the GPS is sampled from the left or right video
                # The prefix of the name in the form GH0x00yz should match
                # the video name
                is_left = any(
                    sample.stem.startswith(left.stem[:8]) for left in self.left_samples[subject]
                )
                is_right = any(
                    sample.stem.startswith(right.stem[:8]) for right in self.right_samples[subject]
                )
                if is_left or is_right:
                    samples[subject].append(sample)
                    logger.info(f"Found matching sample for subject {subject}: {sample}")
                else:
                    logger.warning(f"Discarding sample for subject {subject}: {sample}")

            logger.debug(f"Samples for subject {subject}: {samples[subject]}")

        return samples

    def _gather_subject_sample_metadatas(self) -> Dict[str, Any]:
        """For each sample per subject, gather metadata.

        Returns
        -------
        Dict[str,Any]
            Video file information.
        """
        logger.info("Gathering sample metadata")
        subject_infos = {}

        for subject in self.subjects:
            gaze_metadata = self._get_gaze_metadata(subject)
            logger.info(f"Read gaze metadata for subject {subject}")

            corrected_gps_samples = self.corrected_gps_samples[subject]

            info = {}
            for left, right, corr_gps in zip(
                self.left_samples[subject],
                self.right_samples[subject],
                corrected_gps_samples,
            ):
                video_metadata = self._get_sample_metadata(left, right, gaze_metadata)
                logger.info(f"Read video metadata for subject {subject}")
                logger.debug(f"Metadata for {left}, {right} and {corr_gps}: {video_metadata}")
                info[(left, right, corr_gps)] = video_metadata

            subject_infos[subject] = info

        return subject_infos

    def _interpolate_corrected_gps(self, file: Path) -> pd.DataFrame:
        """Interpolate corrected GPS samples.

        Returns
        -------
        pd.DataFrame
            Interpolated corrected GPS samples.
        """
        logger.info(f"Interpolating corrected GPS samples from {file}")

        # find the file first
        file_metadata = [
            file_metadata
            for sample_metadata in self.subject_sample_metadatas.values()
            for file_metadata in sample_metadata.items()
            if file.samefile(file_metadata[0][2])  # the key is a tuple of (left, right, corr_gps)
        ]

        if len(file_metadata) == 0:
            logger.error(f"Corrected GPS file {file} not found")
            raise ValueError(f"Corrected GPS file {file} not found")

        (left, right, corr_gps), video_metadata = file_metadata[0]

        # interpolate the corrected GPS samples
        gps_df = pd.read_csv(
            corr_gps,
            header=None,
            names=["latitude", "longitude", "milliseconds"],
        )
        logger.info(f"Read corrected GPS samples from {corr_gps}")
        logger.debug(f"Corrected GPS samples from {corr_gps}: {gps_df}")

        # Convert latitude and longitude to meters
        xy = self._convert_gps_coordinates(gps_df[["latitude", "longitude"]].values)
        gps_df["x"] = xy[:, 0]
        gps_df["y"] = xy[:, 1]
        gps_df = gps_df.drop(columns=["latitude", "longitude"])

        # sort by seconds, fix the speed issue with a basic heuristic
        gps_df["seconds"] = gps_df["milliseconds"] / 1000
        gps_df = gps_df.drop(columns=["milliseconds"]).sort_values(by="seconds")

        # Check if the GPS is sampled from the left or right video
        # The prefix of the name in the form GH0x00yz should match the video name
        is_left = left.stem.startswith(corr_gps.stem[:8])
        if is_left:
            logger.debug(
                f"Corrected GPS samples from {corr_gps} are from " f"the left video: {left}"
            )
        elif right.stem.startswith(corr_gps.stem[:8]):
            logger.debug(
                f"Corrected GPS samples from {corr_gps} are from " f"the right video: {right}"
            )
        else:
            raise ValueError(
                f"Corrected GPS samples from {corr_gps} do not match "
                f"any video {left} or {right}"
            )

        origin_time, duration = (
            video_metadata["origin_time"],
            video_metadata["duration"],
        )
        offset = video_metadata["left_offset" if is_left else "right_offset"]

        gps_df["timestamp"] = gps_df["seconds"] + origin_time - offset

        # interpolate the corrected GPS samples
        gps = self._interpolate_gps(gps_df, origin_time, duration)
        logger.info(f"Interpolated corrected GPS samples from {corr_gps}")
        logger.debug(f"Interpolated corrected GPS samples from {corr_gps}: {gps}")

        return gps

    def _interpolate_gps(self, gps: pd.DataFrame, origin_time, duration) -> pd.DataFrame:
        """Interpolate GPS samples with PChip interpolation.

        Interpolates between 0 and duration seconds.

        Parameters
        ----------
        gps : pd.DataFrame
            GPS samples. Has the columns "x", "y" and "timestamp".
        origin_time : float
            Origin time of the video.
        duration : float
            Duration of the video.

        Returns
        -------
        pd.DataFrame
            Interpolated GPS samples, indexed by timestamp.
        """
        # Use scipy.interpolate.PchipInterpolator to interpolate the GPS samples
        # The interpolator is used to interpolate between 0 and duration seconds
        interpolator = interpolate.PchipInterpolator(
            gps["timestamp"], gps[["x", "y"]].values, extrapolate=False
        )
        timestamps = np.arange(
            origin_time,
            origin_time + duration + 1 / self.output_fps,
            1 / self.output_fps,
        )
        interpolated = interpolator(timestamps)
        interpolated_df = pd.DataFrame(
            interpolated,
            index=timestamps,
            columns=["x", "y"],
        )
        logger.debug(f"Interpolated GPS samples: {interpolated_df}")

        # fill missing values with the nearest value
        interpolated_df = interpolated_df.ffill().bfill()

        return interpolated_df

    def _get_sample_metadata(self, left: Path, right: Path, gaze_metadata: dict) -> Dict[str, Any]:
        """Get metadata for a single sample.

        Parameters
        ----------
        left : Path
            Left video file.
        right : Path
            Right video file.
        gaze_metadata : dict
            Gaze metadata.

        Returns
        -------
        Dict[str,Any]
            Metadata for the sample.
        """
        left_metadata = self._read_video_metadata(left)
        logger.info(f"Read left video {left} metadata")
        logger.debug(f"Metadata for {left}: {left_metadata}")

        right_metadata = self._read_video_metadata(right)
        logger.info(f"Read right video {right} metadata")
        logger.debug(f"Metadata for {right}: {right_metadata}")

        gps_start_time = max(
            left_metadata["start_time"],
            right_metadata["start_time"],
            gaze_metadata["start_time_gaze"],
            gaze_metadata["start_time_video"],
        )
        left_offset = max(0, gps_start_time - left_metadata["start_time"])
        right_offset = max(0, gps_start_time - right_metadata["start_time"])
        gaze_sampling_offset = max(0, gps_start_time - gaze_metadata["start_time_gaze"])
        gaze_video_offset = max(0, gps_start_time - gaze_metadata["start_time_video"])
        logger.info(f"Left offset: {left_offset}")
        logger.info(f"Right offset: {right_offset}")
        logger.info(f"Gaze sampling offset: {gaze_sampling_offset}")
        logger.info(f"Gaze video offset: {gaze_video_offset}")

        duration = min(
            left_metadata["duration"] - left_offset,
            right_metadata["duration"] - right_offset,
            gaze_metadata["duration"] - gaze_sampling_offset,
            gaze_metadata["duration"] - gaze_video_offset,
        )
        logger.info(f"Duration: {duration}")

        if self.with_video and left_metadata["duration"] != right_metadata["duration"]:
            logger.warning(f"Duration is not the same for videos {left} and {right}")
            logger.info(f"Left: {left_metadata['duration']}")
            logger.info(f"Right: {right_metadata['duration']}")

        if self.with_video and right_metadata["video_fps"] != left_metadata["video_fps"]:
            logger.warning(f"FPS is not the same for videos {left} and {right}")
            logger.info(f"Left: {left_metadata['video_fps']}")
            logger.info(f"Right: {right_metadata['video_fps']}")

        return {
            "duration": duration,
            "origin_time": gps_start_time,
            "left_offset": left_offset,
            "right_offset": right_offset,
            "gaze_sampling_offset": gaze_sampling_offset,
            "gaze_video_offset": gaze_video_offset,
            "left_metadata": left_metadata,
            "right_metadata": right_metadata,
            "gaze_metadata": gaze_metadata,
        }

    def _create_indexer(self) -> Dict[int, Any]:
        """Create an indexer for the whole dataset.

        Provides subject, sample paths, metadata and start time.

        Returns
        -------
        Dict[int,Any]
            Indexer.
        """
        logger.info("Creating indexer...")
        if self.min_pci or self.max_pci:
            logger.info(f"Filtering samples with pci between {self.min_pci}" f" and {self.max_pci}")
        indexer = {}

        index = 0
        for subject in self.subjects:
            subject_sample_metadata = self.subject_sample_metadatas[subject]
            for (left, right, corr_gps), metadata in subject_sample_metadata.items():
                duration = metadata["duration"]
                chunk_size = self.input_length + self.target_length
                # we need to slide the window by self.step_size
                # to get all possible samples
                start_time = 0
                while start_time <= duration - chunk_size:
                    full_corrected_gps = self._get_full_corrected_gps(corr_gps)
                    # full_corrected_gps is a pandas Dataframe indexed by
                    # timestamp in POSIX time which can be accessed by range index
                    gps_start = metadata["origin_time"] + start_time
                    input_trajectory = full_corrected_gps.loc[
                        gps_start : gps_start + self.input_length
                    ].values
                    target_trajectory = full_corrected_gps.loc[
                        gps_start + self.input_length : gps_start + chunk_size
                    ].values
                    pci = estimate_pci(
                        input_trajectory,
                        target_trajectory,
                        curve_type="linear",
                        lookback_length=6,
                        frequency=self.output_fps,
                        measure="frechet",
                    )
                    if (self.min_pci is not None and pci < self.min_pci) or (
                        self.max_pci is not None and pci > self.max_pci
                    ):
                        start_time += self.step_size
                        continue

                    indexer[index] = {
                        "subject": subject,
                        "left": left,
                        "right": right,
                        "corr_gps": corr_gps,
                        "sample_start_time": start_time,
                        "sample_duration": chunk_size,
                        "trajectory_metadata": metadata,
                        "pci": pci,
                    }
                    index += 1

                    if self.avoid_overlap:
                        start_time += max(self.input_length, self.step_size)
                    else:
                        start_time += self.step_size

        return indexer

    def __len__(self) -> int:
        """Get the dataset length.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self._indexer)

    def __getitem__(  # noqa: C901 -> FIXME please.
        self, idx: int
    ) -> Union[Item, Tuple[Item, dict]]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            Item index.

        Returns
        -------
        Item
            Dataset item.
        dict
            Additional information about the item.
        """
        if idx not in self._indexer:
            raise IndexError(f"Index {idx} is out of range")

        item = self._indexer[idx]
        subject = item["subject"]
        left = item["left"]
        right = item["right"]
        corr_gps = item["corr_gps"]
        start_time = item["sample_start_time"]
        metadata = item["trajectory_metadata"]
        pci = item["pci"]

        logger.info(f"Reading sample {idx} for subject {subject}")

        skip_faulty_sample_found_in_cache = False
        if idx not in self._faulty_samples:
            if self.use_cache:
                data = self._fetch_from_cache(item)
                if data is not None and data.get("is_sample_ok", True):
                    # Delete is_sample_ok from data, as it is not needed anymore
                    if "is_sample_ok" in data:
                        del data["is_sample_ok"]
                    data["pci"] = pci
                    if self._return_info:
                        return data, item
                    return data
                elif data is not None:
                    skip_faulty_sample_found_in_cache = True
            if skip_faulty_sample_found_in_cache:
                # data is not None at this point, as it was found in cache
                # but it was faulty, so we do not need to recompute it
                logger.info(f"Sample {idx} is read and was faulty, skipping")
                is_sample_ok = False
            else:
                logger.info(f"Sample {idx} with start time {start_time} is not in cache, computing")
                data, is_sample_ok = self._get_sample_data(
                    subject, left, right, corr_gps, start_time, metadata
                )
                data["pci"] = pci
        else:
            logger.info(f"Sample {idx} is faulty, skipping")
            data = None
            is_sample_ok = False
            skip_faulty_sample_found_in_cache = True

        if self.use_cache and self._cache_size < self.max_cache_size:
            is_cacheable = (
                is_sample_ok or not skip_faulty_sample_found_in_cache
            )  # we cache faulty samples as well
            if is_cacheable:
                logger.info(f"Sample {idx} is cacheable")
                data["is_sample_ok"] = is_sample_ok
                self._push_to_cache(item, data)
            else:
                logger.info(f"Sample {idx} is not cacheable due to faulty data")

        if not is_sample_ok:
            self._faulty_samples.add(idx)
            # FIXME: none of the samples should do it, we need to check the
            #  source of this issue. For now, just return another sample
            logger.warning(f"Sample {idx} is not valid, returning a random sample instead")
            next_idx = self._faulty_sample_replacer.integers(0, len(self))
            del data
            return self.__getitem__(next_idx)

        # Delete is_sample_ok from data, as it is not needed anymore
        if "is_sample_ok" in data:
            del data["is_sample_ok"]

        logger.info(f"Returning data for sample {idx}")

        if self._return_info:
            return data, item

        return data

    def _fetch_from_cache(self, item: dict) -> Item:
        # hash the item repr deterministically and efficiently
        item_hash = self._hash_item(item)
        # read from cache, which is a zstd-compressed pickle file
        # under cache_dir with the name [hash].pkl.zstd
        candidate_cache_file = self.cache_dir / f"{item_hash}.pkl.zstd"
        if candidate_cache_file.exists():
            logger.info(f"Reading from cache {candidate_cache_file}")
            try:
                with open(candidate_cache_file, "rb") as f:
                    data = pickle.loads(zstd.decompress(f.read()))
                return data
            except zstd.Error:
                logger.warning(f"Failed to decompress {candidate_cache_file}")
                logger.info("Deleting corrupt cache file")
                candidate_cache_file.unlink()
        else:
            logger.info(f"Cache file {candidate_cache_file} does not exist")

        return None

    def _push_to_cache(self, item: dict, data: Item):
        item_hash = self._hash_item(item)
        candidate_cache_file = self.cache_dir / f"{item_hash}.pkl.zstd"
        if not candidate_cache_file.exists():
            logger.info(f"Writing to cache {candidate_cache_file}")
            with open(candidate_cache_file, "wb") as f:
                compressed_data = zstd.compress(pickle.dumps(data), 3, 2)
                f.write(compressed_data)
                self._cache_size += len(compressed_data)
        else:
            logger.info(f"Cache file {candidate_cache_file} already exists")

    def _hash_item(self, item: Item) -> str:
        hashstring = repr(item) + (
            # relevant parameters
            repr(self.crop_videos)
            + repr(self.undistort_videos)
            + repr(self.stitch_videos)
            + repr(self.gopro_scaling_factor)
            + repr(self.front_scaling_factor)
            + repr(self.frame_transform)
            + repr(self.video_transform)
            + repr(self.output_format)
            + repr(self.dilution_threshold)
            + repr(self.with_video)
            + repr(self.with_audio)
            + repr(self.with_gaze)
            + repr(self.mask_nonfixations)
        )
        item_hash = hashlib.blake2b(hashstring.encode(), digest_size=32).hexdigest()
        logger.debug(f"Hashed {hashstring} to {item_hash}")
        return item_hash

    def __iter__(self) -> Iterator[Item]:
        """Get an iterator over the dataset.

        Returns
        -------
        Iterator[Item]
            Dataset iterator.
        """
        for idx in range(len(self)):
            yield self[idx]

    def get_with_info(self, idx: int) -> Tuple[Item, dict]:
        """Get an item from the dataset with additional information.

        Parameters
        ----------
        idx : int
            Item index.

        Returns
        -------
        Item
            Dataset item.
        dict
            Additional information about the item.
        """
        self._return_info = True
        item, info = self.__getitem__(idx)
        self._return_info = False
        return item, info

    def _get_sample_data(
        self,
        subject: str,
        left: Path,
        right: Path,
        corr_gps: Path,
        start_time: int,
        metadata: Dict[str, Any],
    ) -> Tuple[Item, bool]:
        """Get data for a single sample.

        Parameters
        ----------
        subject : str
            Subject ID.
        left : Path
            Left video file.
        right : Path
            Right video file.
        corr_gps : Path
            Corrected GPS file.
        start_time : int
            Start time.
        metadata : Dict[str, Any]
            Sample metadata.

        Returns
        -------
        Item
            Sample data.
        bool
            Whether the sample is valid.
        """
        logger.info(f"Getting data for {subject}, {left}, {right}, {corr_gps}, {start_time}")
        logger.debug(f"Metadata: {metadata}")
        gaze_metadata = metadata["gaze_metadata"]
        left_offset = metadata["left_offset"]
        right_offset = metadata["right_offset"]
        origin_time = metadata["origin_time"]
        data, start_posix, end_posix = self._get_video_data(
            left, right, corr_gps, start_time, origin_time, left_offset, right_offset
        )
        gaze_data = self._get_gaze_data(subject, gaze_metadata, start_posix, end_posix)
        data.update(gaze_data)
        data = self._check_sanity(data)
        if self.with_video:
            data = self._apply_scaling(data)
            data = self._convert_to_float16(data)
        if self.stitch_videos:
            data = self._move_to_tensor(data)
            # data is now a dict of tensors
            # currently this cannot be used with multiple workers
            if not hasattr(self, "stitcher"):
                self._initialize_stitcher()
            data = self._stitch_videos(data)
        data = self._apply_transforms(data)
        data, is_sample_ok = self._train_target_split(data, subject)
        logger.info(f"Computed data for {subject}, {left}, {right}, {corr_gps}, {start_time}")
        return data, is_sample_ok

    def _get_video_data(
        self,
        left: Path,
        right: Path,
        corr_gps: Path,
        start: float,
        origin_time: float,
        left_offset: float,
        right_offset: float,
    ) -> Dict[str, Any]:
        end = start + self.input_length + self.target_length
        # Ensure that the frame counts are not shorter than expected
        end += 1 / self.VIDEO_FPS

        need_to_read_video = self.with_video or self.with_audio

        if need_to_read_video:
            logger.info(f"Reading video {left} from {start} to {end}")
            left_data = self._read_video(left, start + left_offset, end + left_offset)

            logger.info(f"Reading video {right} from {start} to {end}")
            right_data = self._read_video(right, start + right_offset, end + right_offset)
        else:
            left_data = {}
            right_data = {}

        if self.with_video:
            if self.undistort_videos:
                logger.info(f"Undistorting video {left} from {start} to {end}")
                left_data["video"] = self._undistort_video(
                    left_data["video"],
                    self.LEFT_VIDEO_CAMERA_INTRINSICS,
                    self.LEFT_VIDEO_DISTORTION_COEFFICIENTS,
                )

                logger.info(f"Undistorting video {right} from {start} to {end}")
                right_data["video"] = self._undistort_video(
                    right_data["video"],
                    self.RIGHT_VIDEO_CAMERA_INTRINSICS,
                    self.RIGHT_VIDEO_DISTORTION_COEFFICIENTS,
                )
            if self.crop_videos:
                logger.info(f"Cropping video {left}")
                left_data["video"] = left_data["video"][
                    :,
                    :,
                    int(0.3 * left_data["video"].shape[2]) : int(0.7 * left_data["video"].shape[2]),
                ]
                logger.info(f"Cropping video {right}")
                right_data["video"] = right_data["video"][
                    :,
                    :,
                    int(0.3 * right_data["video"].shape[2]) : int(
                        0.7 * right_data["video"].shape[2]
                    ),
                ]

        start_posix = origin_time + start
        end_posix = origin_time + end
        data = self._combine_data(left_data, right_data, corr_gps, start_posix, end_posix)

        return data, start_posix, end_posix

    def _check_sanity(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.with_video:
            left_video = data["left_video"]
            right_video = data["right_video"]
            front_video = data["front_video"] if self.with_gaze else None

            if left_video.shape[0] != right_video.shape[0] or (
                front_video is not None and left_video.shape[0] != front_video.shape[0]
            ):
                min_len = min(left_video.shape[0], right_video.shape[0], front_video.shape[0])
                logger.warning("Video length is not the same for left, right and front videos")
                logger.info(f"Left: {left_video.shape[0]}")
                logger.info(f"Right: {right_video.shape[0]}")
                if front_video is not None:
                    logger.info(f"Front: {front_video.shape[0]}")
                left_video = left_video[:min_len]
                right_video = right_video[:min_len]
                front_video = front_video[:min_len] if front_video is not None else None

            data["left_video"] = left_video
            data["right_video"] = right_video
            if front_video is not None:
                data["front_video"] = front_video

        if self.with_audio:
            left_audio = data["left_audio"]
            right_audio = data["right_audio"]
            front_audio = data["front_audio"]
            logger.info(f"Left audio shape: {left_audio.shape}")
            logger.info(f"Right audio shape: {right_audio.shape}")
            logger.info(f"Front audio shape: {front_audio.shape}")

            if (
                left_audio.shape[0] != right_audio.shape[0]
                or left_audio.shape[0] != front_audio.shape[0]
            ):
                min_len = min(left_audio.shape[0], right_audio.shape[0], front_audio.shape[0])
                logger.warning("Audio length is not the same for left, right and front audio")
                logger.info(f"Left: {left_audio.shape[0]}")
                logger.info(f"Right: {right_audio.shape[0]}")
                logger.info(f"Front: {front_audio.shape[0]}")
                left_audio = left_audio[:min_len]
                right_audio = right_audio[:min_len]
                front_audio = front_audio[:min_len]

            data["left_audio"] = left_audio
            data["right_audio"] = right_audio
            data["front_audio"] = front_audio

        return data

    def _apply_transforms(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the transforms to the data.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Data to transform.

        Returns
        -------
        dict[str, torch.Tensor]
            Transformed data.
        """
        if self.frame_transform is not None:
            logger.info("Applying frame transform")
            # torch.stack with list comprehension leads to memory issues
            for frame_idx in range(data["left_video"].shape[0]):
                data["left_video"][frame_idx] = self.frame_transform(data["left_video"][frame_idx])
            for frame_idx in range(data["right_video"].shape[0]):
                data["right_video"][frame_idx] = self.frame_transform(
                    data["right_video"][frame_idx]
                )
            for frame_idx in range(data["front_video"].shape[0]):
                data["front_video"][frame_idx] = self.frame_transform(
                    data["front_video"][frame_idx]
                )
            if self.stitch_videos:
                for frame_idx in range(data["stitched_video"].shape[0]):
                    data["stitched_video"][frame_idx] = self.frame_transform(
                        data["stitched_video"][frame_idx]
                    )

        if self.video_transform is not None:
            logger.info("Applying video transform")
            data["left_video"] = self.video_transform(data["left_video"])
            data["right_video"] = self.video_transform(data["right_video"])
            data["front_video"] = self.video_transform(data["front_video"])
            if self.stitch_videos:
                data["stitched_video"] = self.video_transform(data["stitched_video"])

        return data

    def _apply_scaling(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Scale the videos.

        Parameters
        ----------
        data : dict[str, np.array]
            Data to stitch.

        Returns
        -------
        dict[str, np.array]
            Data dict with scaled videos
        """
        videos_to_scale = []
        if self.gopro_scaling_factor != 1:
            videos_to_scale.extend(["left_video", "right_video"])
        if self.front_scaling_factor != 1:
            videos_to_scale.append("front_video")

        if videos_to_scale:
            logger.info(f"Scaling videos {videos_to_scale}")
            for video in videos_to_scale:
                frames = [frame.transpose(1, 2, 0) for frame in data[video]]
                factor = (
                    self.front_scaling_factor
                    if video == "front_video"
                    else self.gopro_scaling_factor
                )
                target_resolution = (
                    int(frames[0].shape[1] * factor),
                    int(frames[0].shape[0] * factor),
                )
                logger.debug(
                    f"Scaling {video} by {factor} from {frames[0].shape} " f"to {target_resolution}"
                )
                if self.num_workers == 1:
                    frames = [
                        cv2.resize(
                            frame,
                            target_resolution,
                            None,
                            None,
                            None,
                            cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
                        )
                        for frame in frames
                    ]
                else:
                    with mp.Pool(self.num_workers) as pool:
                        frames = pool.starmap(
                            cv2.resize,
                            [
                                (
                                    frame,
                                    target_resolution,
                                    None,
                                    None,
                                    None,
                                    cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
                                )
                                for frame in frames
                            ],
                        )
                data[video] = np.stack(frames).transpose(0, 3, 1, 2)

        return data

    def _convert_to_float16(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert the videos to float16 from uint8.

        Parameters
        ----------
        data : dict[str, np.array]
            Data to convert.

        Returns
        -------
        dict[str, np.array]
            Data dict with converted videos
        """
        logger.info("Converting videos to float16")
        for video in ["left_video", "right_video", "front_video"]:
            if video in data:
                data[video] = data[video].astype(np.float16) / 255.0
        return data

    @torch.no_grad()
    def _move_to_tensor(self, data: dict[str, np.array]) -> dict[str, torch.Tensor]:
        """Move the data to the device and convert to tensors.

        Parameters
        ----------
        data : dict[str, np.array]
            Data to move.

        Returns
        -------
        dict[str, torch.Tensor]
            Moved data.
        """
        for key in data:
            if isinstance(data[key], np.ndarray):
                logger.debug(f"Moving {key} with shape {data[key].shape} to {self.device}")
                data[key] = torch.from_numpy(data[key]).to(self.device)
            else:
                data[key] = data[key].to(self.device)

        return data

    @torch.no_grad()
    @_set_num_workers
    def _stitch_videos(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Stitch the videos together.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Data to stitch.

        Returns
        -------
        dict[str, torch.Tensor]
            Stitched data.
        """
        frame_count = data["left_video"].shape[0]

        output_frames = []
        left_right_homography_matrix = None
        front_intermediate_homography_matrix = None
        for frame_idx in range(frame_count):
            logger.info(f"Stitching frame {frame_idx + 1}/{frame_count}")
            # first stitch the left and right videos, then the front video
            left_frame, right_frame = data["left_video"][frame_idx].unsqueeze(0), data[
                "right_video"
            ][frame_idx].unsqueeze(0)
            frame_width = left_frame.shape[3]
            left_frame = F.pad(left_frame, (0, int(frame_width * 0.9)), mode="constant", value=0)
            intermediate_frame, left_right_homography_matrix = self.stitcher(
                left_frame,
                right_frame,
                homography_matrices=left_right_homography_matrix,
            )
            # pad the front frame
            front_frame = data["front_video"][frame_idx].unsqueeze(0)
            front_frame = F.pad(
                front_frame,
                (frame_width // 2, frame_width // 2),
                mode="constant",
                value=0,
            )
            try:
                output_frame, front_intermediate_homography_matrix = self.stitcher(
                    front_frame,
                    intermediate_frame,
                )
            except:  # noqa: E722
                # if the homography matrix is not invertible, use the previous one
                output_frame, front_intermediate_homography_matrix = self.stitcher(
                    front_frame,
                    intermediate_frame,
                    homography_matrices=front_intermediate_homography_matrix,
                )
            output_frames.append(output_frame)
        output_video = torch.cat(output_frames, dim=0)
        data["stitched_video"] = output_video
        return data

    def _train_target_split(self, data: dict[str, torch.Tensor], subject: str) -> Tuple[Item, bool]:
        """Split the data into train and target."""
        data_keys = list(data.keys())
        is_sample_ok = True
        for phase in ["train", "target"]:
            phase_data = {}
            for key in data_keys:
                logger.debug(f"Slicing {key} for {phase}")
                input_frame_count, target_frame_count = self._get_frame_counts(key, subject)
                phase_start = 0 if phase == "train" else input_frame_count
                phase_end = (
                    input_frame_count
                    if phase == "train"
                    else input_frame_count + target_frame_count
                )
                phase_data[key] = data[key][phase_start:phase_end]
                logger.debug(f"{key} {phase} shape: {phase_data[key].shape}")
                if phase == "target" and data[key].shape[0] != phase_end:
                    issue = "longer" if data[key].shape[0] > phase_end else "shorter"
                    logger.warning(f"Target data for {key} is {issue} than expected")
                    logger.info(f"Expected {phase_end}, got {data[key].shape[0]}")
                    if issue == "shorter":
                        # The sample cannot be concatenated with others
                        is_sample_ok = False
            data[phase] = phase_data

        for key in data_keys:
            del data[key]

        # if the sample is ok, and the subject belongs to on of 009 or 010,
        # we upsample the gaze data
        if is_sample_ok and subject in ["009", "010"]:
            logger.info(f"Upsampling gaze data for subject {subject}")
            desired_input_frames, desired_target_frames = self._get_frame_counts("gaze", "001")
            for phase, target_frame_count in zip(
                ["train", "target"], [desired_input_frames, desired_target_frames]
            ):
                phase_data = data[phase]
                if self.with_gaze:
                    phase_data["gaze"] = self._upsample_gaze_data(
                        phase_data["gaze"], target_frame_count
                    )
                data[phase] = phase_data

        return data, is_sample_ok

    def _upsample_gaze_data(self, gaze_data: np.ndarray, target_frame_count: int) -> np.ndarray:
        """Upsample the gaze data.

        Parameters
        ----------
        gaze_data : np.ndarray
            Gaze data to upsample.
        target_frame_count : int
            Target frame count.

        Returns
        -------
        np.ndarray
            Upsampled gaze data.
        """
        # gaze_data is a 2D array of shape (num_samples, 2)
        # we need to upsample it to (target_frame_count, 2)
        timestamps = np.linspace(0, 1, num=gaze_data.shape[0])
        target_timestamps = np.linspace(0, 1, num=target_frame_count)
        upsampled_gaze_data = interpolate.interp1d(
            timestamps,
            gaze_data,
            axis=0,
            kind="nearest",
            fill_value="extrapolate",
            assume_sorted=True,
        )(target_timestamps)

        return upsampled_gaze_data

    def _get_frame_counts(self, key: str, subject: str) -> Tuple[int, int]:
        """Get the frame counts for a key.

        Parameters
        ----------
        key : str
            Key to get the frame counts for.
        subject : str
            Subject ID.

        Returns
        -------
        Tuple[int,int]
            Frame counts for the key.
        """
        if "video" in key or "gps" in key:
            return self.input_video_frame_count, self.target_video_frame_count
        elif "audio" in key:
            return self.input_audio_frame_count, self.target_audio_frame_count
        elif "gaze" in key:
            if subject in ["009", "010"]:
                return (
                    self.alternative_input_gaze_frame_count,
                    self.alternative_target_gaze_frame_count,
                )
            return self.input_gaze_frame_count, self.target_gaze_frame_count
        else:
            raise ValueError(f"Unknown key {key}")

    def _get_gaze_metadata(self, subject: str) -> Dict[str, Any]:
        """Get metadata for the gaze data of a subject.

        Parameters
        ----------
        subject : str
            Subject name.

        Returns
        -------
        Dict[str,Any]
            Metadata for the gaze data.
        """
        # read both info.invisible.json and info.player.json
        # they should be consistent, if not, raise a warning
        invisible = self.root / "02EyeTracker" / subject / "info.invisible.json"
        player = self.root / "02EyeTracker" / subject / "info.player.json"
        if not invisible.exists():
            raise FileNotFoundError(f"File {invisible} does not exist")
        if not player.exists():
            raise FileNotFoundError(f"File {player} does not exist")

        metadata = json.loads(invisible.read_text())
        player_metadata = json.loads(player.read_text())

        # start_time is the the sampling start time for gaze data (I think?)
        # Because otherwise (that is, the start_time is some absolute value
        # that the gaze_timestamps should be added to) the timestamps between
        # gopro videos and world video have a miniscule difference.
        metadata["start_time_gaze"] = metadata["start_time"] / 1e9
        metadata["duration"] = metadata["duration"] / 1e9

        if metadata["start_time_gaze"] != player_metadata["start_time_synced_s"]:
            logger.warning(f"Start time for subject {subject} does not match")
            logger.info(f"Start time from info.invisible.json: {metadata['start_time_gaze']}")
            logger.info(
                "Start time from info.player.json: " f"{player_metadata['start_time_synced_s']}"
            )

        if metadata["duration"] != player_metadata["duration_s"]:
            logger.warning(f"Duration for subject {subject} does not match")
            logger.info(f"Duration from info.invisible.json: {metadata['duration']}")
            logger.info(f"Duration from info.player.json: {player_metadata['duration_s']}")

        # There is a couple seconds difference between the start time of the video
        # and the start time of the gaze data
        # We need this to provide a full sample in the first iteration
        gaze_paths = self.gaze_samples[subject]
        gaze_data = load_pldata_file(gaze_paths["gaze"].parent, "gaze")

        video_timestamps = np.load(self.video_samples[subject]["time"])

        # "start_time" is some absolute start_time value that the gaze timestamps
        # are assumed to be based on
        metadata["start_time"] = metadata["start_time_gaze"] - gaze_data.timestamps[0]
        metadata["start_time_video"] = metadata["start_time"] + video_timestamps[0]

        logger.info(
            "Absolute start time for gaze data collection for"
            f"subject {subject}: {metadata['start_time']}"
        )
        logger.info(
            f"Start time for gaze data for subject {subject}: " f"{metadata['start_time_gaze']}"
        )
        logger.info(
            f"Start time for video for subject {subject}: " f"{metadata['start_time_video']}"
        )

        # Read video intrinsics
        intrinsics_path = self.video_samples[subject]["intrinsics"]
        intrinsics = load_object(intrinsics_path)

        metadata["camera_matrix"] = np.array(
            intrinsics["(1088, 1080)"]["camera_matrix"],
            dtype=np.float32,
        )
        metadata["dist_coefs"] = np.array(
            intrinsics["(1088, 1080)"]["dist_coefs"],
            dtype=np.float32,
        ).flatten()
        # This object is required for fixation detector
        metadata["intrinsics"] = intrinsics
        metadata["frame_size"] = self.GAZE_RESOLUTION

        logger.debug(f"Gaze metadata for subject {subject}: {metadata}")

        return metadata

    @_set_num_workers
    def _get_gaze_data(
        self, subject: str, gaze_metadata: dict, start_posix: float, end_posix: float
    ) -> Item:
        """Get gaze data and world video for given time interval.

        Parameters
        ----------
        subject : str
            Subject name.
        gaze_metadata : dict
            Metadata for the gaze data.
        start_posix : float
            Start time of the video in POSIX time.
        end_posix : float
            End time of the video in POSIX time.

        Returns
        -------
        Item
            Gaze data for the video, numpy arrays
        """
        # Ensure that the frame counts are not shorter than expected
        end_posix += 10 / self.GAZE_FPS

        if self.with_gaze:
            world_video = self._read_world_video(subject, gaze_metadata, start_posix, end_posix)
            logger.info(f"Read world video for subject {subject} from {start_posix} to {end_posix}")
            logger.debug(
                f"World video for subject {subject} has shape " f"{world_video['video'].shape}"
            )

            gaze_data = self._read_gaze_data(subject, gaze_metadata, start_posix, end_posix)
            logger.info(f"Read gaze data for subject {subject}")
            logger.debug(f"Gaze data for subject {subject} has shape {gaze_data.shape}")
        else:
            gaze_data = None
            world_video = {}

        data = {}

        if gaze_data is not None:
            data["gaze"] = gaze_data

        if "video" in world_video:
            logger.debug(
                f"World video for subject {subject} " f"has shape {world_video['video'].shape}"
            )
            data["front_video"] = world_video["video"]

        if "audio" in world_video:
            data["front_audio"] = world_video["audio"]

        return data

    def _read_gaze_data(
        self, subject: str, gaze_metadata: dict, start_posix: float, end_posix: float
    ) -> np.ndarray:
        """Read gaze data for a subject.

        Parameters
        ----------
        subject : str
            Subject name.
        gaze_metadata : dict
            Metadata for the gaze data.
        start_posix : float
            Start time of the video in POSIX time.
        end_posix : float
            End time of the video in POSIX time.

        Returns
        -------
        np.ndarray
            Gaze data for the video.
        """
        gaze_paths = self.gaze_samples[subject]
        if gaze_paths["gaze"] in self.gaze_data_cache:
            gaze_pos, gaze_timestamps, is_fixation = self.gaze_data_cache[gaze_paths["gaze"]]
        else:
            gaze_data = load_pldata_file(gaze_paths["gaze"].parent, "gaze")
            gaze_data = [data for data in gaze_data.data if data["topic"] == "gaze.pi"]
            is_fixation = detect_fixations(gaze_metadata, gaze_data)
            gaze_pos, gaze_timestamps = zip(
                *[
                    (
                        data["norm_pos"],
                        data["timestamp"] + gaze_metadata["start_time_gaze"],
                    )
                    for data in gaze_data
                ]
            )

            gaze_pos = np.array(gaze_pos, dtype=np.float64)
            gaze_timestamps = np.array(gaze_timestamps, dtype=np.float64)

            self.gaze_data_cache[gaze_paths["gaze"]] = (
                gaze_pos,
                gaze_timestamps,
                is_fixation,
            )

        # Unnormalize gaze positions from [0,1] to GAZE_RESOLUTION
        # See https://pupil-labs.com/products/invisible/tech-specs/
        # and https://docs.pupil-labs.com/core/terminology/#coordinate-system
        gaze_pos = gaze_pos * np.array(self.GAZE_RESOLUTION)[np.newaxis, ...]
        filt = (gaze_timestamps >= start_posix) & (gaze_timestamps <= end_posix)
        gaze_pos = gaze_pos[filt]
        is_fixation = is_fixation[filt]

        if len(gaze_pos) == 0:
            logger.warning(f"No gaze data for subject {subject} in the given time interval")
            return np.empty((0, 2), dtype=np.float32)

        if self.undistort_videos:
            # undistort points using camera intrinsics - we also undistort the
            # video with the same intrinsics
            logger.debug(f"Undistorting gaze data with shape {gaze_pos.shape}")
            gaze_pos = cv2.undistortPoints(
                gaze_pos,
                gaze_metadata["camera_matrix"],
                gaze_metadata["dist_coefs"],
                None,
                gaze_metadata["camera_matrix"],
            ).squeeze()

        # Renormalize gaze positions to [0,1]
        gaze_pos = gaze_pos / np.array(self.GAZE_RESOLUTION)

        if self.mask_nonfixations:
            gaze_pos[~is_fixation] = -1

        return gaze_pos

    def _read_world_video(
        self, subject: str, gaze_metadata: dict, start_posix: float, end_posix: float
    ) -> dict[str, np.ndarray]:
        """Read world video for a subject.

        Parameters
        ----------
        subject : str
            Subject name.
        gaze_metadata : dict
            Metadata for the gaze data.
        start_posix : float
            Start time of the video in POSIX time.
        end_posix : float
            End time of the video in POSIX time.

        Returns
        -------
        dict[str, np.ndarray]
            Video and audio data from the video.
        """
        video_paths = self.video_samples[subject]

        start_sec = start_posix - gaze_metadata["start_time_video"]
        end_sec = end_posix - gaze_metadata["start_time_video"]
        video_data = self._read_video(video_paths["video"], start_sec, end_sec)

        data = {}

        if "video" in video_data:
            data["video"] = video_data["video"]
            if self.undistort_videos:
                data["video"] = self._undistort_video(
                    data["video"],
                    gaze_metadata["camera_matrix"],
                    gaze_metadata["dist_coefs"],
                )

        if "audio" in video_data:
            data["audio"] = video_data["audio"]

        return data

    @_set_num_workers
    def _undistort_video(
        self, video: np.ndarray, camera_matrix: np.ndarray, dist_coefs: np.ndarray
    ) -> np.ndarray:
        """Undistort a video.

        Parameters
        ----------
        video : np.ndarray
            Video to undistort.
        camera_matrix : np.ndarray
            Camera matrix.
        dist_coefs : np.ndarray
            Distortion coefficients.

        Returns
        -------
        np.ndarray
            Undistorted video.
        """
        logger.info(f"Undistorting video with shape {video.shape}")
        if self.num_workers == 1:
            # use a single process to undistort the video
            frames = [
                cv2.undistort(
                    frame.transpose(1, 2, 0),
                    camera_matrix,
                    dist_coefs,
                    None,
                    camera_matrix,
                )
                for frame in video
            ]
        else:
            # use multiprocessing to speed up the undistortion
            with mp.Pool(self.num_workers) as pool:
                frames = [frame.transpose(1, 2, 0) for frame in video]
                frames = pool.starmap(
                    cv2.undistort,
                    [
                        (
                            frame,
                            camera_matrix,
                            dist_coefs,
                            None,
                            camera_matrix,
                        )
                        for frame in frames
                    ],
                )

        return np.array(frames).transpose(0, 3, 1, 2)

    @torch.no_grad()
    @_set_num_workers
    def _combine_data(self, left_data, right_data, corr_gps, gps_start, gps_end):
        data = {}

        if self.with_video:
            data["left_video"] = left_data["video"]
            data["right_video"] = right_data["video"]

        if self.with_audio:
            data["left_audio"] = left_data["audio"]
            data["right_audio"] = right_data["audio"]

        full_corrected_gps = self._get_full_corrected_gps(corr_gps)
        # full_corrected_gps is a pandas Dataframe indexed by timestamp
        # in POSIX time which can be accessed by range index
        data["gps"] = full_corrected_gps[gps_start:gps_end].values

        return data

    def _get_full_corrected_gps(self, file: Path) -> pd.DataFrame:
        # check the cache first
        if file in self.corrected_gps_cache:
            return self.corrected_gps_cache[file]

        # read the corrected GPS data
        corrected_gps = self._interpolate_corrected_gps(file)
        self.corrected_gps_cache[file] = corrected_gps
        return corrected_gps

    def _smoothly_interpolate_gps(self, gps_data, start, end):
        # sort by timestamp
        gps_data = gps_data[gps_data[:, 0].argsort()]
        gps_data[:, 0] += 1e-6 * np.arange(gps_data.shape[0])
        # fit an interolator to each column
        logger.info("Interpolating GPS data to align with video frames")
        weights = (1 / gps_data[:, 1]) ** 2
        xi = np.arange(
            start,
            end,
            1 / self.output_fps,
        )
        gps, smoothing_result = csaps(
            gps_data[:, 0],
            gps_data[:, 2:],
            xi,
            weights=weights,
            normalizedsmooth=True,
            axis=0,
        )
        logger.info(f"GPS smoothing result: {smoothing_result}")
        return gps

    def _read_video_metadata(self, file: Path) -> Dict[str, Any]:
        """Read metadata for a video file.

        Currently returns video FPS and total duration. Note that
        duration is the total duration of the longest stream in the
        video. Therefore, if the stream includes GPS data, the duration
        will be at least 1 second, and possibly in 1-second increments.

        Parameters
        ----------
        file : Path
            Path to the video file.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the video metadata.
        """
        with av.open(str(file)) as container:
            # Timecode info, as well as the TCD track is incorrect.
            # We need to read the GPS track to get the correct start time.
            try:
                gps_stream_id = self._get_gps_stream_id(container)
                logger.debug(f"GPS stream ID of {file}: {gps_stream_id}")
                raw_gps_data = self._read_data_track(gps_stream_id, str(file), 0, 10)
                gps_data_objects = gpmf.parseStream(raw_gps_data)
                logger.debug(f"Read {len(gps_data_objects)} GPS data objects")
                gps_points, _ = self._build_gps_points(gps_data_objects)
                start_time = gps_points[0].time.replace(tzinfo=timezone.utc).timestamp()
            except (ValueError, av.AVError) as e:
                logger.warning(
                    f"Could not find GPS data in {file}. "
                    f"Using start time of 0 for video. "
                    f"Error: {e}"
                )
                start_time = 0

            metadata = {
                "duration": container.duration / 1e6,
                "video_fps": container.streams.video[0].average_rate,
                "start_time": start_time,
            }
            logger.debug(f"Metadata for {file}: {metadata}")

        return metadata

    @_set_num_workers
    def _read_video(
        self, file: Path, start_sec: float = 0.0, end_sec: float = float("inf")
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read a video from a file, return an iterable of desired frames.

        This is a reimplementation of the torchvision.io.read_video function, which
        does not admit threading or reading metadata.

        Parameters
        ----------
        file : Path
            Path to the video file.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the video data (frames, audio, gps).
        """
        video_frames = []
        audio_frames = []
        gps_frames = []

        try:
            with av.open(str(file)) as container:
                video_frames, audio_frames, gps_frames = self._extract_frames(
                    file, start_sec, end_sec, container
                )

        except av.AVError as e:
            logger.warning(f"Error reading video {file}, error message from PyAV: {e}")
            pass

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]
        gframes_list = [np.array(frame) for frame in gps_frames]

        logger.debug(f"Read {len(vframes_list)} video frames")
        logger.debug(f"Read {len(aframes_list)} audio frames")
        logger.debug(f"Read {len(gframes_list)} gps frames")

        data_dict = {}

        if self.with_video:
            if vframes_list:
                vframes = np.stack(vframes_list, dtype=np.uint8)
                if self.output_format == "TCHW":
                    # [T,H,W,C] --> [T,C,H,W]
                    vframes = vframes.transpose(0, 3, 1, 2)
                data_dict["video"] = vframes
            else:
                logger.warning("No video frames found, returning empty tensor")
                data_dict["video"] = np.empty((0, 3, 0, 0), dtype=np.uint8)

        if self.with_audio:
            if aframes_list:
                aframes = np.concatenate(aframes_list, 1, dtype=np.float32)
                # Convert to mono
                aframes = aframes.mean(0, keepdims=True).T
                data_dict["audio"] = aframes
            else:
                logger.warning("No audio frames found, returning empty tensor")
                data_dict["audio"] = np.empty((0, 0), dtype=np.float32)

        if gframes_list:
            gframes = np.stack(gframes_list)
        else:
            logger.warning("No GPS data found, returning empty tensor")
            gframes = np.empty((0, 4))
        data_dict["gps"] = gframes

        return data_dict

    def _extract_frames(self, file, start_sec, end_sec, container):
        video_frames = []
        audio_frames = []
        gps_frames = []

        if container.streams.video and self.with_video:
            container.streams.video[0].thread_count = 0
            ctx = None
            if self.with_gpu_codec:
                ctx = av.Codec("h264_cuvid", "r").create()
                ctx.extradata = container.streams.video[0].codec_context.extradata
            video_frames = self._read_from_stream(
                container,
                ctx,
                start_sec,
                end_sec,
                "sec",
                container.streams.video[0],
                {"video": 0},
            )

            if self.output_fps != self.VIDEO_FPS:
                logger.info(f"Resampling video to {self.output_fps} FPS")
                video_frames = video_frames[:: int(self.VIDEO_FPS / self.output_fps)]

        if container.streams.audio and self.with_audio:
            container.streams.audio[0].thread_count = 0
            audio_frames = _read_from_stream(
                container,
                start_sec,
                end_sec,
                "sec",
                container.streams.audio[0],
                {"audio": 0},
            )

        # This code is not absolutely necessary, but it does not have a huge
        # performance impact, so we can leave it here for anybody who might
        # need the original GPS data
        if container.streams.data:
            # Then this is the GoPro video
            # find the ID of the container with GPS data
            # which is the one that has the handler_name GoPro MET
            gps_stream_id = self._get_gps_stream_id(container)

            # read data stream directly, do not rely on the _read_from_stream
            # function because it does not support data streams
            raw_gps_data = self._read_data_track(gps_stream_id, str(file), start_sec, end_sec)
            gps_data_objects = gpmf.parseStream(raw_gps_data)
            gps_points, dilutions = self._build_gps_points(gps_data_objects)
            gps_frames = [
                (
                    gps_point.time.replace(tzinfo=datetime.timezone.utc).timestamp(),
                    dilution,
                    gps_point.latitude,
                    gps_point.longitude,
                    gps_point.speed,
                    gps_point.elevation,
                )
                for gps_point, dilution in zip(gps_points, dilutions)
            ]

        # https://github.com/PyAV-Org/PyAV/issues/1117
        logger.debug("Closing streams")
        try:
            if container.streams.video and self.with_video:
                container.streams.video[0].close()
            if container.streams.audio and self.with_audio:
                container.streams.audio[0].close()
            if container.streams.data:
                container.streams.data[gps_stream_id].close()
        except Exception as e:
            logger.warning(f"Error closing streams, error message from PyAV: {e}")
        container.close()
        logger.debug("Streams closed")
        logger.info(f"Read video and closed streams for {file} from {start_sec} to {end_sec}")

        return video_frames, audio_frames, gps_frames

    def _read_from_stream(
        self,
        container: "av.container.Container",
        ctx: "av.codec.Codec",
        start_offset: float,
        end_offset: float,
        pts_unit: str,
        stream: "av.stream.Stream",
        stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
    ) -> List["av.frame.Frame"]:
        gc.collect()

        if pts_unit == "sec":
            # TODO: we should change all of this from ground up to simply take
            # sec and convert to MS in C++
            start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
            if end_offset != float("inf"):
                end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
        else:
            warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

        frames = {}
        should_buffer = True
        max_buffer_size = 5
        if stream.type == "video":
            # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
            # so need to buffer some extra frames to sort everything
            # properly
            extradata = stream.codec_context.extradata
            # overly complicated way of finding if `divx_packed` is set, following
            # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
            if extradata and b"DivX" in extradata:
                # can't use regex directly because of some weird characters sometimes...
                pos = extradata.find(b"DivX")
                d = extradata[pos:]
                o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
                if o is None:
                    o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
                if o is not None:
                    should_buffer = o.group(3) == b"p"
        seek_offset = start_offset
        # some files don't seek to the right location, so better be safe here
        seek_offset = max(seek_offset - 1, 0)
        if should_buffer:
            # FIXME this is kind of a hack, but we will jump to the previous keyframe
            # so this will be safe
            seek_offset = max(seek_offset - max_buffer_size, 0)
        try:
            # TODO check if stream needs to always be the video stream here or not
            container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
        except av.AVError:
            # TODO add some warnings in this case
            logger.warning("Could not seek to the desired location")
            return []
        buffer_count = 0
        try:
            if ctx is not None:
                for packet in container.demux(stream):
                    for frame in ctx.decode(packet):
                        frames[frame.pts] = frame
                        if frame.pts >= end_offset:
                            if should_buffer and buffer_count < max_buffer_size:
                                buffer_count += 1
                                continue
                            break
                    else:
                        continue
                    break
            else:
                for _idx, frame in enumerate(container.decode(**stream_name)):
                    frames[frame.pts] = frame
                    if frame.pts >= end_offset:
                        if should_buffer and buffer_count < max_buffer_size:
                            buffer_count += 1
                            continue
                        break
        except av.AVError:
            logger.warning("Error reading frames from the video")
            pass
        # ensure that the results are sorted wrt the pts
        result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
        if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
            # if there is no frame that exactly matches the pts of start_offset
            # add the last frame smaller than start_offset, to guarantee that
            # we will have all the necessary data. This is most useful for audio
            preceding_frames = [i for i in frames if i < start_offset]
            if len(preceding_frames) > 0:
                first_frame_pts = max(preceding_frames)
                result.insert(0, frames[first_frame_pts])
        return result

    def _get_gps_stream_id(self, container):
        gps_stream_id = None

        for i, stream in enumerate(container.streams.data):
            if self.GPS_STREAM_HANDLER in stream.metadata.get("handler_name"):
                gps_stream_id = i
                break

        if gps_stream_id is None:
            available_streams = [stream.metadata for stream in container.streams.data]
            logger.error("No GPS stream found in the video file")
            logger.info(f"Available streams: {available_streams}")
            raise ValueError("No GPS stream found in the video file")

        return gps_stream_id

    def _build_gps_points(self, data):
        """Build GPS points from the data stream.

        Adapted and simplified from BuildGPSPoints in gopro2gpx package.

        Data comes in the form of a stream and processed through
        a finite state machine. Also, it is scaled by the SCAL
        label.

        See https://github.com/gopro/gpmf-parser.
        See https://exiftool.org/TagNames/GoPro.html for more details.
        Also see https://www.trekview.org/blog/2022/injecting-camm-gpmf-telemetry-videos-part-5-gpmf/. # noqa: E501

        GET
        - SCAL     Scale value
        - GPSF     GPS Fix
        - GPSU     GPS Time
        - GPS5     GPS Data
        """
        points = []
        dilutions = []
        # The values that we keep track of between batches inside a stream.
        # GPSU is the timestamp of the first point in GPS5
        SCAL = fourCC.XYZData(1.0, 1.0, 1.0)
        GPSU = None
        GPSP = None
        GPSFIX = 0  # no lock.

        for d in data:
            if d.fourCC == "SCAL":
                logger.debug(f"SCAL: {d.data}")
                SCAL = d.data
            elif d.fourCC == "GPSU":
                logger.debug(f"GPSU: {d.data}")
                GPSU = d.data
            elif d.fourCC == "GPSF":
                logger.debug(f"GPSF: {d.data}")
                if d.data != GPSFIX:
                    logger.info(f"GPSFIX change to {d.data} ({fourCC.LabelGPSF.xlate[d.data]})")
                GPSFIX = d.data
            elif d.fourCC == "GPSP":
                logger.info(f"Dilution of precision (GPSP) change to {d.data}")
                GPSP = d.data
            elif d.fourCC == "GPS5":
                new_points, new_dilutions = self._parse_gps5_stream(d, SCAL, GPSU, GPSP, GPSFIX)
                points.extend(new_points)
                dilutions.extend(new_dilutions)

        logger.debug(f"Parsed GPS data point count: {len(points)}")
        points = self._fix_timestamps(points)
        # Remove points with dilution of precision > threshold
        filtered_points, filtered_dilutions = self._filter_points_by_dilution(points, dilutions)

        logger.info(f"GPS data points: {len(points)} (OK: {len(filtered_points)})")

        return filtered_points, filtered_dilutions

    def _parse_gps5_stream(self, data, SCAL, GPSU, GPSP, GPSFIX):
        points, dilutions = [], []

        for item in data.data:
            if item.lon == item.lat == item.alt == 0:
                logger.warning("Empty GPS data point, skipping")
                continue

            retdata = [float(x) / float(y) for x, y in zip(item._asdict().values(), list(SCAL))]

            gpsdata = fourCC.GPSData._make(retdata)
            p = gpshelper.GPSPoint(gpsdata.lat, gpsdata.lon, gpsdata.alt, GPSU, gpsdata.speed)
            # GPSU is the timestamp of the first point in GPS5
            # We fill the rest via _fix_timestamps function
            GPSU = None

            points.append(p)

            if GPSFIX == 0:
                dilutions.append(float("inf"))
                logger.warning("GPSFIX=0, skipping")
            else:
                dilutions.append(GPSP)

        return points, dilutions

    def _filter_points_by_dilution(
        self, points, dilutions
    ) -> Tuple[List[gpshelper.GPSPoint], List[float]]:
        filtered_points, filtered_dilutions = [], []
        for i, dilution in enumerate(dilutions):
            if dilution < self.dilution_threshold:
                filtered_points.append(points[i])
                filtered_dilutions.append(dilution)
        return filtered_points, filtered_dilutions

    def _fix_timestamps(self, points: List[gpshelper.GPSPoint]):
        """Fix timestamps of GPS points.

        The FPS of the GPS data is around 18 Hz, and the first
        datapoints of each batch have a timestamp. We interpolate
        the timestamps for the rest of the points in the batch.
        """
        timestamps = [p.time for p in points]
        # We assign a frame rate value for each timestamp
        fps_list = self._estimate_fps(timestamps)

        # Fill missing timestamps using the fps values
        # starting from the first valid timestamp
        last_valid_ts_idx = None
        for ts_idx, (ts, fps) in enumerate(zip(timestamps, fps_list)):
            if ts is not None:
                last_valid_ts_idx = ts_idx
            else:
                if last_valid_ts_idx is not None:
                    timestamps[ts_idx] = timestamps[last_valid_ts_idx] + datetime.timedelta(
                        seconds=(ts_idx - last_valid_ts_idx) / fps
                    )

        # Fill the initial missing timestamps, if any
        # by counting backwards from the first valid timestamp
        first_valid_ts_idx = None
        for ts_idx, ts in enumerate(timestamps):
            if ts is not None:
                first_valid_ts_idx = ts_idx
                break

        if first_valid_ts_idx is None:
            logger.warning("No valid timestamps found")
            logger.info(timestamps)
            return points

        for ts_idx in range(first_valid_ts_idx):
            timestamps[ts_idx] = timestamps[first_valid_ts_idx] - datetime.timedelta(
                seconds=(first_valid_ts_idx - ts_idx) / fps_list[ts_idx]
            )

        # Assign the timestamps to the points
        for ts_idx, ts in enumerate(timestamps):
            points[ts_idx].time = ts

        return points

    def _estimate_fps(self, timestamps):  # noqa: C901
        """Estimate an FPS value for each timestamp.

        This frame rate will then be used to interpolate between
        the missing or erroneous timestamp values.
        """
        fps_list = []
        last_ts_idx = None
        for ts_idx, ts in enumerate(timestamps):
            if ts is not None:
                if last_ts_idx is not None:
                    datapoint_count = ts_idx - last_ts_idx
                    total_seconds = (ts - timestamps[last_ts_idx]).total_seconds()
                    if total_seconds == 0:
                        estimated_fps = np.nan
                    else:
                        estimated_fps = datapoint_count / total_seconds
                    if estimated_fps > 18.5 or estimated_fps < 17.5 or np.isnan(estimated_fps):
                        # remove the invalid timestamps
                        # As we know that the frame rate is somewhat stable
                        # in a single camera run, if it varies too much
                        # in rolling estimates, we assume that the timestamp is
                        # erroneous and remove it. For example, see the first
                        # couple GPSU values in 01GoPro/001/left/GH010008.MP4
                        logger.warning(
                            f"Invalid fps estimated between timestamps "
                            f"{timestamps[last_ts_idx]} and {ts}. "
                            f" Estimated fps: {estimated_fps}. "
                            f"Removing timestamp {last_ts_idx}"
                        )
                        timestamps[last_ts_idx] = None
                        fps_list.append(np.nan)
                    else:
                        logger.info(f"Estimated fps: {estimated_fps}")
                        fps_list.append(estimated_fps)
                else:
                    fps_list.append(np.nan)
                last_ts_idx = ts_idx
            else:
                fps_list.append(np.nan)

        # Fill missing fps values by going backwards
        # find last valid fps first
        last_valid_fps = None
        for fps in reversed(fps_list):
            if not np.isnan(fps):
                last_valid_fps = fps
                break

        # FPS cannot be estimated if there is only a single batch
        # So we use a default value (see https://github.com/gopro/gpmf-parser)
        if last_valid_fps is None:
            last_valid_fps = 18.17

        for fps_idx, fps in enumerate(reversed(fps_list)):
            if np.isnan(fps):
                fps_list[-fps_idx - 1] = last_valid_fps
            else:
                last_valid_fps = fps
        return fps_list

    def _read_data_track(
        self,
        track: int,
        file: Path,
        start_sec: float = 0,
        end_sec: float = float("inf"),
    ) -> bytes:
        """Reimplementation from gopro2gpx package.

        PyAV cannot handle the data tracks properly, so we directly use ffmpeg.

        Parameters
        ----------
        track : int
            The data track index to read. Provided to ffmpeg with the -map option,
            i.e., -map 0:d:%d
        file : Path
            Path to the video file.
        """
        # Convert seconds to ffmpeg time format (HH:MM:SS.sss)
        start_time = str(datetime.timedelta(seconds=start_sec)) if start_sec > 0 else None
        end_time = str(datetime.timedelta(seconds=end_sec)) if end_sec < float("inf") else None

        args = []
        if start_time:
            args += ["-ss", start_time]
        if end_time:
            args += ["-to", end_time]
        args += [
            "-y",
            "-i",
            str(file),
            "-codec",
            "copy",
            "-map",
            "0:d:%d" % track,
            "-f",
            "rawvideo",
            "-",
        ]

        cmd = self._ffmpeg_cmd()
        result = subprocess.run([cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.error(f"ffmpeg stderr: {result.stderr.decode('utf-8')}")
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

        logger.debug(f"Read {len(result.stdout)} bytes of GPS data")

        return result.stdout

    def _ffmpeg_cmd(self):
        if platform.system() == "Windows":
            ffmpeg = "ffmpeg.exe"
        else:
            ffmpeg = "ffmpeg"

        return ffmpeg

    def _convert_gps_coordinates(self, gps_data: np.ndarray) -> np.ndarray:
        """Convert GPS coordinate system from EPSG:4326 to EPSG:3857.

        Parameters
        ----------
        gps_data : np.ndarray
            Numpy array with the GPS data, latitude and longitude ordered.

        Returns
        -------
        np.ndarray
            Numpy array with the converted GPS data.
        """
        return np.apply_along_axis(
            lambda x: self._coordinate_transformer.transform(x[0], x[1]), 1, gps_data
        )
