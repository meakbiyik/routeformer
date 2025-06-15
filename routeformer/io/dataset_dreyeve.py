"""Dataloader for Dreyeve dataset."""

import atexit
import hashlib
import json
import logging
import pickle
import random
import shutil
import sys
import tempfile
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Tuple, TypedDict

import av
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map
import zstd
from natsort import natsorted
from pyproj import Transformer
from pympler import asizeof

# from torchvision.io.image import read_image
from torchvision.io.video import _read_from_stream
from tqdm import tqdm

from routeformer.score import estimate_pci
from routeformer.visualize.plot import plot_gps_data_on_map

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


def _set_num_workers(func):
    """Set the number of threads/workers as decorator.

    Cannot be used to decorate a generator method (e.g., __iter__)
    """

    def wrapper(self: "DreyeveDataset", *args, **kwargs):
        logger.debug("Setting number of threads to %d", self.num_workers)
        prev_num_workers = torch.get_num_threads()
        torch.set_num_threads(self.num_workers)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.set_num_threads(prev_num_workers)

    return wrapper


def time_it(func):
    """Measure the execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        logger.debug(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class TimeIt:
    """Measure the execution time of a function over multiple executions."""

    def __init__(self, num_executions):
        """Measure the execution time of a function over multiple executions.

        Parameters
        ----------
        num_executions : int
            Number of executions to average the execution time over.
        """
        self.num_executions = num_executions

    def __call__(self, func):
        """Measure the execution time of a function over multiple executions."""

        def wrapper(*args, **kwargs):
            total_execution_time = 0
            for i in range(self.num_executions):
                start_time = time()
                result = func(*args, **kwargs)
                end_time = time()
                execution_time = end_time - start_time
                total_execution_time += execution_time
                logger.debug(
                    f"Execution time of {func.__name__}: {execution_time:.6f} seconds"
                )
            avg_execution_time = total_execution_time / self.num_executions
            logger.debug(
                f"""Average execution time of {func.__name__}:
                  {avg_execution_time:.6f} seconds"""
            )
            return result

        return wrapper


class DreyeveFileStructure:
    """The base filestructure of the Dreyeve dataset."""

    SESSION_ID = "SESSION"

    def __init__(self, root_dpath) -> None:
        """Build the base filestructure of the Dreyeve dataset."""
        self.root_dpath = Path(root_dpath).resolve()
        self.session_id_format = "{:02d}"
        self.subsequences_fpath = self.root_dpath / "subsequences.txt"
        self.design_fpath = self.root_dpath / "dr(eye)ve_design.txt"

        session_dpath = self.root_dpath / self.SESSION_ID
        self.mean_frame_fpath = session_dpath / "mean_frame.png"

        self.mean_gt_fpath = session_dpath / "mean_gt.png"
        self.etg_samples_fpath = session_dpath / "etg_samples.txt"
        self.speed_course_fpath = session_dpath / "speed_course_coord.txt"
        self.video_etg_fpath = session_dpath / "video_etg.avi"
        self.video_garmin_fpath = session_dpath / "video_garmin.avi"
        self.video_sailency_fpath = session_dpath / "video_sailency.avi"
        self.video_etg_frames_fpath = session_dpath / "video_etg_frames" / "{:06d}.jpg"
        self.video_garmin_frames_fpath = (
            session_dpath / "video_garmin_frames" / "{:06d}.jpg"
        )

    def get_session_ids(self):
        """Get the session IDs."""
        return [int(d.name) for d in natsorted(self.root_dpath.iterdir()) if d.is_dir()]


def to_frames(video_fpath, frames_fpath: Path):
    """Convert a video to frames and save them in a folder."""
    frames_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(video_fpath, "rb") as f:
        container = av.open(f)
        container.streams.video[0].thread_type = "AUTO"
        container.streams.video[0].thread_count = 0
        for i, frame in enumerate(container.decode(video=0)):
            # logger.debug(f"Frame {i}", frames_fpath.format(i))
            frame.to_image().save(str(frames_fpath).format(i))
        container.streams.video[0].close()
        container.close()


class DreyeveFileStructureSession:
    """The file structure of a single session."""

    @staticmethod
    def replace(path: str | Path, old_str: str, new_str: str):
        """Replace a string in a Path."""
        new_path = Path(str(path).replace(old_str, new_str))
        return new_path

    def __init__(self, root: [str | Path], session_id: int) -> None:
        """Set the file structure of a session."""
        self.fs = DreyeveFileStructure(root)
        self.session_id = session_id
        self.session_id_label = self.fs.session_id_format.format(session_id)
        self.mean_frame_fpath = self.replace(
            self.fs.mean_frame_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.mean_gt_fpath = self.replace(
            self.fs.mean_gt_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.etg_samples_fpath = self.replace(
            self.fs.etg_samples_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.speed_course_fpath = self.replace(
            self.fs.speed_course_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.video_etg_fpath = self.replace(
            self.fs.video_etg_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.video_garmin_fpath = self.replace(
            self.fs.video_garmin_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.video_sailency_fpath = self.replace(
            self.fs.video_sailency_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.video_etg_frames_fpath = self.replace(
            self.fs.video_etg_frames_fpath, self.fs.SESSION_ID, self.session_id_label
        )
        self.video_garmin_frames_fpath = self.replace(
            self.fs.video_garmin_frames_fpath, self.fs.SESSION_ID, self.session_id_label
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return str(self.__dict__)

    def build_frames(self):
        """Build the frames of the videos."""
        to_frames(self.video_etg_fpath, self.video_etg_frames_fpath)
        to_frames(self.video_garmin_fpath, self.video_garmin_frames_fpath)


# create an enumerator for the scenes
class DreyeveDesignScene:
    """Scene enumerator."""

    DOWNTOWN = "Downtown"
    HIGHWAY = "Highway"
    COUNTRYSIDE = "Countryside"


# create an enumerator for all the possible weather conditions
class DreyeveDesignWeather:
    """Scene enumerator."""

    SUNNY = "Sunny"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"


# create an enumerator for all the possible day times
class DreyeveDesignTime:
    """Scene enumerator."""

    MORNING = "Morning"
    EVENING = "Evening"
    NIGHT = "Night"


class DreyeveFileStructureSessionLibrary:
    """A collection of all subject file structures in the Dreyeve dataset."""

    def __init__(self, root: str | Path) -> None:
        """Build a collection of all subject file structures in the Dreyeve dataset."""
        self.fs = DreyeveFileStructure(root)
        self.sessions = {
            i: DreyeveFileStructureSession(root, i) for i in self.fs.get_session_ids()
        }

        self.data_design = pd.read_csv(
            self.fs.design_fpath,
            sep="\t",
            header=None,
            names=["session_id", "time", "weather", "scene", "subject", "set"],
            dtype={
                "session_id": "int32",
                "scene": "string",
                "weather": "string",
                "time": "string",
                "subject": "string",
                "set": "string",
            },
            index_col=None,
        )

    def __getitem__(self, key: int) -> DreyeveFileStructureSession:
        """Get a subject file structure by its ID."""
        return self.sessions[key]

    def __iter__(self):
        """Iterate over the subjects."""
        return iter(self.sessions.values())

    def build_frames(self):
        """Build the frames of the videos."""
        process_map(
            partial(DreyeveFileStructureSession.build_frames),
            self.sessions.values(),
            max_workers=12,
        )

    def __len__(self):
        """Return the number of subjects."""
        return len(self.sessions)


def worker_indexer(args):
    """Load a data entry in memory."""
    self, entry = args
    return self[entry]


class DreyeveDataset(torch.utils.data.Dataset):
    """A handler of the Dreyeve dataset to be used with a dataloader."""

    PCI_VERSION = 1
    DATA_CACHE_VERSION = 3.2
    DATA_SPLIT = {
        "train": list(range(1, 45)),
        "val": list(range(45, 60)),
        "train+val": list(range(1, 60)),
        "test": list(range(60, 75)),
    }

    def __init__(
        self,
        root_dir: str | Path = "~/projects/gaze/routeformer/data/dreyeve/DREYEVE_DATA",
        split: Literal["train", "val", "test"] | List[str] = "train",
        input_length: float = 8,
        target_length: float = 6,
        step_size: float = 2,
        min_pci: float = 0,
        max_pci: float = None,
        output_fps: float = 5,
        gopro_scaling_factor: float = 1.0,
        front_scaling_factor: float = 1.0,
        output_format: str = "TCHW",
        use_cache: bool = False,
        cache_dir: str | Path | None = None,
        build_frames=False,
        max_cache_size: int = int(10e9),
        use_frames=True,
        use_memory_cache=False,
        max_memory_cache_size: int = int(100e9),
        with_video=True,
        crop_videos=True,
        enable_pci_split=False,
        pci_split_n_samples_per_bin=200,
        max_length=None,
        seed: int = 4242,
        filter_scene: List[DreyeveDesignScene] = None,
    ):
        """Handle the Dreyeve dataset to be used with a dataloader.

        Parameters
        ----------
        root_dir : str | Path, optional
            Path to the root directory of the dataset,
            by default "~/projects/gaze/routeformer/data/dreyeve/DREYEVE_DATA"
        split : Literal["train", "val", "test"] | List[str], optional
            Split of the dataset to use, by default "train"
        input_length : float, optional
            Length of the input sequence in seconds, by default 8
        target_length : float, optional
            Length of the target sequence in seconds, by default 6
        step_size : float, optional
            Step size in seconds, by default 2
        min_pci : float, optional
            Minimum pci of the data, by default None
        max_pci : float, optional
            Maximum pci of the data, by default None
        output_fps : float, optional
            FPS of the output, by default 5
        gopro_scaling_factor : float, optional
            Scaling factor for the GoPro video, by default 1.0
        front_scaling_factor : float, optional
            Scaling factor for the front video, by default 1.0
        output_format : str, optional
            Output format of the data, by default "TCHW"
        use_cache : bool, optional
            Whether to use the cache, by default False
        cache_dir : str | Path | None, optional
            Path to the cache directory, by default None
        build_frames : bool, optional
            Whether to build the frames, by default False
        max_cache_size : int, optional
            Maximum size of the cache in bytes, by default int(10e9)
        use_frames : bool, optional
            Whether to use the frames, by default True
        use_memory_cache : bool, optional
            Whether to use the memory cache, by default True
        with_video : bool, optional
            Whether to load the video and gaze data, by default False
        enable_pci_split : bool, optional
            Whether to enable the pci split, by default False
        pci_split_n_samples_per_bin : int, optional
            Number of samples per bin in the pci split, by default 200
        seed : int, optional
            Random seed, by default 4242
        crop_videos : bool, optional
            Whether to crop the videos, by default True
        filter_scene : List[DreyeveDesignScene], optional
            List of dreyeve scene types to use in training, by default None
        max_length : int, optional
            Maximum number of data entries to use, by default None
        """
        super().__init__()
        random.seed(seed)

        self.index_column = "frame_gar"
        self.input_fps = 30
        self.output_fps = output_fps
        self.fps_divisor = self.input_fps // self.output_fps
        self.step_size = step_size
        self.undistort_videos = False
        self.use_memory_cache = use_memory_cache
        self.max_memory_cache_size = max_memory_cache_size
        self.with_video = with_video
        self.crop_videos = crop_videos
        self.enable_pci_split = enable_pci_split
        self.filter_scene = filter_scene
        self.max_length = max_length

        # Check if pci split is running on 1 GPU

        assert (
            self.fps_divisor > 0 and self.input_fps % self.fps_divisor == 0
        ), "fps_divisor must divide output_fps"

        self.use_cache = use_cache  # the metadata cache
        self.use_pci_cache = self.use_cache  # the pci cache
        self.use_data_cache = use_cache and with_video  # the getitem cache
        self.max_cache_size = max_cache_size
        self.cache_dpath = Path(cache_dir) / "dreyeve_dataset"
        self.cache_metadata_fpath = self.cache_dpath / "metadata.json"
        self.cache_pci_fpath = self.cache_dpath / (
            f"pci_stepsize-{self.step_size}.json"
            if self.step_size != 1
            else "pci.json"
        )
        self.cache_gps_metadata_fpath = self.cache_dpath / "gps_metadata.json"
        self.use_frames = use_frames
        self.gopro_scaling_factor = gopro_scaling_factor
        self.front_scaling_factor = front_scaling_factor

        if self.use_cache:
            self._cache_size = 0
            if self.cache_dpath is None:
                self.cache_dpath = Path(tempfile.mkdtemp())
                atexit.register(lambda: shutil.rmtree(self.cache_dpath))
            else:
                self.cache_dpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache at {self.cache_dpath}")

        self.split = split if isinstance(split, list) else self.DATA_SPLIT[split]

        self.output_format = output_format
        seq_length_in_seconds = input_length + target_length
        self.fs_sessions = DreyeveFileStructureSessionLibrary(root_dir)
        assert len(self.fs_sessions) > 0, "No data found!"

        if (
            build_frames
            or not self.fs_sessions[1].video_etg_frames_fpath.parent.exists()
            or not self.fs_sessions[1].video_garmin_frames_fpath.parent.exists()
        ):
            logger.info("Building frames...")
            self.fs_sessions.build_frames()
        self.seq_length = int(self.input_fps / self.fps_divisor) * seq_length_in_seconds
        self.seq_length_input = int(self.input_fps / self.fps_divisor) * input_length
        self.seq_length_target = int(self.input_fps / self.fps_divisor) * target_length
        self.min_pci = min_pci
        self.max_pci = max_pci

        self._coordinate_transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        self.metadata = self.__generate_metadata(filter_scene=self.filter_scene)
        step_size_frames = self.step_size * self.input_fps
        logger.info("Building Data...")
        self.data = self._build_data(
            metadata=self.metadata,
            seq_length=self.seq_length,
            step_size_frames=step_size_frames,
            fps_divisor=self.fps_divisor,
        )

        bin_step_size = 10
        max_bin = 70
        n_samples_per_bin = pci_split_n_samples_per_bin  # 200  # 478
        n_samples_per_bin_val = 60
        self.data_bins = {}

        self.data = [
            entry
            for entry in self.data
            if entry["pci"] >= self.min_pci
        ]
        if self.enable_pci_split:
            self.data = sorted(self.data, key=lambda x: x["pci"])
            (
                self.data_bins,
                self.data_bins_keys,
                self.bin_epoch_size,
            ) = self.__build_pci_split(
                bin_step_size,
                max_bin,
                n_samples_per_bin_val,
                n_samples_per_bin,
                split,
                self.data,
            )

        logger.info(f"Number of data entries: {len(self.data)}")

        self.data = np.array(self.data)

        self.full_dataset = {}
        self.memory_cache_size = 0

        # A helper to return the sample info for a given index
        # Used internally. Please refer to get_with_info() method.
        self._return_info = False

    def __build_pci_split(
        self,
        bin_step_size,
        max_bin,
        n_samples_per_bin_val,
        n_samples_per_bin,
        split,
        data,
    ):
        bin_skip_size = self.min_pci // bin_step_size
        data_bins = {}
        for entry in data:
            if entry["pci"] <= max_bin:
                bin_key = int(entry["pci"] // bin_step_size) - bin_skip_size
            else:
                bin_key = max_bin // bin_step_size - bin_skip_size
            if bin_key not in data_bins:
                data_bins[bin_key] = []
            data_bins[bin_key].append(entry)

        bin_epoch_size = None
        if split == "train":
            bin_epoch_size = n_samples_per_bin * len(data_bins.keys())
            for key in data_bins.keys():
                # print("Bin", key*bin_step_size, "has", len(data_bins[key]), "entries")
                random.shuffle(data_bins[key])
        elif split == "val":
            bin_min_length = min(
                n_samples_per_bin_val,
                np.array([len(data_bins[key]) for key in data_bins.keys()]).min(),
            )
            bin_epoch_size = bin_min_length * len(data_bins.keys())
            for key in data_bins.keys():
                random.shuffle(data_bins[key])
                data_bins[key] = data_bins[key][:bin_min_length]

        data_bins_keys = natsorted(data_bins.keys())
        return data_bins, data_bins_keys, bin_epoch_size

    def __generate_metadata(self, filter_scene: List[DreyeveDesignScene] = None):
        if self.use_cache and self.cache_metadata_fpath.exists():
            logger.info("Reading metadata from cache...")
            self.metadata = {}
            data_json = pd.read_json(self.cache_metadata_fpath)
            for i, row in data_json.iterrows():
                self.metadata[row["keys"]] = pd.DataFrame(row["values"])

            # self.gps_metadata = {}
            # data_json = pd.read_json(self.cache_gps_metadata_fpath)
            # for i, row in data_json.iterrows():
            #     self.gps_metadata[row["keys"]] = pd.DataFrame(row["values"])
        else:
            logger.info("Generating metadata...")
            self.metadata = {}
            self.gps_metadata = {}
            for session in self.fs_sessions:
                # self.metadata[subject.subject_id] = {}

                # READ GAZE METADATA
                gaze_metadata = pd.read_csv(
                    session.etg_samples_fpath,
                    sep=" ",
                    header=None,
                    names=[
                        "frame_etg",
                        "frame_gar",
                        "X",
                        "Y",
                        "event_type",
                        "timestamp",
                    ],
                    dtype={
                        "frame_etg": "int32",
                        "frame_gar": "int32",
                        "X": "float64",
                        "Y": "float64",
                        "event_type": "string",
                        "timestamp": "int32",
                    },
                    skiprows=1,
                )

                gaze_metadata["X"] = gaze_metadata["X"].interpolate()  # .fillna(method='ffill', inplace=True)
                gaze_metadata["Y"] = gaze_metadata["Y"].interpolate()
                # self.metadata[subject.subject_id]["gaze"] = gaze_metadata

                # group by garmin camera frames
                n_gaze_readings_per_frame = 2

                def first_n(x):
                    return (
                        x.iloc[:n_gaze_readings_per_frame].tolist()
                        if len(x.iloc[:n_gaze_readings_per_frame]) >= 2
                        else [x.iloc[0], x.iloc[0]]
                    )

                gaze_metadata = gaze_metadata.groupby("frame_gar", group_keys=True)
                gaze_metadata = gaze_metadata.agg(
                    {
                        "frame_etg": "first",
                        "X": first_n,
                        "Y": first_n,
                        "event_type": first_n,
                        "timestamp": first_n,
                    }
                )
                gaze_metadata = gaze_metadata.reset_index()

                # READ GPS METADATA
                gps_metadata = pd.read_csv(
                    session.speed_course_fpath,
                    sep="\t",
                    header=None,
                    names=["frame", "speed", "course", "lat", "lon"],
                    dtype={
                        "frame": "int32",
                        "speed": "float",
                        "course": "float",
                        "lat": "float64",
                        "lon": "float64",
                    },
                    index_col=False,
                )
                # self.gps_metadata[session.session_id] = gps_metadata.copy()
                # self.gps_metadata[session.session_id].dropna(subset=['lat', 'lon'],
                # how='any', inplace=True)

                gps_metadata[["lat", "lon"]] = self._convert_gps_coordinates(
                    gps_metadata[["lat", "lon"]].values
                )
                gps_metadata["course"] = gps_metadata["course"].interpolate()  # fillna(method='ffill', inplace=True)
                gps_metadata["speed"] = gps_metadata["speed"].interpolate()  # fillna(method='ffill', inplace=True)
                gps_metadata["lat"] = gps_metadata["lat"].interpolate(
                    limit_area="inside", method="pchip"
                )
                gps_metadata["lon"] = gps_metadata["lon"].interpolate(
                    limit_area="inside", method="pchip"
                )
                gps_metadata = gps_metadata.dropna(subset=["lat", "lon"], how="any")

                # self.metadata[subject.subject_id]["gps"] = gps_metadata
                self.metadata[session.session_id] = gaze_metadata.join(
                    gps_metadata.set_index("frame"), on=self.index_column, how="inner"
                ).reset_index(drop=True)

            if self.use_cache:
                self.cache_dpath.parent.mkdir(parents=True, exist_ok=True)
                data_frame = pd.DataFrame(
                    {
                        "keys": list(self.metadata.keys()),
                        "values": list(self.metadata.values()),
                    }
                )
                data_frame.to_json(self.cache_metadata_fpath)

                # data_frame = pd.DataFrame({"keys":list(self.gps_metadata.keys()),
                #  "values":list(self.gps_metadata.values())})
                # data_frame.to_json(self.cache_gps_metadata_fpath)

                # metadata_frame = pd.DataFrame.from_dict(self.metadata,orient='index',
                #  columns=['value'])
                # metadata_frame.to_json(self.cache_metadata_fpath)
                # metadata_dict = {}
                # for session_id, session_metadata in self.metadata.items():
                #     metadata_dict[session_id] =
                #  session_metadata.to_dict(orient='list')

                # with open(self.cache_metadata_fpath, 'w') as f:
                #     json.dump(metadata_dict, f)

        if filter_scene is not None:
            data_design = self.fs_sessions.data_design
            filtered_ids = data_design[data_design["scene"].isin(filter_scene)][
                "session_id"
            ].tolist()
            self.metadata = {
                key: value
                for key, value in self.metadata.items()
                if key in filtered_ids
            }

        metadata = {
            k: self.metadata[k] for k in self.metadata.keys() if k in self.split
        }
        return metadata

    def _extract_frames(self, start_frame, end_frame, container):
        """Extract frames from a video container."""
        video_frames = []

        container.streams.video[0].thread_type = "AUTO"
        container.streams.video[0].thread_count = 0
        video_frames = _read_from_stream(
            container,
            start_frame,
            end_frame,
            "frame",
            container.streams.video[0],
            {"video": 0},
        )

        # if self.output_fps != self.VIDEO_FPS:
        #     logger.info(f"Resampling video to {self.output_fps} FPS")
        #     video_frames = video_frames[:: int(self.VIDEO_FPS / self.output_fps)]

        # https://github.com/PyAV-Org/PyAV/issues/1117
        logger.debug("Closing streams")
        try:
            if container.streams.video:
                container.streams.video[0].close()
        except Exception as e:
            logger.warning(f"Error closing streams, error message from PyAV: {e}")
        container.close()
        logger.debug("Streams closed")

        return video_frames

    def _read_frames(
        self, file: Path, start_frame_id: int, num_frames: int, fps_divisor: int = 1
    ) -> List[av.VideoFrame]:
        """Read frames from a video file.

        Parameters
        ----------
        file : Path
            Path to the video file.
        frame_idx : int
            Index of the first frame to read.
        num_frames : int
            Number of frames to read.

        Returns
        -------
        List[av.VideoFrame]
            List with the video frames
        """
        frames = []
        with av.open(str(file)) as container:
            container.seek(start_frame_id)

            # Read the desired number of frames
            for i in range(num_frames):
                frame = container.decode(video=0)
                if not frame:
                    break
                if i % fps_divisor == 0:
                    # convert frame to RGB
                    frame = frame.to_rgb().to_ndarray()
                    frames.append(frame)

        # frames = [frame.to_rgb().to_ndarray() for frame in frames]
        return frames

    # @_set_num_workers
    def _read_video(
        self, file: Path, start_frame: int, end_frame: int, fps_divisor: int = 1
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

        try:
            with av.open(str(file)) as container:
                video_frames = self._extract_frames(start_frame, end_frame, container)

        except av.AVError as e:
            logger.warning(f"Error reading video {file}, error message from PyAV: {e}")
            pass

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]

        logger.debug(f"Read {len(vframes_list)} video frames")

        if vframes_list:
            vframes = np.stack(vframes_list, dtype=np.uint8)[::fps_divisor]
            if self.output_format == "TCHW":
                # [T,H,W,C] --> [T,C,H,W]
                vframes = vframes.transpose(0, 3, 1, 2)
            data_result = vframes
        else:
            logger.warning("No video frames found, returning empty tensor")
            data_result = np.empty((0, 3, 0, 0), dtype=np.uint8)

        return data_result

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

    @time_it
    def _build_data(self, metadata, seq_length, step_size_frames, fps_divisor=1):
        """Build the data entries to be iterated over."""
        # load pci data from cache
        should_rebuild_pci = True
        is_cache_invalidated = False
        pci_cache_fpath = self.cache_pci_fpath
        if self.use_pci_cache and pci_cache_fpath.exists():
            logger.info("Reading pci from cache...")
            with open(pci_cache_fpath, "r") as f:
                pci_dict = json.load(f)
                should_rebuild_pci = False
                if (
                    pci_dict["seq_length_full"] != seq_length * fps_divisor
                    or pci_dict["step_size"] != step_size_frames
                    or "version" not in pci_dict.keys()
                    or pci_dict["version"] != self.PCI_VERSION
                ):
                    should_rebuild_pci = True

        if should_rebuild_pci:
            is_cache_invalidated = True
            pci_dict = {
                "version": self.PCI_VERSION,
                "seq_length_full": seq_length * fps_divisor,
                "step_size": step_size_frames,
                "pci": {},
            }

        data = []
        for session_id, session_metadata in metadata.items():
            if (
                should_rebuild_pci
                or str(session_id) not in pci_dict["pci"].keys()
            ):
                pci_dict["pci"][str(session_id)] = {}

            n_frames = session_metadata.shape[0]
            for i in range(0, n_frames - seq_length * fps_divisor, step_size_frames):
                if (
                    should_rebuild_pci
                    or str(i)
                    not in pci_dict["pci"][str(session_id)].keys()
                ):
                    is_cache_invalidated = True
                    input_gps = np.array(
                        session_metadata[["lat", "lon"]][
                            i : i + self.seq_length_input * fps_divisor
                        ]
                    )
                    target_gps = np.array(
                        session_metadata[["lat", "lon"]][
                            i
                            + self.seq_length_input * fps_divisor : i
                            + self.seq_length_input * fps_divisor
                            + self.seq_length_target * fps_divisor
                        ]
                    )
                    pci = estimate_pci(
                        input_gps,
                        target_gps,
                        curve_type="linear",
                        lookback_length=6,
                        frequency=self.output_fps,
                        measure="frechet",
                    )
                    pci_dict["pci"][str(session_id)][
                        str(i)
                    ] = pci
                else:
                    pci = pci_dict["pci"][str(session_id)][
                        str(i)
                    ]

                if (
                    self.min_pci is not None
                    and pci < self.min_pci
                ) or (
                    self.max_pci is not None
                    and pci > self.max_pci
                ):
                    # start_time += self.step_size
                    continue

                entry = {
                    "pci": pci,
                    "session_id": session_id,
                    "start_index": i,
                    # "frame_gar": session_metadata["frame_gar"][i],
                    # "frame_etg": session_metadata["frame_etg"][i],
                    "seq_length": seq_length,
                    "fps_divisor": fps_divisor,
                    # "gaze_metadata": session_metadata[['X', 'Y', 'event_type',
                    #  'timestamp']][i:i+seq_length*fps_divisor:fps_divisor],
                    # "gps_metadata": session_metadata[['speed', 'course', 'lat',
                    # 'lon']][i:i+seq_length*fps_divisor:fps_divisor]
                }
                data.append(entry)

        if self.use_pci_cache and is_cache_invalidated:
            with open(pci_cache_fpath, "w") as f:
                json.dump(pci_dict, f)

        return data

    def plot(self, metadata, output_fpath):
        """Plot the GPS data on a map."""
        traj_df = pd.DataFrame(
            {
                "longitude": metadata["lon"][:50],
                "latitude": metadata["lat"][:50],
            }
        )

        ax = plot_gps_data_on_map(traj_df)
        ax.get_figure().savefig(output_fpath)

    def __read_frames(self, frame_fpath, frame_ids, scaling_factor=1):
        """Read frames from a folder."""

        def worker(frame_id, frame_fpath, scaling_factor=1):
            """Read a single frame from a file."""
            frame = cv2.imread(str(frame_fpath).format(frame_id))
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * scaling_factor),
                    int(frame.shape[0] * scaling_factor),
                ),
                interpolation=cv2.INTER_AREA,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            return frame

        max_workers = cpu_count()
        with ThreadPool(max_workers) as pool:
            frames = pool.starmap(
                worker,
                [(frame_id, frame_fpath, scaling_factor) for frame_id in frame_ids],
            )

        frames = np.stack(frames, axis=0)
        frames = frames.transpose(0, 3, 1, 2)
        return frames

    def _hash_item(self, item: Item) -> str:
        hashstring = repr(item) + (
            # relevant parameters
            repr(self.undistort_videos)
            + repr(self.gopro_scaling_factor)
            + repr(self.front_scaling_factor)
            + repr(self.output_format)
            + repr(self.step_size)
            + repr(self.seq_length_input)
            + repr(self.seq_length_target)
            + repr(self.fps_divisor)
            + repr(self.DATA_CACHE_VERSION)
            if self.DATA_CACHE_VERSION > 0
            else ""
        )
        item_hash = hashlib.blake2b(hashstring.encode(), digest_size=32).hexdigest()
        logger.debug(f"Hashed {hashstring} to {item_hash}")
        return item_hash

    def _fetch_from_cache(self, item: dict) -> Item:
        # hash the item repr deterministically and efficiently
        item_hash = self._hash_item(item)
        # read from cache, which is a zstd-compressed pickle file
        # under cache_dir with the name [hash].pkl.zstd
        candidate_cache_file = self.cache_dpath / f"{item_hash}.pkl.zstd"
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
        candidate_cache_file = self.cache_dpath / f"{item_hash}.pkl.zstd"
        if not candidate_cache_file.exists():
            logger.info(f"Writing to cache {candidate_cache_file}")
            with open(candidate_cache_file, "wb") as f:
                compressed_data = zstd.compress(pickle.dumps(data), 3, 2)
                f.write(compressed_data)
                self._cache_size += len(compressed_data)
        else:
            logger.info(f"Cache file {candidate_cache_file} already exists")

    def __get_uncached_item(self, session_id, start_index, seq_length, fps_divisor):
        session_metadata = self.metadata[session_id]
        gaze_data = np.array(
            [
                [item for item in entry]
                for entry in session_metadata[["X", "Y"]][
                    start_index : start_index + seq_length * fps_divisor : fps_divisor
                ].values
            ],
            dtype=np.float32,
        )
        gps_data = np.array(
            session_metadata[["lat", "lon"]][
                start_index : start_index + seq_length * fps_divisor : fps_divisor
            ].to_numpy()
        )

        gaze_data[:, 0] = gaze_data[:, 0] / 1080
        gaze_data[:, 1] = gaze_data[:, 1] / 720
        gaze_data = gaze_data.transpose(0, 2, 1)
        gaze_seq_length_input = gaze_data.shape[1] * self.seq_length_input
        gaze_data = gaze_data.reshape(-1, 2)
        start_frame_gar = int(session_metadata["frame_gar"].iloc[start_index])
        start_frame_etg = int(session_metadata["frame_etg"].iloc[start_index])

        if self.with_video:
            end_frame_gar = int(
                session_metadata["frame_gar"].iloc[start_index + seq_length * fps_divisor]
            )
            end_frame_etg = int(
                session_metadata["frame_etg"].iloc[start_index + seq_length * fps_divisor]
            )

            n_frames_gar = end_frame_gar - start_frame_gar
            n_frames_etg = end_frame_etg - start_frame_etg

            max_seq = max(n_frames_gar, n_frames_etg)

            frame_ids_gar = list(
                session_metadata["frame_gar"][
                    start_index : start_index + seq_length * fps_divisor
                ]
            )[::fps_divisor]
            frame_ids_etg = list(
                session_metadata["frame_etg"][
                    start_index : start_index + seq_length * fps_divisor
                ]
            )[::fps_divisor]

            if self.use_frames:
                frames_gar = self.__read_frames(
                    self.fs_sessions[session_id].video_garmin_frames_fpath,
                    frame_ids_gar,
                    self.gopro_scaling_factor,
                )
                frames_etg = self.__read_frames(
                    self.fs_sessions[session_id].video_etg_frames_fpath,
                    frame_ids_etg,
                    self.front_scaling_factor,
                )

            else:
                frame_ids_gar = [
                    frame_id - start_frame_gar for frame_id in frame_ids_gar
                ]
                frame_ids_etg = [
                    frame_id - start_frame_etg for frame_id in frame_ids_etg
                ]

                frames_gar = self._read_video(
                    self.fs_sessions[session_id].video_garmin_fpath,
                    start_frame_gar,
                    start_frame_gar + max_seq,
                    1,
                )
                frames_etg = self._read_video(
                    self.fs_sessions[session_id].video_etg_fpath,
                    start_frame_etg,
                    start_frame_etg + max_seq,
                    1,
                )

                frames_gar = frames_gar[[frame_ids_gar], :, :, :]
                frames_etg = frames_etg[[frame_ids_etg], :, :, :]
        else:
            frames_gar = None
            frames_etg = None

        data_input = Data(
            # stitched_video=None,
            # right_video=None,
            # left_audio=None,
            # right_audio=None,
            gps=gps_data[: self.seq_length_input],
            # front_audio=None,
            gaze=gaze_data[:gaze_seq_length_input],
        )

        if self.with_video:
            data_input["left_video"] = frames_gar[: self.seq_length_input]
            data_input["front_video"] = frames_etg[: self.seq_length_input]

        data_target = Data(
            # stitched_video=None,
            # right_video=None,
            # left_audio=None,
            # right_audio=None,
            gps=gps_data[self.seq_length_input :],
            # front_audio=None,
            gaze=gaze_data[gaze_seq_length_input:],
        )

        if self.with_video:
            data_target["left_video"] = frames_gar[self.seq_length_input :]
            data_target["front_video"] = frames_etg[self.seq_length_input :]

        data = Item(train=data_input, target=data_target)
        return data

    def __load_in_memory(self, idx, item):
        """Load the data in memory."""
        try:
            item_size = asizeof.asizeof(item)
        except Exception as e:
            item_size = sys.getsizeof(item)
        if self.memory_cache_size + item_size < self.max_memory_cache_size:
            self.full_dataset[idx] = item
            self.memory_cache_size += item_size
        else:
            logger.info("Memory cache full, not loading more data in memory")
            
        return

    def __postprocess(self, data):
        """Postprocess the data after an uncached item is created."""
        if self.with_video:
            data["train"]["left_video"] = (
                data["train"]["left_video"].astype(np.float16) / 255.0
            )
            data["train"]["front_video"] = (
                data["train"]["front_video"].astype(np.float16) / 255.0
            )
            data["target"]["left_video"] = (
                data["target"]["left_video"].astype(np.float16) / 255.0
            )
            data["target"]["front_video"] = (
                data["target"]["front_video"].astype(np.float16) / 255.0
            )
            if self.crop_videos:
                self._crop_videos(data)

        return data

    def __len__(self):
        """Return the number of data entries."""
        length = len(self.data)
        if self.max_length is not None:
            length = min(length, self.max_length)
        if self.enable_pci_split:
            length = min(length, self.bin_epoch_size)

        return len(self.data)

    def __getitem__(self, idx):
        """Get a data entry by its index."""
        if self.use_memory_cache and self.full_dataset is not None and idx in self.full_dataset:
            return self.full_dataset[idx]

        if self.enable_pci_split:
            bin_id = idx % len(self.data_bins.keys())
            entry_id = (idx // len(self.data_bins.keys())) % len(
                self.data_bins[self.data_bins_keys[bin_id]]
            )
            entry = self.data_bins[self.data_bins_keys[bin_id]][entry_id]
        else:
            entry = self.data[idx]

        pci = entry["pci"]
        session_id = entry["session_id"]
        start_index = entry["start_index"]

        seq_length = entry["seq_length"]
        fps_divisor = entry["fps_divisor"]

        if self.use_data_cache:
            data = self._fetch_from_cache(entry)
            if data is not None:
                # Delete is_sample_ok from data, as it is not needed anymore
                data["pci"] = pci
                data = self.__postprocess(data)
                if self._return_info:
                    return data, entry
                return data

        data = self.__get_uncached_item(
            session_id, start_index, seq_length, fps_divisor
        )
        data["pci"] = pci

        if self.use_data_cache and self._cache_size < self.max_cache_size:
            logger.info(f"Caching {idx}")
            self._push_to_cache(entry, data)

        data = self.__postprocess(data)

        if self.use_memory_cache:
            self.__load_in_memory(idx, data)

        if self._return_info:
            return data, entry
        return data

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

    def _crop_videos(self, data):
        """Crop the left video 15% from top and 35% from bottom."""
        if self.with_video:
            for key in ["train", "target"]:
                height = data[key]["left_video"].shape[2]
                data[key]["left_video"] = data[key]["left_video"][
                    :, :, int(0.15 * height) : int(0.65 * height), :
                ]
        return data
