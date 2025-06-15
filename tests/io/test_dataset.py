"""Tests for the GEMDataset class."""
import logging

import pytest
import torch

from routeformer.io import GEMDataset
from routeformer.utils import set_logger_config

set_logger_config(logging.WARNING)


def video_downscaler(x):
    """Downscale the video by a factor of 8."""
    return x[:, :, ::8, ::8]


def test_routeformer_dataset(shared_datadir) -> None:
    """Test the GEMDataset class initialization and functions."""
    # Misaligned dataset (3 GoPro subjects, 4 EyeTracker subjects)
    dataset = GEMDataset(
        shared_datadir / "dataset1",
        split=["001", "002", "003"],
        with_audio=True,
        input_length=0.3,
        target_length=0.2,
        step_size=0.5,
        crop_videos=False,
        video_transform=video_downscaler,
        frame_transform=lambda x: x,
        undistort_videos=False,  # this is very costly on CPU
        output_fps=30,
        min_pci=None,
        use_corrected_gps=False,
    )
    assert len(dataset.subjects) == 2
    assert len(dataset.left_samples["001"]) == 1
    assert len(dataset.right_samples["001"]) == 1
    assert len(dataset.left_samples["002"]) == 1
    assert len(dataset.right_samples["002"]) == 1
    expected_train_video_frame_counts = [9, 9, 9]
    expected_target_video_frame_counts = [6, 6, 6]
    iter_count = 0
    for train, target, data in zip(
        expected_train_video_frame_counts, expected_target_video_frame_counts, dataset
    ):
        assert data["train"]["left_video"].shape == (train, 3, 2160 // 8, 3840 // 8)
        assert data["train"]["right_video"].shape == (train, 3, 2160 // 8, 3840 // 8)
        assert data["target"]["left_video"].shape == (target, 3, 2160 // 8, 3840 // 8)
        assert data["target"]["right_video"].shape == (target, 3, 2160 // 8, 3840 // 8)
        iter_count += 1
    assert iter_count == len(expected_train_video_frame_counts)
    # Aligned dataset (1 GoPro subject, 1 EyeTracker subject)
    dataset = GEMDataset(
        shared_datadir / "dataset1",
        split=["001", "002", "003"],
        with_audio=True,
        with_video=False,
        output_fps=30,
        undistort_videos=False,
        min_pci=None,
        use_corrected_gps=False,
    )
    for data in dataset:
        break
    # Give empty directory
    with pytest.raises(ValueError):
        dataset = GEMDataset(shared_datadir / "empty_dataset")
    # Test wrong output format
    with pytest.raises(ValueError):
        dataset = GEMDataset(shared_datadir / "dataset1", output_format="wrong")
    # Test scaling
    dataset = GEMDataset(
        shared_datadir / "dataset1",
        split=["001", "002", "003"],
        input_length=0.2,
        target_length=0.2,
        step_size=0.5,
        gopro_scaling_factor=1 / 8,
        crop_videos=True,
        output_fps=30,
        undistort_videos=True,
        min_pci=None,
        use_corrected_gps=False,
    )
    expected_train_video_frame_counts = [6, 6, 6]
    expected_target_video_frame_counts = [6, 6, 6]
    iter_count = 0
    for train, target, data in zip(
        expected_train_video_frame_counts, expected_target_video_frame_counts, dataset
    ):
        assert data["train"]["left_video"].shape == (
            train,
            3,
            2160 // 8 * 0.4,  # Cropping removes 60% of the image
            3840 // 8,
        )
        assert data["train"]["right_video"].shape == (
            train,
            3,
            2160 // 8 * 0.4,  # Cropping removes 60% of the image
            3840 // 8,
        )
        assert data["target"]["left_video"].shape == (
            target,
            3,
            2160 // 8 * 0.4,  # Cropping removes 60% of the image
            3840 // 8,
        )
        assert data["target"]["right_video"].shape == (
            target,
            3,
            2160 // 8 * 0.4,  # Cropping removes 60% of the image
            3840 // 8,
        )
        iter_count += 1
    assert iter_count == len(expected_train_video_frame_counts)

    # Mock windows (by monkey patching platform.system)
    import platform

    previous_platform_system = platform.system
    platform.system = lambda: "Windows"
    with pytest.raises(FileNotFoundError):
        # should raise FileNotFoundError because ffmpeg.exe is not in PATH
        dataset = GEMDataset(
            shared_datadir / "dataset1",
            split=["001", "002", "003"],
            input_length=0.2,
            target_length=0.2,
            step_size=0.5,
            undistort_videos=False,
            min_pci=None,
            use_corrected_gps=False,
        )
    platform.system = previous_platform_system

    with pytest.raises(ValueError):
        GEMDataset(
            shared_datadir / "dataset1",
            output_fps=29,
            split=["001", "002", "003"],
            undistort_videos=False,
            min_pci=None,
            use_corrected_gps=False,
        )
