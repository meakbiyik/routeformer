"""Filters for time-series data."""
import torch


def median_downsampler(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Downsamples a tensor by applying a median filter.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to downsample. Shape: (batch_size, time_steps, channels).
    target_length : int
        Target length of the downsampled tensor.

    Returns
    -------
    torch.Tensor
        Downsampled tensor. Shape: (batch_size, target_length, channels).
    """
    if target_length >= tensor.shape[1]:
        raise ValueError("Target length must be less than the current time steps.")

    # Compute the stride (step size) by dividing
    # the current time steps by the target_length
    batch_size, time_steps, channels = tensor.shape
    stride = time_steps // target_length

    # Create an empty tensor to hold the downsampled values
    downsampled_tensor = torch.empty(
        (batch_size, target_length, channels), dtype=tensor.dtype, device=tensor.device
    )

    # Iterate over each window and compute the median for each channel in the window
    for i in range(target_length):
        start_idx = i * stride
        end_idx = start_idx + stride
        window = tensor[:, start_idx:end_idx, :]

        # Compute the median along the time dimension
        median = window.median(dim=1).values
        downsampled_tensor[:, i, :] = median

    return downsampled_tensor
