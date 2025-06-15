"""Utility functions for working with vectors."""
import torch
from torch.amp import autocast


def rotate(tensor: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate a vector tensor by a given angle.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to rotate. Shape (batch_size, input_length, 2).
    angle : torch.Tensor
        Angles in radians to rotate the tensor by. Shape (batch_size, 1).

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same dtype as the input tensor.
    """
    with autocast(tensor.device.type, enabled=False):
        # Store original dtype
        original_dtype = tensor.dtype

        tensor = tensor.float()
        angle = angle.float()

        batch_size = tensor.shape[0]

        # Construct the rotation matrices for each batch
        cos_angles = torch.cos(angle).squeeze(-1)
        sin_angles = torch.sin(angle).squeeze(-1)

        rotation_matrices = torch.stack(
            [
                cos_angles,
                -sin_angles,
                sin_angles,
                cos_angles,
            ],
            dim=1,
        ).reshape(batch_size, 2, 2)

        # reshape the tensor to (batch_size, 2, input_length)
        tensor = tensor.permute(0, 2, 1)

        # rotate the tensor
        tensor = torch.matmul(rotation_matrices, tensor)

        # reshape the tensor back to (batch_size, input_length, 2)
        tensor = tensor.permute(0, 2, 1)

    # Convert back to original dtype before returning
    return tensor.to(original_dtype)


def estimate_angle(tensor: torch.Tensor) -> torch.Tensor:
    """Estimate the angle of a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to estimate the angle of. Shape (*, 2).

    Returns
    -------
    torch.Tensor
        Estimated angle in radians, shape (*, 1), in float32.
    """
    tensor = tensor.float()

    # reshape the tensor to (*, 2)
    original_shape = tensor.shape[:-1]
    tensor = tensor.reshape(-1, 2)
    
    # get the angle of the tensor
    angle = torch.atan2(tensor[:, 1], tensor[:, 0])
    
    # reshape the tensor back to (*, 1)
    angle = angle.reshape(*original_shape, 1)

    return angle

def estimate_angle_and_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Estimate the angle and norm of a vector tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to estimate the angle and norm of. Shape (*, 2).

    Returns
    -------
    torch.Tensor
        Estimated angle in radians, shape (*, 1).
    torch.Tensor
        Estimated norm, shape (*, 1).
    """
    tensor = tensor.float()

    # reshape the tensor to (*, 2)
    original_shape = tensor.shape[:-1]
    tensor = tensor.reshape(-1, 2)
    
    # get the angle and norm of the tensor
    angle = torch.atan2(tensor[:, 1], tensor[:, 0])
    norm = torch.norm(tensor, dim=1)

    # reshape the tensors back to (*, 1)
    return angle.reshape(*original_shape, 1), norm.reshape(*original_shape, 1)
