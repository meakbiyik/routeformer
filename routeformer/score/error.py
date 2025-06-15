"""Average Displacement Error (ADE) and Final Displacement Error (FDE) metrics.

These metrics are commonly used for evaluating the performance of trajectory
prediction models.
"""

import torch


def ade(predicted_trajectory, ground_truth_trajectory):
    """Calculate the Average Displacement Error (ADE).

    Parameters
    ----------
    predicted_trajectory : Tensor
        The trajectory predicted by the model.
    ground_truth_trajectory : Tensor
        The ground truth trajectory.

    Returns
    -------
    float
        The ADE of the predicted trajectory.
    """
    assert (
        predicted_trajectory.shape == ground_truth_trajectory.shape
    ), "Predicted and ground truth trajectories must be of the same shape"

    return torch.norm(predicted_trajectory - ground_truth_trajectory, dim=-1).mean()


def fde(predicted_trajectory, ground_truth_trajectory):
    """Calculate the Final Displacement Error (FDE).

    Parameters
    ----------
    predicted_trajectory : Tensor
        The trajectory predicted by the model.
    ground_truth_trajectory : Tensor
        The ground truth trajectory.

    Returns
    -------
    float
        The FDE of the predicted trajectory.
    """
    assert (
        predicted_trajectory.shape == ground_truth_trajectory.shape
    ), "Predicted and ground truth trajectories must be of the same shape"

    return torch.norm(predicted_trajectory[-1] - ground_truth_trajectory[-1])
