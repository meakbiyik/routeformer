"""Baseline models for time-series forecasting."""
import torch
import torch.nn as nn

from .config import GPSBackboneConfig


class StationaryBaseline(nn.Module):
    """Time-series forecasting baseline model that predicts 0 for all time steps.

    Assuming the input is the dynamics (velocity) of the time-series.
    """

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        """Forward pass.

        Predict 0 for all time steps.

        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape: [Batch, Input length, Channel]

        Returns
        -------
        torch.Tensor
            Prediction. Shape: [Batch, Prediction length, Channel]
        """
        return torch.zeros(x.shape[0], self.pred_len, 2, device=x.device)


class LinearBaseline(nn.Module):
    """Time-series forecasting baseline model.

    Predicts a linear trajectory for all time steps. Assuming that the input is
    the dynamics (velocity) of the time-series.
    """

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        """Forward pass.

        Take the average of last 5 time steps and output that for all time steps.

        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape: [Batch, Input length, Channel]

        Returns
        -------
        torch.Tensor
            Prediction. Shape: [Batch, Prediction length, Channel]
        """
        average = torch.mean(x[:, -5:, :2], dim=1, keepdim=True)
        return average.repeat(1, self.pred_len, 1)
