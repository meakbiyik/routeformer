"""Implements FutureDiscountedLoss that discounts the future based on the epoch."""
from typing import Dict, Union

import lightning as pl

import torch
import torch.nn as nn


class FutureDiscountedLoss(pl.LightningModule):
    """Calculates MSE loss on the last prediction, with future predictions discounted.

    The loss is calculated as:
        1 * (y_pred[-1] - y_true[-1])^2
        + discount_factor * (y_pred[-2] - y_true[-2])^2
        + discount_factor^2 * (y_pred[-3] - y_true[-3])^2
        + ...
    where discount_factor is a number between 0 and 1 that depends on the epoch.
    """

    def __init__(
        self,
        discount_factor: Union[float, Dict[int, float]] = 0.9,
        epsilon: float = None,
        loss_function: str = "mse",
    ):
        """Initialize the loss.

        Parameters
        ----------
        discount_factors : Union[float, Dict[int, float]], optional
            The discount factor, by default 0.9.
            If a float, the discount factor is constant.
            If a dict, the discount factor is a function of the epoch.
            The dict changes the discount factor every time the epoch is in the keys.
            It has to have a key for epoch 0.
        epsilon : float, optional
            The epsilon to use in epsilon-insensitive loss, by default None.
        loss_function : str, optional
            The loss function to use, by default "mse".
            Can be "mse", "mae", or "smooth_l1".
        """
        super().__init__()
        self.current_discount_factor = (
            discount_factor
            if isinstance(discount_factor, float)
            else discount_factor[0]
        )
        self.discount_factor_dict = (
            discount_factor if isinstance(discount_factor, dict) else {}
        )
        self.epsilon = epsilon
        self.loss_function = loss_function

        if self.loss_function not in ["mae", "mse", "smooth_l1"]:
            raise ValueError(f"Unknown loss function {self.loss_function}")
        
        if self.loss_function == "smooth_l1":
            self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, y_pred, y_true):
        """Calculate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values, in the shape (B, T, *)
        y_true : torch.Tensor
            The true values, in the shape (B, T, *)

        Returns
        -------
        torch.Tensor
            The loss.
        """
        epoch = self.current_epoch
        # Update the discount factor if needed
        if epoch in self.discount_factor_dict:
            self.current_discount_factor = self.discount_factor_dict[epoch]
        # Calculate the loss
        add_dimension_count = len(y_pred.shape) - 2
        assert add_dimension_count >= 0
        factors = torch.pow(
            self.current_discount_factor,
            torch.arange(y_pred.shape[1], device=y_pred.device),
        )
        factors = factors.view(1, -1, *([1] * add_dimension_count))

        # ignore errors smaller than epsilon
        error = y_pred - y_true
        error = torch.where(
            torch.abs(error) < self.epsilon, torch.zeros_like(error), error
        )

        if self.loss_function == "mae":
            return torch.abs(error).mul_(factors).mean()
        elif self.loss_function == "mse":
            return error.pow(2).mul_(factors).mean()
        elif self.loss_function == "smooth_l1":
            return self.loss(y_pred, y_true).mul(factors).mean()
        else:
            raise ValueError(f"Unknown loss function {self.loss_function}")
