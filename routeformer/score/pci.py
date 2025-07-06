"""Estimate the pci of some crafted trajectories.

This is a subtask of trajectory extrapolation, which seems to be a
not-so-well-studied problem. The goal is to extrapolate a trajectory
given a set of points, and a direction of travel.

Candidate ways:
1. Linear or quadratic extrapolation of the last N points
2. Quadratic extrapolation of the last N points, with constraints on
    the first and second derivatives
3. Gaussian process regression

In the first two cases, the pci would be measured as the MSE
between the extrapolated trajectory and the real trajectory. In the
third case, the pci would be measured as the negative log
likelihood of the real trajectory given the GP model.
"""
from typing import Literal

import numpy as np
from frechetdist import frdist
from numpy.polynomial import Polynomial
from scipy.optimize import minimize


def fit_quadratic_with_constraints(t, y, max_speed, max_accel, domain=None):
    """Fit a constrained quadratic function to the given data.

    The curve is subject to constraints on the first and second derivatives.

    Parameters
    ----------
    t : np.ndarray
        The time values of the data points.
    y : np.ndarray
        The y values of the data points.
    max_speed : float
        The maximum speed of the quadratic function.
    max_accel : float
        The maximum acceleration of the quadratic function.
    domain : list, optional
        The domain of the quadratic function, by default None. If None,
        the domain is set to the minimum and maximum values of t.

    Returns
    -------
    np.ndarray
        The parameters of the quadratic function.
    """
    t = np.array(t)
    y = np.array(y)

    if domain is None:
        domain = [t.min(), t.max()]

    # Quadratic function to fit
    def f(t, params):
        a, b, c = params
        return a * t**2 + b * t + c

    # Speed is the first derivative
    def speed(t, params):
        a, b = params[0], params[1]
        return 2 * a * t + b

    # Acceleration is the second derivative
    def accel(params):
        a = params[0]
        return 2 * a

    # Objective function: squared error
    def objective(params):
        return np.sum((y - f(t, params)) ** 2)

    # Constraints: speed and acceleration
    constraints = (
        {
            "type": "ineq",
            "fun": lambda params: max_speed
            - np.max(np.abs(speed(np.linspace(domain[0], domain[1], 10), params))),
        },
        {"type": "ineq", "fun": lambda params: max_accel - np.abs(accel(params))},
    )

    # Initial guess
    params0 = [0, 0, 0]

    # Perform constrained optimization
    result = minimize(objective, params0, constraints=constraints)

    return result.x


def pci(
    real_trajectory: np.ndarray,
    regular_trajectory: np.ndarray,
    measure: Literal["mse", "frechet"] = "frechet",
) -> float:
    """Calculate the pci of a regular trajectory compared to a real trajectory.

    Parameters
    ----------
    real_trajectory : np.ndarray
        The real trajectory, N x 2 array.
    regular_trajectory : np.ndarray
        The regular trajectory, N x 2 array.
    measure : Literal["mse", "frechet"], optional
        The measure of pci to use, by default "frechet".

    Returns
    -------
    float
        The pci of the regular trajectory compared to the real trajectory.
    """
    if measure == "mse":
        return np.mean((real_trajectory - regular_trajectory) ** 2)
    elif measure == "frechet":
        return frdist(real_trajectory, regular_trajectory)
    else:
        raise ValueError("Invalid pci measure.")


def estimate_regular_trajectory(
    input_trajectory: np.ndarray,
    time_steps: int,
    curve_type: Literal["linear", "quadratic", "constrained_quadratic"] = "linear",
    lookback_length: int = 6,
    constraints: dict = None,
    frequency: float = 30,
):
    """Create a regular trajectory by extending the last segment of the input.

    Parameters
    ----------
    input_trajectory : np.ndarray
        The input trajectory, N x 2 array.
    time_steps : int
        The number of time steps to extend the trajectory.
    curve_type : Literal["linear", "quadratic", "constrained_quadratic"], optional
        The type of curve to use for the regular trajectory, by default "linear".
    lookback_length : int, optional
        The last N points of the input trajectory to use for the regular trajectory,
        by default 6. Corresponds to 200ms at 30Hz.
    constraints : dict, optional
        Constraints on the regular trajectory. If curve_type is
        "constrained_quadratic", then constraints must be a dict with keys
        "max_speed" and "max_accel", corresponding to the maximum speed and
        acceleration of the regular trajectory.
    frequency : float, optional
        The frequency of the input trajectory in Hertz, by default 30.

    Returns
    -------
    np.ndarray
        The regular trajectory, time_steps x 2 array.
    """
    if input_trajectory.shape[0] < lookback_length:
        raise ValueError("Lookback length is greater than the number of points in the trajectory.")

    lookback_points = input_trajectory[-lookback_length:]
    time = np.arange(lookback_length + time_steps) / frequency
    input_time = time[:lookback_length]
    target_time = time[lookback_length:]
    x = lookback_points[:, 0]
    y = lookback_points[:, 1]

    if curve_type == "constrained_quadratic":
        if constraints is None:
            raise ValueError("Constraints must be provided if curve_type is constrained_quadratic.")

        fit_x_params = fit_quadratic_with_constraints(
            input_time,
            x,
            constraints["max_speed"],
            constraints["max_accel"],
            domain=[time[0], time[-1]],
        )
        fit_y_params = fit_quadratic_with_constraints(
            input_time,
            y,
            constraints["max_speed"],
            constraints["max_accel"],
            domain=[time[0], time[-1]],
        )

        fit_x = lambda t: fit_x_params[0] * t**2 + fit_x_params[1] * t + fit_x_params[2]
        fit_y = lambda t: fit_y_params[0] * t**2 + fit_y_params[1] * t + fit_y_params[2]

    else:
        degree = {"linear": 1, "quadratic": 2}.get(curve_type)
        if degree is None:
            raise ValueError(
                "Invalid curve_type. Choose from 'linear', " "'quadratic', 'constrained_quadratic'."
            )

        # Fit the polynomial for x and y separately using Polynomial.fit
        fit_x = Polynomial.fit(
            input_time,
            x,
            degree,
            domain=[input_time[0], input_time[-1]],
        )
        fit_y = Polynomial.fit(
            input_time,
            y,
            degree,
            domain=[input_time[0], input_time[-1]],
        )

    # Evaluate the polynomial at the new t values
    new_x = fit_x(target_time)
    new_y = fit_y(target_time)

    # combine to create new trajectory
    new_trajectory = np.stack((new_x, new_y), axis=-1)

    return new_trajectory


def estimate_pci(
    input_trajectory,
    target_trajectory,
    curve_type: Literal["linear", "quadratic", "constrained_quadratic"] = "linear",
    lookback_length: int = 6,
    constraints: dict = None,
    frequency: float = 30,
    measure: Literal["mse", "frechet"] = "frechet",
    return_regular_trajectory=False,
):
    """Estimate the pci of a target trajectory from an input trajectory.

    Parameters
    ----------
    input_trajectory : np.ndarray
        The input trajectory, N x 2 array.
    target_trajectory : np.ndarray
        The target trajectory, N x 2 array.
    curve_type : Literal["linear", "quadratic", "constrained_quadratic"], optional
        The type of curve to use for the regular trajectory, by default "linear".
    lookback_length : int, optional
        The last N points of the input trajectory to use for the regular trajectory,
        by default 6. Corresponds to 200ms at 30Hz.
    constraints : dict, optional
        Constraints on the regular trajectory. If curve_type is
        "constrained_quadratic", then constraints must be a dict with
        keys "max_speed" and "max_accel", corresponding to the maximum
        speed and acceleration of the regular trajectory.
    frequency : float, optional
        The frequency of the trajectory in Hertz, by default 30. Used
        in estimating the constraints. Note that trajectory is assumed to be in meters.
    measure : Literal["mse", "frechet"], optional
        The measure to use for pci, by default "frechet".
    return_regular_trajectory : bool, optional
        Whether to return the regular trajectory, by default False.

    Returns
    -------
    float
        The pci of the target trajectory from the input trajectory.
    np.ndarray
        The regular trajectory, N x 2 array.
    """
    regular_trajectory = estimate_regular_trajectory(
        input_trajectory,
        len(target_trajectory),
        curve_type,
        lookback_length,
        constraints,
        frequency,
    )
    trajectory_pci = pci(target_trajectory, regular_trajectory, measure)
    if return_regular_trajectory:
        return trajectory_pci, regular_trajectory
    else:
        return trajectory_pci
