from typing import Callable
from collections import namedtuple

import jax
import jax.numpy as jnp

from ..types import MVNormal

__all__ = ["SigmaPoints", "mean_sigma_points", "covariance_sigma_points",
            "get_mv_normal_parameters", "integrate"]

SigmaPoints = namedtuple(
    'SigmaPoints', ['points', 'wm', 'wc']
)

def mean_sigma_points(points):
    """
    Computes the mean of sigma points

    Parameters
    ----------
    points: SigmaPoints
        The sigma points

    Returns
    -------
    mean: array_like
        the mean of the sigma points
    """
    return jnp.dot(points.wm, points.points)


def covariance_sigma_points(points_1, mean_1, points_2, mean_2):
    """
    Computes the covariance between two sets of sigma points

    Parameters
    ----------
    points_1: SigmaPoints
        first set of sigma points
    mean_1: array_like
        assumed mean of the first set of points
    points_2: SigmaPoints
        second set of sigma points
    points_1: SigmaPoints
        assumed mean of the second set of points

    Returns
    -------
    cov: array_like
        the covariance of the two sets
    """
    one = (points_1.points.reshape(-1, mean_1.size) - mean_1.reshape(1, -1)).T * points_1.wc.reshape(1, -1)
    two = points_2.points.reshape(-1, mean_2.size) - mean_2.reshape(1, -1)
    return jnp.dot(one, two)

def get_mv_normal_parameters(sigma_points: SigmaPoints) -> MVNormal:
    """ Computes the MV Normal distribution parameters associated with the sigma points

    Parameters
    ----------
    sigma_points: SigmaPoints
        shape of sigma_points.points is (n_dim, 2*n_dim)
    Returns
    -------
    out: MVNormal
        Mean and covariance of RV of dimension K computed from sigma-points
    """
    m = mean_sigma_points(sigma_points)
    cov = covariance_sigma_points(sigma_points, m, sigma_points, m)
    return MVNormal(m, cov=cov)

def integrate(function: Callable[jnp.ndarray, jnp.ndarray],
            sigma_points: SigmaPoints,
            theta: jnp.ndarray) -> jnp.ndarray:
    vfun = jax.vmap(function, (0, None), 0)
    X = MVNormal(sigma_points.points, [])
    Z = vfun(X, theta).squeeze()
    Z_sigma_points = SigmaPoints(Z, sigma_points.wm, sigma_points.wc)
    return mean_sigma_points(Z_sigma_points)
