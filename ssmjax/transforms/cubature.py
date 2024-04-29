"""Adapted from https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers

Original author: [Adrien Corenflos](https://adriencorenflos.github.io/)
"""
from typing import Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..types import MVNormal
from .common import SigmaPoints
from . import common

def cubature_weights(n_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem

    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return wm, wc, xi

def get_sigma_points(mv_normal_parameters: MVNormal) -> SigmaPoints:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mv_normal_parameters: MVNormal
        Mean and Covariance of the distribution

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform

    """
    mean = mv_normal_parameters.mean
    n_dim = mean.shape[0]

    wm, wc, xi = cubature_weights(n_dim)

    sigma_points = jnp.repeat(mean.reshape(1, -1), wm.shape[0], axis=0) \
                   + jnp.dot(jnp.linalg.cholesky(mv_normal_parameters.cov), xi.T).T

    return SigmaPoints(sigma_points, wm, wc)

def integrate(function: Callable[jnp.ndarray, jnp.ndarray],
            x: MVNormal,
            theta: jnp.ndarray,
            params: dict = None) -> jnp.ndarray:
    sigma_points = get_sigma_points(x)
    return common.integrate(function, sigma_points, theta)

def estimate_gradient(observation: jnp.ndarray,
                    transition_covariance: jnp.ndarray,
                    observation_covariance: jnp.ndarray,
                    distribution: MVNormal,
                    theta: jnp.ndarray,
                    loss: Callable) -> jnp.ndarray:
    grad = jax.jacfwd(partial(loss,
                            observation,
                            transition_covariance,
                            observation_covariance),
                            argnums=-1)
    return integrate(grad, distribution, theta)
