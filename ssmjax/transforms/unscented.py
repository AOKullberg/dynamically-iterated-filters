from typing import Tuple, Callable

import jax.numpy as jnp
import numpy as np

from .common import SigmaPoints
from ..types import MVNormal
from . import common

def unscented_weights(n_dim: int,
                    params: dict = {}) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Weights for sigma points of the unscented transform.
    Defaults to UT1 parameters, i.e.,

    $\alpha=\sqrt{3/n_x},~\beta=3/n_x-1,~\kappa=0$

    Parameters
    ----------
    n_dim : int
        dimensionality of the problem
    params : dict
        unscented parameters alpha, beta, kappa -- see above

    Returns
    -------
    wm: np.ndarray
        Weights for the means
    wc: np.ndarray
        Weights for the covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    alpha = params.get('alpha', jnp.sqrt(3/n_dim))
    beta = params.get('beta', 3/n_dim - 1)
    kappa = params.get('kappa', 0)
    _lambda = alpha**2*(n_dim+kappa)-n_dim
    w0 = jnp.array([_lambda/(n_dim+_lambda)])
    wi = 1/(2*(n_dim+_lambda))*jnp.ones(2*n_dim)
    wm = jnp.concatenate([w0, wi])
    wc = wm
    wc = wc.at[0].add(1-alpha**2 + beta)
    xi = jnp.concatenate([jnp.eye(n_dim), -jnp.eye(n_dim)], axis=0) * jnp.sqrt((n_dim+_lambda))
    # np.sqrt((n_dim/(1-w0))
    return wm, wc, xi

def get_sigma_points(distribution: MVNormal,
                    params: dict = {}) -> SigmaPoints:
    """ Computes the unscented sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim + 1

    Parameters
    ----------
    mv_normal_parameters : MVNormal
        Mean and Covariance of the distribution
    params : dict
        Parameters for the unscented transform, optional. Defaults to UT1, i.e.,
        $\alpha=\sqrt{3/n_dim},~\beta=3/n_dim-1,~\kappa=0$

    Returns
    -------
    out: SigmaPoints
        sigma points for the unscented transform

    """
    mean = distribution.mean
    n_dim = distribution.mean.shape[0]
    wm, wc, xi = unscented_weights(n_dim, params)

    sigma_points = jnp.repeat(mean.reshape(1, -1), wm.shape[0]-1, axis=0) \
                   + jnp.dot(jnp.linalg.cholesky(distribution.cov), xi.T).T

    X = jnp.vstack([mean[None, :], sigma_points])
    return SigmaPoints(X, wm, wc)

def integrate(function: Callable[jnp.ndarray, jnp.ndarray],
            x: MVNormal,
            theta: jnp.ndarray,
            params: dict = None) -> jnp.ndarray:
    sigma_points = get_sigma_points(x, params)
    return common.integrate(function, sigma_points, theta)
