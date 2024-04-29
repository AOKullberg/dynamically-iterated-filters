from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from ..types import LinearizedModel, MVNormal
from . import cubature as ct
from . import unscented as ut
from .common import mean_sigma_points, covariance_sigma_points, SigmaPoints

def first_taylor(function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                state_distribution: MVNormal,
                parameter_distribution: MVNormal,
                **kwargs) \
                -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """First-order Taylor expansion of the passed function.

    Linearizes both w.r.t. a state distribution and a parameter distribution.
    If the parameters are point estimates, pass a MVNormal with None
    as cov.

    Parameters
    ----------
    function : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    state_distribution : MVNormal
        State distribution to linearize w.r.t. to.
    parameter_distribution : MVNormal
        Parameter distribution to linearize w.r.t. to.

    Returns
    -------
    LinearizedModel

    """
    state_jac, param_jac = jax.jacfwd(function, argnums=(0, 1))(state_distribution.mean,
                                                            parameter_distribution.mean)
    offset = function(state_distribution.mean, parameter_distribution.mean) - \
                    state_jac@state_distribution.mean - \
                    param_jac@parameter_distribution.mean
    linearization_error = jnp.zeros((offset.size, offset.size))
    # To work with 1D functions as well
    state_jac = jnp.reshape(state_jac, (offset.size, state_distribution.mean.size))
    param_jac = jnp.reshape(param_jac, (offset.size, parameter_distribution.mean.size))
    return LinearizedModel(state_jac, param_jac, offset, linearization_error)

def slr(function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            state_distribution: MVNormal,
            parameter_distribution: MVNormal,
            sigma_points: SigmaPoints) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    vectorized_fun = jax.vmap(function, (0, 0), 0)
    fun_val = vectorized_fun(sigma_points.points[:, :state_distribution.mean.size],
                        sigma_points.points[:, state_distribution.mean.size:])
    transformed_sigma_points = SigmaPoints(fun_val, sigma_points.wm, sigma_points.wc)
    transformed_mean = mean_sigma_points(transformed_sigma_points)
    cross_covariance = covariance_sigma_points(sigma_points,
                jnp.concatenate([state_distribution.mean,
                                parameter_distribution.mean]),
                                transformed_sigma_points, transformed_mean)
    transformed_cov = covariance_sigma_points(transformed_sigma_points,
                                            transformed_mean,
                                            transformed_sigma_points,
                                            transformed_mean)
    jac = jnp.linalg.solve(jax.scipy.linalg.block_diag(state_distribution.cov,
                                                    parameter_distribution.cov),
                            cross_covariance).T
    state_jac = jac[:, :state_distribution.mean.size]
    param_jac = jac[:, state_distribution.mean.size:]
    offset = transformed_mean - state_jac@state_distribution.mean - \
            param_jac@parameter_distribution.mean
    linearization_error = transformed_cov - \
            state_jac@state_distribution.cov@state_jac.T - \
            param_jac@parameter_distribution.cov@param_jac.T
    return LinearizedModel(state_jac, param_jac, offset, linearization_error)

def ut_slr(function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            state_distribution: MVNormal,
            parameter_distribution: MVNormal,
            params: dict = {}) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    param_distribution = guard_parameter_distribution(parameter_distribution)
    linearization_point = MVNormal(\
                        jnp.concatenate([state_distribution.mean,
                                        param_distribution.mean]),
                    cov=jitter_cov(jax.scipy.linalg.block_diag(state_distribution.cov,
                                                    param_distribution.cov)))
    sigma_points = ut.get_sigma_points(linearization_point, params)
    state_distribution = MVNormal(\
                        linearization_point.mean[:state_distribution.mean.size],
                        cov=linearization_point.cov[:state_distribution.mean.size,
                                                :state_distribution.mean.size])
    param_distribution = MVNormal(\
                        linearization_point.mean[state_distribution.mean.size:],
                        cov=linearization_point.cov[state_distribution.mean.size:,
                                                state_distribution.mean.size:])
    return slr(function, state_distribution, param_distribution, sigma_points)

def ct_slr(function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            state_distribution: MVNormal,
            parameter_distribution: MVNormal,
            **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    param_distribution = guard_parameter_distribution(parameter_distribution)
    linearization_point = MVNormal(\
                        jnp.concatenate([state_distribution.mean,
                                        param_distribution.mean]),
                    cov=jitter_cov(jax.scipy.linalg.block_diag(state_distribution.cov,
                                                    param_distribution.cov)))
    sigma_points = ct.get_sigma_points(linearization_point)
    state_distribution = MVNormal(\
                        linearization_point.mean[:state_distribution.mean.size],
                        cov=linearization_point.cov[:state_distribution.mean.size,
                                                :state_distribution.mean.size])
    param_distribution = MVNormal(\
                        linearization_point.mean[state_distribution.mean.size:],
                        cov=linearization_point.cov[state_distribution.mean.size:,
                                                state_distribution.mean.size:])
    return slr(function, state_distribution, param_distribution, sigma_points)

def guard_parameter_distribution(parameter_distribution):
    cov = parameter_distribution.cov
    if parameter_distribution.cov is None:
        cov = jnp.identity(parameter_distribution.mean.size)*1e-16
    return MVNormal(parameter_distribution.mean, cov=cov)

def jitter_cov(matrix: jnp.ndarray) -> jnp.ndarray:
    x = jax.lax.cond(jnp.linalg.det(matrix) < 1e-12,
                lambda x: x + jnp.identity(matrix.shape[0])*1e-16,
                lambda x: x,
                matrix)
    return x
