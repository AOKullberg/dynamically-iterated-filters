import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from ..types import MVNormal

__all__ = ["build_propagate", "build_update", "build_smooth"]

def build_propagate(linearization_method, transition_function):
    def propagate(prior, transition_covariance, linearization_point, theta, **kwargs):
        if linearization_point is None:
            linearization_point = prior
        # "Copies" the parameter in order to modify the contents without
        # affecting the reference.
        param = jax.tree_map(lambda x: x, theta)
        if not isinstance(param, MVNormal):
            param = MVNormal(param,
                    jnp.zeros((int((param.size**2+param.size)/2),)))
        if param._cov is None and param._chol is None:
            param._cov = jnp.zeros((param.mean.size, param.mean.size))
        # Linearizes w.r.t. a state distribution and possibly a parameter
        # distribution (or point estimate if that is the case)
        linearized_model = linearization_method.method(function=transition_function,
                                        state_distribution=linearization_point,
                                        parameter_distribution=param,
                                        params=linearization_method.parameters)
        # Propagate state uncertainty and process noise
        cov = jnp.dot(linearized_model.state_jac, jnp.dot(prior.cov, linearized_model.state_jac.T)) + transition_covariance
        # Include parameter uncertainty (if present)
        cov = cov + jnp.dot(linearized_model.param_jac, jnp.dot(param.cov, linearized_model.param_jac.T))
        # Include linearization error
        cov = cov + linearized_model.linearization_error
        # The parameter transition is captured in the offset term
        # (linearized w.r.t. the mean of the parameter distribution)
        mean = jnp.dot(linearized_model.state_jac, prior.mean) + \
                linearized_model.offset + \
                jnp.dot(linearized_model.param_jac, param.mean)
        return MVNormal(mean, cov=0.5 * (cov + cov.T)), linearized_model
    return propagate
def build_update(linearization_method, observation_function):
    def update(predicted, observation, observation_covariance, linearization_point, theta, **kwargs):
        if linearization_point is None:
            linearization_point = predicted
        param = jax.tree_map(lambda x: x, theta)
        if not isinstance(param, MVNormal):
            param = MVNormal(param, jnp.zeros((int((param.size**2+param.size)/2),)))
        if param._cov is None and param._chol is None:
            param._cov = jnp.zeros((param.mean.size, param.mean.size))
        linearized_model = linearization_method.method(function=observation_function,
                                        state_distribution=linearization_point,
                                        parameter_distribution=param,
                                        params=linearization_method.parameters)

        obs_mean = jnp.dot(linearized_model.state_jac, predicted.mean) + \
                    linearized_model.offset + \
                    jnp.dot(linearized_model.param_jac, param.mean)

        residual = observation - obs_mean
        residual_covariance = jnp.dot(linearized_model.state_jac,
                            jnp.dot(predicted.cov, linearized_model.state_jac.T))
        residual_covariance = residual_covariance + \
                            observation_covariance + \
                            linearized_model.linearization_error + \
                            jnp.dot(linearized_model.param_jac,
                            jnp.dot(param.cov, linearized_model.param_jac.T))

        gain = jnp.dot(predicted.cov, jnp.linalg.solve(residual_covariance,
                                                linearized_model.state_jac).T)

        mean = predicted.mean + jnp.dot(gain, residual)
        cov = predicted.cov - jnp.dot(gain, jnp.dot(residual_covariance, gain.T))
        updated_state = MVNormal(mean, cov = 0.5 * (cov + cov.T))

        loglikelihood = multivariate_normal.logpdf(residual,
                                jnp.zeros_like(residual), residual_covariance)
        return loglikelihood, (updated_state, linearized_model)
    return update
def build_smooth(linearization_method, transition_function):
    def smooth(filtered_state, smoothed_state, transition_covariance, linearization_point, theta, **kwargs):
        if linearization_point is None:
            linearization_point = filtered_state
        param = jax.tree_map(lambda x: x, theta)
        if not isinstance(param, MVNormal):
            param = MVNormal(param,
                    jnp.zeros((int((param.size**2+param.size)/2),)))
        if param._cov is None and param._chol is None:
            param._cov = jnp.zeros((param.mean.size, param.mean.size))
        linearized_model = linearization_method.method(function=transition_function,
                                            state_distribution=linearization_point,
                                            parameter_distribution=param,
                                            params=linearization_method.parameters)

        Cx = jnp.dot(filtered_state.cov, linearized_model.state_jac.T)
        Cxx = jnp.dot(linearized_model.state_jac, Cx) + \
            transition_covariance + \
            linearized_model.linearization_error + \
            jnp.dot(linearized_model.param_jac, jnp.dot(param.cov,
                                                linearized_model.param_jac.T))

        gain = jnp.linalg.solve(Cxx, Cx.T).T
        mean = jnp.dot(linearized_model.state_jac, filtered_state.mean) + \
                linearized_model.offset + \
                jnp.dot(linearized_model.param_jac, param.mean)
        mean = filtered_state.mean + jnp.dot(gain, smoothed_state.mean - mean)
        cov = filtered_state.cov + jnp.dot(gain,
                                    jnp.dot(smoothed_state.cov - Cxx, gain.T))
        return MVNormal(mean, cov=0.5*(cov + cov.T)), \
                jnp.dot(gain, smoothed_state.cov)
    return smooth
