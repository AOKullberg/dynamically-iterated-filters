import jax
import jax.numpy as jnp

from .filters import base_filter, build_forward_step, build_filter_body
from .ifilters import base_iterated_filter
from . import common

from ..types import LinearizationMethod, LineSearchIterationOptions, MVNormal
from ..transforms.linearization import first_taylor, ct_slr, ut_slr

__all__ = ["base_linesearch_filter",
            "lsiekf", "lsiukf", "lsickf",
            "lsiplf", "utlsiplf", "ctlsiplf"]

def build_loss(model):
    """
    Builds the loss function. It is built up of two components:
    - An observation loss relating the iterate at time k to the observation at
    time k
    - A prior loss relating the iterate at time k to the prior at time k
    """
    def loss(prior, observation, observation_covariance, theta, iterate):
        ye = observation - model.observation_function(iterate.mean, theta.mean)
        xp = prior.mean - iterate.mean
        # Observation loss
        Vy = ye.T@jnp.linalg.solve(observation_covariance, ye)
        # Prior loss
        Vp = xp.T@jnp.linalg.solve(prior.cov, xp)
        return Vy + Vp
    return loss

def build_linesearch_iplf(update, linearization_method, model, loss,
                        inner_iterations, outer_iterations):
    def ls_iplf_update(predicted, observation, observation_covariance,
                        linearization_point, theta, options):
        linearization_point = predicted if linearization_point is None else linearization_point
        linesearch = build_linesearch(loss, predicted, observation,
                                    observation_covariance, theta, options)

        upd = jax.tree_util.Partial(update,
                observation_covariance=observation_covariance,
                predicted=predicted, observation=observation,
                theta=theta)
        param = jax.tree_map(lambda x: x, theta)
        if not isinstance(param, MVNormal):
            param = MVNormal(param, jnp.zeros((int((param.size**2+param.size)/2),)))
        if param._cov is None and param._chol is None:
            param._cov = jnp.zeros((param.mean.size, param.mean.size))

        def inner_loop(carry, _):
            old_iterate, current_iterate, linearization_error = carry
            alpha = linesearch(old_iterate, current_iterate)
            # More or less ordinary measurement update
            linearized_model = linearization_method.method(function=model.observation_function,
                                            state_distribution=current_iterate,
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
                                linearization_error + \
                                jnp.dot(linearized_model.param_jac,
                                jnp.dot(param.cov, linearized_model.param_jac.T))

            gain = jnp.dot(predicted.cov, jnp.linalg.solve(residual_covariance,
                                                    linearized_model.state_jac).T)

            mean = predicted.mean + jnp.dot(gain, residual)
            new_iterate = MVNormal((1-alpha)*mean + alpha*current_iterate.mean,
                            cov=current_iterate.cov)
            return (current_iterate, new_iterate, linearization_error), None

        # First update
        ell, (updated_state, linearized_observation) = \
            upd(linearization_point=linearization_point)
        first_iterate = MVNormal(updated_state.mean, cov=predicted.cov)

        def outer_loop(carry, _):
            ell, old_iterate, current_iterate, linearized_observation = carry
            # Inner loop updates mean given a linearization error and a
            # particular state error covariance
            (_, new_iterate, _), _ = jax.lax.scan(inner_loop,
                        (old_iterate, current_iterate,
                        linearized_observation.linearization_error),
                        jnp.arange(inner_iterations))
            old_iterate = current_iterate
            # Update the covariance and the linearization error, neglect the
            # mean update here.
            ell, (new_updated, linearized_observation) = \
                    upd(linearization_point=new_iterate)
            new_iterate = MVNormal(new_iterate.mean, cov=new_updated.cov)
            return (ell, old_iterate, new_iterate, linearized_observation), None

        (ell, _, updated_state, linearized_observation), _ = \
                    jax.lax.stop_gradient(jax.lax.scan(outer_loop,
                        (ell, predicted, first_iterate, linearized_observation),
                                jnp.arange(outer_iterations)))
        return ell, (updated_state, linearized_observation)
    return ls_iplf_update

def build_linesearch(loss, predicted, observation, observation_covariance, theta, options):
    vloss = jax.value_and_grad(loss, argnums=-1)
    def search(old_iterate, current_iterate):
        oldV, dV = vloss(predicted, observation, observation_covariance,
                        theta, old_iterate)
        alpha = 1.
        # Search direction
        pi = current_iterate.mean - old_iterate.mean
        Vi = loss(predicted, observation, observation_covariance, theta,
            MVNormal(old_iterate.mean + alpha*pi, cov=old_iterate.cov))
        def cond(carry):
            Vi, alpha = carry
            return Vi > oldV + alpha*options.linesearch.gamma*dV.mean.T@pi
        def body(carry):
            Vi, alpha = carry
            alpha = alpha * options.linesearch.beta
            Vi = loss(predicted, observation, observation_covariance, theta,
                MVNormal(old_iterate.mean + alpha*pi, cov=old_iterate.cov))
            return (Vi, alpha)
        Vi, alpha = jax.lax.while_loop(cond, body, (Vi, alpha))
        return alpha
    return search

def build_linesearch_update(update, loss, iterations, freeze_covariance):
    """ Linesearch based damped iterated updates.
    Corresponds to the damped IEKF, damped IUKF and an (alternative) version of
    the inner loop of the damped IPLF.

    """
    def linesearch_update(predicted, observation, observation_covariance,
                        linearization_point, theta, options):
        linearization_point = predicted if linearization_point is None else linearization_point

        linesearch = build_linesearch(loss, predicted, observation,
                    observation_covariance, theta, options)
        upd = jax.tree_util.Partial(update,
                observation_covariance=observation_covariance,
                predicted=predicted, observation=observation,
                theta=theta)
        # First update
        ell, (updated_state, linearized_observation) = \
            upd(linearization_point=linearization_point)
        first_iterate = MVNormal(updated_state.mean, cov=linearization_point.cov)

        def body(carry, _):
            ell, old_updated, curr_updated, _ = carry
            alpha = linesearch(old_updated, curr_updated)
            curr_updated = MVNormal((1-alpha)*old_updated.mean + \
                                    alpha*curr_updated.mean,
                                    cov=curr_updated.cov)
            old_updated = curr_updated
            # This corresponds to (damped) IUKF style updates
            if freeze_covariance:
                curr_updated._cov = linearization_point._cov
            ell, (new_updated, linearized_observation) = \
                    upd(linearization_point=curr_updated)
            return (ell, old_updated, new_updated, linearized_observation), None

        (ell, _, updated_state, linearized_observation), _ = \
                    jax.lax.stop_gradient(jax.lax.scan(body,
                        (ell, predicted, first_iterate, linearized_observation),
                                jnp.arange(1, iterations)))
        return ell, (updated_state, linearized_observation)
    return linesearch_update

class base_linesearch_filter(base_iterated_filter):
    def __init__(self, model, propagate_linearization, update_linearization,
                propagate_first=False, freeze_covariance=False, iterations=10):
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations)
        propagate = common.build_propagate(propagate_linearization,
                                        model.transition_function)
        update = common.build_update(update_linearization,
                                    model.observation_function)
        linesearch_update = build_linesearch_update(update, build_loss(model),
                                                    iterations, freeze_covariance)
        self.update = jax.jit(jax.value_and_grad(linesearch_update, argnums=-1,
                                                has_aux=True))
        forward_step = build_forward_step(linesearch_update, propagate)
        self.filter_body = build_filter_body(forward_step, model)

    def __call__(self, initial_state, initial_theta, observations,
                linearization_points=None, options=LineSearchIterationOptions()):
        return base_iterated_filter.__call__(self, initial_state, initial_theta,
                                observations, linearization_points, options)

class lsiekf(base_linesearch_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(first_taylor)
        update_linearization = LinearizationMethod(first_taylor)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations=iterations)

class lsiukf(base_linesearch_filter):
    def __init__(self, model, linearization_params, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr, linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr, linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, freeze_covariance=True, iterations=iterations)

class lsickf(base_linesearch_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, freeze_covariance=True, iterations=iterations)

class lsiplf(base_linesearch_filter):
    def __init__(self, model, propagate_linearization, update_linearization,
                propagate_first=False, inner_iterations=10, outer_iterations=10):
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first)
        propagate = common.build_propagate(propagate_linearization,
                                        model.transition_function)
        update = common.build_update(update_linearization,
                                    model.observation_function)
        ls_update = build_linesearch_iplf(update, update_linearization, model,
                            build_loss(model), inner_iterations, outer_iterations)
        self.update = jax.jit(jax.value_and_grad(ls_update, argnums=-1,
                                                has_aux=True))
        forward_step = build_forward_step(ls_update, propagate)
        self.filter_body = build_filter_body(forward_step, model)

class utlsiplf(lsiplf):
    def __init__(self, model, linearization_params, propagate_first=False,
                inner_iterations=10, outer_iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr,
                                    linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr,
                                    linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, inner_iterations, outer_iterations)

class ctlsiplf(lsiplf):
    def __init__(self, model, propagate_first=False,
                inner_iterations=10, outer_iterations=10):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, inner_iterations, outer_iterations)
