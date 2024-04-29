import jax
import jax.numpy as jnp

from .filters import base_filter, build_forward_step, build_filter_body
from . import common

from ..types import LinearizationMethod, IterationOptions, \
                    IteratedOptimizationOptions
from ..transforms.linearization import first_taylor, ct_slr, ut_slr

from ..utility import mvnormal_kl

__all__ = ["base_iterated_filter",
            "iekf", "iukf", "ickf",
            "base_dynamical_iterated_filter",
            "diekf", "diukf", "dickf"]

def build_iterated_update(update, iterations, freeze_covariance=False):
    def iterated_update(predicted, observation, observation_covariance,
                        linearization_point, theta, options):
        upd = jax.tree_util.Partial(update,
                observation_covariance=observation_covariance,
                predicted=predicted, observation=observation,
                theta=theta)
        # First update
        _, (updated_state, linearized_observation) = \
            upd(linearization_point=linearization_point)
        # def cond(carry):
        #     old_updated, curr_updated, i = carry
        #     if old_updated is None:
        #         return i < options.max_iter
        #     kl = mvnormal_kl(old_updated, curr_updated)
        #     kl_cond = kl > options.gamma
        #     iter_cond = i < options.max_iter
        #     return jnp.logical_and(kl_cond, iter_cond)
        # def body(carry):
        #     _, curr_updated, i = carry
        #     old_updated = curr_updated
        #     ell, (new_updated, linearized_observation) = \
        #             upd(linearization_point=curr_updated)
        #     i += 1
        #     return (old_updated, new_updated, i)
        #
        # _, updated, i = jax.lax.stop_gradient(jax.lax.while_loop(cond, body,
        #                                     (predicted, updated_state, 2)))
        # ell, (updated_state, linearized_observation) = \
        #             upd(linearization_point=updated)
        def body(carry, _):
            _, curr_updated, _ = carry
            old_updated = curr_updated
            _, (new_updated, linearized_observation) = \
                    upd(linearization_point=curr_updated)
            if freeze_covariance:
                new_updated._cov = predicted._cov
            return (old_updated, new_updated, linearized_observation), None
        
        (_, updated_state, linearized_observation), _ = \
                    jax.lax.stop_gradient(jax.lax.scan(body,
                        (predicted, updated_state, linearized_observation),
                                jnp.arange(1, iterations)))

        ell, (updated_state, linearized_observation) = \
                    upd(linearization_point=updated_state)

        return ell, (updated_state, linearized_observation)
    return iterated_update

def build_dynamical_iterated_update(update, propagate, smooth, iterations,
                                    freeze_covariance=False):
    """Builds the joint forward step (propagate and update) for the iterated
    dynamical filters.

    Parameters
    ----------
    update : Callable
        Update function w/ signature according to ssmjax.algs.stateestimation.common
    propagate : Callable
        Propagate function w/ signature according to ssmjax.algs.stateestimation.common
    smooth : Callable
        Smooth function w/ signature according to ssmjax.algs.stateestimation.common
    iterations : int
        Number of iterations to perform

    """
    def dynamical_iterated_update(previous_state, transition_covariance,
                                observation_covariance, observation,
                                prev_linearization_point, linearization_point,
                                options, theta):
        # Prepare propagation, update and smoothing functions to clean up the
        # body of the internal iterations
        prop = jax.tree_util.Partial(propagate, prior=previous_state,
                    theta=theta, transition_covariance=transition_covariance)
        upd = jax.tree_util.Partial(update, observation=observation,
                    theta=theta, observation_covariance=observation_covariance)
        smth = jax.tree_util.Partial(smooth, filtered_state=previous_state,
                    theta=theta, transition_covariance=transition_covariance)
        def internal_body(carry, _):
            # Internal body of iterations. Performs a propagation, update and
            # smoothing step iteratively.
            _, curr_smoothed, old_updated, curr_updated, _, _, _, i = carry
            """Freeze covariance updates. If using a sigma-point filter,
            the covariance updates may lead to inconsistent linearizations due
            to a contraction of the sigma-points.
            Freezing the cov. updates turns the iterations into an IUKF style
            filter. Unfrozen is an IPLF style update."""
            if freeze_covariance:
                curr_smoothed._cov = previous_state.cov
                curr_updated._cov = old_updated.cov
            # Save the current smoothed and updated iterates.
            old_smoothed, old_updated = curr_smoothed, curr_updated
            # Propagate with smoothed linearization point
            predicted_state, linearized_transition = \
                    prop(linearization_point=curr_smoothed)
            # Update with new predicted state and posterior linearization point
            _, (new_updated, linearized_observation) = \
                    upd(predicted=predicted_state,
                        linearization_point=curr_updated)
            # Smooth with new updated point and current smoothed linearization point
            new_smoothed, cross_covariance = smth(smoothed_state=new_updated,
                                            linearization_point=curr_smoothed)
            i = i + 1
            return (old_smoothed, new_smoothed, old_updated, new_updated, \
                    cross_covariance, linearized_transition, linearized_observation,
                    i), i
        # Perform one propagation, update and smooth step as initialization to the
        # iterations
        predicted_state, linearized_transition = \
                prop(linearization_point=prev_linearization_point)
        _, (updated_state, linearized_observation) = \
                upd(predicted=predicted_state,
                    linearization_point=linearization_point)
        smoothed_state, cross_covariance = smth(smoothed_state=updated_state,
                                        linearization_point=prev_linearization_point)

        (_, smoothed_state, _, updated_state, \
                *_), _  = \
                jax.lax.scan(internal_body,
                            (smoothed_state, smoothed_state,
                            predicted_state, updated_state, cross_covariance,
                            linearized_transition, linearized_observation,
                            1),
                            jnp.arange(iterations))
        # Perform one propagation, update and smooth step as initialization to the
        # iterations
        predicted_state, linearized_transition = \
                prop(linearization_point=smoothed_state)
        ell, (updated_state, linearized_observation) = \
                upd(predicted=predicted_state,
                    linearization_point=updated_state)
        smoothed_state, cross_covariance = smth(smoothed_state=updated_state,
                                        linearization_point=smoothed_state)
        return ell, (updated_state, smoothed_state, \
                linearized_transition, linearized_observation)
    return jax.value_and_grad(dynamical_iterated_update, argnums=-1,
                            has_aux=True)

class base_iterated_filter(base_filter):
    """Base class for (simple) iterated filters -- IEKF/IUKF/ICKF.

    Parameters
    ----------
    model : StateSpaceModel
        See ssmjax.types.model
    propagate_linearization : LinearizationMethod
        See ssmjax.types.model
    update_linearization : LinearizationMethod
        See ssmjax.types.model
    propagate_first : bool
        Set to propagate before the first measurement update.
    iterations : int
        Number of iterations to use for the iterated updates.

    """
    def __init__(self, model, propagate_linearization, update_linearization,
                propagate_first=False, iterations=10, freeze_covariance=False):
        """ Initializes the filter_body (see ssmjax.algs.stateestimation.filters)
        """
        propagate = common.build_propagate(propagate_linearization,
                                        model.transition_function)
        self.propagate = propagate
        update = common.build_update(update_linearization,
                                    model.observation_function)
        iterated_update = build_iterated_update(update, iterations, freeze_covariance)
        self.update = jax.jit(jax.value_and_grad(iterated_update, argnums=-1,
                                                has_aux=True))
        forward_step = build_forward_step(iterated_update, propagate)
        self.filter_body = build_filter_body(forward_step, model)
        self.model = model
        self.propagate_first = propagate_first

    def __call__(self, initial_state, initial_theta, observations,
                linearization_points=None, options=IterationOptions()):
        return base_filter.__call__(self, initial_state, initial_theta,
                    observations, linearization_points, options)

class iekf(base_iterated_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(first_taylor)
        update_linearization = LinearizationMethod(first_taylor)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations)

class iukf(base_iterated_filter):
    def __init__(self, model, linearization_params, propagate_first=False,
                iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr, linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr, linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations, freeze_covariance=True)

class iplf(base_iterated_filter):
    def __init__(self, model, linearization_params, propagate_first=False,
                iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr, linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr, linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations)

class ickf(base_iterated_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations, freeze_covariance=True)

class base_dynamical_iterated_filter(base_filter):
    """Base class for dynamically iterated filters -- DIEKF/DIUKF/DICKF.

    Parameters
    ----------
    model : StateSpaceModel
        See ssmjax.types.model
    propagate_linearization : LinearizationMethod
        See ssmjax.types.model
    update_linearization : LinearizationMethod
        See ssmjax.types.model
    propagate_first : bool
        Set to propagate before the first measurement update.
    iterations : int
        Number of iterations to use for the iterated updates.

    """
    def __init__(self, model, propagate_linearization, update_linearization,
                propagate_first=False, iterations=10, freeze_covariance=False):
        """ Initializes the filter_body (see ssmjax.algs.stateestimation.filters)
        """
        propagate = common.build_propagate(propagate_linearization,
                                        model.transition_function)
        self.propagate = propagate
        update = common.build_update(update_linearization,
                                    model.observation_function)
        self.update = jax.jit(jax.value_and_grad(update, argnums=-1,
                                                has_aux=True))
        smooth = common.build_smooth(propagate_linearization,
                                    model.transition_function)
        self.smooth = smooth
        forward_step = build_dynamical_iterated_update(update, propagate,
                                                smooth, iterations,
                                                freeze_covariance)
        self.filter_body = build_filter_body(forward_step, model)
        self.model = model
        self.propagate_first = propagate_first

    def __call__(self, initial_state, initial_theta, observations,
                linearization_points=None, options=IterationOptions()):
        return base_filter.__call__(self, initial_state, initial_theta,
                    observations, linearization_points, options)

class diekf(base_dynamical_iterated_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(first_taylor)
        update_linearization = LinearizationMethod(first_taylor)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations)

class diukf(base_dynamical_iterated_filter):
    def __init__(self, model, linearization_params, propagate_first=False,
                iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr,
                                linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr,
                                linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations, freeze_covariance=True)

class dickf(base_dynamical_iterated_filter):
    def __init__(self, model, propagate_first=False, iterations=10):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations, freeze_covariance=True)

class diplf(base_dynamical_iterated_filter):
    def __init__(self, model, linearization_params, propagate_first=False,
                iterations=10):
        propagate_linearization = LinearizationMethod(ut_slr,
                                linearization_params.transition_parameters)
        update_linearization = LinearizationMethod(ut_slr,
                                linearization_params.observation_parameters)
        super().__init__(model, propagate_linearization, update_linearization,
                        propagate_first, iterations)
