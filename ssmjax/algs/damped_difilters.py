import jax
import jax.numpy as jnp
import jaxopt

from .filters import base_filter, build_forward_step, build_filter_body
from .ifilters import base_iterated_filter, base_dynamical_iterated_filter
from . import common

from ..types import (
    LinearizationMethod,
    LineSearchIterationOptions,
    MVNormal
)
from ..utility import combine_mean
from ..transforms.linearization import first_taylor, ct_slr, ut_slr

__all__ = [
    "base_linesearch_dynamical_filter",
    "lsdiekf",
    "lsdiukf",
    "lsdickf"
]


def build_loss(model):
    """
    Builds the loss function. It is built up of three components:
    - An observation loss relating the iterate at time k to the observation at
    time k
    - A transition loss relating the iterate at time k-1 to the iterate at time k
    - A prior loss relating the iterate at time k-1 to the prior at time k-1
    """

    def loss(
        prior,
        observation,
        observation_covariance,
        transition_covariance,
        theta,
        updated_iterate,
        smoothed_iterate,
    ):
        # ye = observation - model.observation_function(updated_iterate.mean,
        #                                             theta.mean)
        ye = observation - model.observation_function(updated_iterate, theta.mean)
        # te = updated_iterate.mean - model.transition_function(smoothed_iterate.mean,
        #                                             theta.mean)
        te = updated_iterate - model.transition_function(smoothed_iterate, theta.mean)
        # xp = prior.mean - smoothed_iterate.mean
        xp = prior.mean - smoothed_iterate
        # Observation loss
        Vy = ye.T @ jnp.linalg.solve(observation_covariance, ye)
        # Transition loss
        Vt = te.T @ jnp.linalg.solve(transition_covariance, te)
        # Prior loss
        Vp = xp.T @ jnp.linalg.solve(prior.cov, xp)
        return Vy + Vt + Vp

    return loss

def build_linesearch(
    loss,
    previous_state,
    observation,
    observation_covariance,
    transition_covariance,
    theta,
    options,
    linesearch_method,
):
    def backtracking(old_updated, current_updated, old_smoothed, current_smoothed):
        def fun(x):
            new_updated, new_smoothed = jnp.split(x, 2)
            V = loss(
                previous_state,
                observation,
                observation_covariance,
                transition_covariance,
                theta,
                new_updated.squeeze(),
                new_smoothed.squeeze(),
            )
            return V

        xk = jnp.vstack([old_updated.mean, old_smoothed.mean]).squeeze()
        pi_upd = current_updated.mean - old_updated.mean
        pi_smth = current_smoothed.mean - old_smoothed.mean
        pi = jnp.vstack([pi_upd, pi_smth]).squeeze()

        ls = jaxopt.BacktrackingLineSearch(
            fun=fun,
            maxiter=20,
            condition="strong-wolfe",
            decrease_factor=options.linesearch.beta,
            c1=options.linesearch.gamma,
        )
        stepsize, _ = ls.run(init_stepsize=1.0, params=xk, descent_direction=pi)
        return stepsize

    def exact(old_updated, current_updated, old_smoothed, current_smoothed):
        def fun(stepsize, old_iterate, current_iterate):
            new_iterate = combine_mean(old_iterate, current_iterate, stepsize)
            return loss(
                previous_state,
                observation,
                observation_covariance,
                transition_covariance,
                theta,
                *jnp.split(new_iterate, 2)
            )

        # Construct optimizer
        ls = jaxopt.LBFGSB(fun=fun, stop_if_linesearch_fails=True, maxiter=10)
        bounds = (
            jnp.zeros(
                1,
            ),
            jnp.ones(
                1,
            ),
        )

        old_iterate = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), old_updated, old_smoothed
        )
        current_iterate = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), current_updated, current_smoothed
        )

        stepsize = ls.run(
            jnp.ones(
                1,
            ).squeeze(),
            bounds=bounds,
            old_iterate=old_iterate,
            current_iterate=current_iterate,
        ).params
        # Stepsize returns as NaN if loss is constant w.r.t. stepsize
        stepsize = jax.lax.cond(
            jnp.isnan(stepsize), lambda x: jnp.array(0.0), lambda x: x, stepsize
        )
        return stepsize

    if linesearch_method == "exact":
        return exact
    elif linesearch_method == "backtracking":
        return backtracking
    else:
        raise NotImplementedError("{} is not implemented.".format(linesearch_method))


def build_linesearch_forward_step(
    update,
    propagate,
    smooth,
    loss,
    iterations,
    freeze_covariance,
    linesearch_method="exact",
):
    def forward_step(
        previous_state,
        transition_covariance,
        observation_covariance,
        observation,
        prev_linearization_point,
        linearization_point,
        options,
        theta,
    ):
        """

        Parameters
        ----------
        previous_state : ssmjax.types.MVNormal
            The prior for this step forward, i.e. $x_{k|k}$.
        """
        linesearch = build_linesearch(
            loss,
            previous_state,
            observation,
            observation_covariance,
            transition_covariance,
            theta,
            options,
            linesearch_method,
        )

        # Prepare propagation, update and smoothing functions to clean up the
        # body of the internal iterations
        prop = jax.tree_util.Partial(
            propagate,
            prior=previous_state,
            theta=theta,
            transition_covariance=transition_covariance,
        )
        upd = jax.tree_util.Partial(
            update,
            observation=observation,
            theta=theta,
            observation_covariance=observation_covariance,
        )
        smth = jax.tree_util.Partial(
            smooth,
            filtered_state=previous_state,
            theta=theta,
            transition_covariance=transition_covariance,
        )

        def internal_body(carry, _):
            # Internal body of iterations. Performs a propagation, update and
            # smoothing step iteratively.
            old_smoothed, curr_smoothed, old_updated, curr_updated, _, _, _, i = carry
            stepsize = linesearch(
                old_updated, curr_updated, old_smoothed, curr_smoothed
            )

            curr_updated = MVNormal(
                combine_mean(old_updated, curr_updated, stepsize), cov=curr_updated.cov
            )
            curr_smoothed = MVNormal(
                combine_mean(old_smoothed, curr_smoothed, stepsize),
                cov=curr_smoothed.cov,
            )
            # Corresponds to a (damped) DIUKF style update
            if freeze_covariance:
                curr_smoothed._cov = previous_state.cov
                curr_updated._cov = old_updated.cov
            # Save the current smoothed and updated iterates.
            old_smoothed, old_updated = curr_smoothed, curr_updated
            # Propagate with smoothed linearization point
            predicted_state, linearized_transition = prop(
                linearization_point=curr_smoothed
            )
            # Update with new predicted state and posterior linearization point
            _, (new_updated, linearized_observation) = upd(
                predicted=predicted_state, linearization_point=curr_updated
            )
            # Smooth with new updated point and current smoothed linearization point
            new_smoothed, cross_covariance = smth(
                smoothed_state=new_updated, linearization_point=curr_smoothed
            )
            i = i + 1
            return (
                old_smoothed,
                new_smoothed,
                old_updated,
                new_updated,
                cross_covariance,
                linearized_transition,
                linearized_observation,
                i,
            ), i

        # Perform one propagation, update and smooth step as initialization to the
        # iterations
        predicted_state, linearized_transition = prop(
            linearization_point=prev_linearization_point
        )
        ell, (updated_state, linearized_observation) = upd(
            predicted=predicted_state, linearization_point=linearization_point
        )
        smoothed_state, cross_covariance = smth(
            smoothed_state=updated_state, linearization_point=prev_linearization_point
        )

        (
            _,
            smoothed_state,
            _,
            updated_state,
            _,
            linearized_transition,
            linearized_observation,
            _,
        ), _ = jax.lax.scan(
            internal_body,
            (
                smoothed_state,
                smoothed_state,
                predicted_state,
                updated_state,
                cross_covariance,
                linearized_transition,
                linearized_observation,
                1,
            ),
            jnp.arange(iterations),
        )

        return ell, (
            updated_state,
            smoothed_state,
            linearized_transition,
            linearized_observation,
        )

    return jax.value_and_grad(forward_step, argnums=-1, has_aux=True)

class base_linesearch_dynamical_filter(base_dynamical_iterated_filter):
    def __init__(
        self,
        model,
        propagate_linearization,
        update_linearization,
        propagate_first=False,
        iterations=10,
        freeze_covariance=False,
        linesearch_method="exact",
    ):
        super().__init__(
            model,
            propagate_linearization,
            update_linearization,
            propagate_first,
            iterations,
        )
        propagate = common.build_propagate(
            propagate_linearization, model.transition_function
        )
        update = common.build_update(update_linearization, model.observation_function)
        smooth = common.build_smooth(propagate_linearization, model.transition_function)
        forward_step = build_linesearch_forward_step(
            update,
            propagate,
            smooth,
            build_loss(model),
            iterations,
            freeze_covariance,
            linesearch_method,
        )
        self.filter_body = build_filter_body(forward_step, model)

    def __call__(
        self,
        initial_state,
        initial_theta,
        observations,
        linearization_points=None,
        options=LineSearchIterationOptions(),
    ):
        return base_iterated_filter.__call__(
            self,
            initial_state,
            initial_theta,
            observations,
            linearization_points,
            options,
        )


class lsdiekf(base_linesearch_dynamical_filter):
    def __init__(self, model, **kwargs):
        propagate_linearization = LinearizationMethod(first_taylor)
        update_linearization = LinearizationMethod(first_taylor)
        super().__init__(model, propagate_linearization, update_linearization, **kwargs)


class lsdiukf(base_linesearch_dynamical_filter):
    def __init__(self, model, linearization_params, **kwargs):
        propagate_linearization = LinearizationMethod(
            ut_slr, linearization_params.transition_parameters
        )
        update_linearization = LinearizationMethod(
            ut_slr, linearization_params.observation_parameters
        )
        super().__init__(
            model,
            propagate_linearization,
            update_linearization,
            freeze_covariance=True,
            **kwargs
        )


class lsdickf(base_linesearch_dynamical_filter):
    def __init__(self, model, **kwargs):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(
            model,
            propagate_linearization,
            update_linearization,
            freeze_covariance=True,
            **kwargs
        )
