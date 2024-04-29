from functools import partial

import jax
import jax.numpy as jnp

from . import common
from ..types import LinearizationMethod, Options

from ..transforms.linearization import first_taylor, ct_slr, ut_slr

from ..utility import make_linearization_points

__all__ = ["base_filter", "ekf", "ukf", "ckf"]


def build_forward_step(update, propagate):
    def forward_step(
        state,
        transition_covariance,
        observation_covariance,
        observation,
        prev_linearization_point,
        linearization_point,
        options,
        theta,
    ):
        predicted_state, linearized_transition = propagate(
            transition_covariance=transition_covariance,
            prior=state,
            linearization_point=prev_linearization_point,
            theta=theta,
            options=options,
        )
        loglikelihood, (updated_state, linearized_observation) = update(
            observation_covariance=observation_covariance,
            predicted=predicted_state,
            observation=observation,
            linearization_point=linearization_point,
            theta=theta,
            options=options,
        )
        return loglikelihood, (
            updated_state,
            updated_state,
            linearized_transition,
            linearized_observation,
        )

    return jax.value_and_grad(forward_step, argnums=-1, has_aux=True)


def build_filter_body(forward_step, model):
    def body(carry, inputs):
        running_ell, current_state, prev_linearization_point, options, theta = carry
        (
            observation,
            linearization_point,
            transition_covariance,
            observation_covariance,
            k,
        ) = inputs
        (
            ell,
            (updated_state, _, linearized_transition, linearized_observation),
        ), theta_grad = forward_step(
            current_state,
            transition_covariance,
            observation_covariance,
            observation,
            prev_linearization_point,
            linearization_point,
            options,
            theta,
        )
        if linearization_point is None:
            linearization_point = updated_state
        carry = (running_ell + ell, updated_state, linearization_point, options, theta)
        output = (
            updated_state,
            linearized_transition,
            linearized_observation,
            theta_grad,
        )
        return carry, output

    return body


class base_filter:
    def __init__(
        self,
        model,
        propagate_linearization,
        update_linearization,
        propagate_first=False,
    ):
        propagate = common.build_propagate(
            propagate_linearization, model.transition_function
        )
        update = common.build_update(update_linearization, model.observation_function)
        self.propagate = propagate
        self.update = jax.jit(jax.value_and_grad(update, argnums=-1, has_aux=True))
        forward_step = build_forward_step(update, propagate)
        self.filter_body = build_filter_body(forward_step, model)
        self.model = model
        self.propagate_first = propagate_first

    def __call__(
        self,
        initial_state,
        initial_theta,
        observations,
        linearization_points=None,
        options=Options(),
    ):
        n_observations = observations.shape[0]

        if self.model.transition_covariance.ndim != 3:
            transition_covariance = jnp.tile(
                self.model.transition_covariance.value, (n_observations, 1, 1)
            )
        else:
            transition_covariance = self.model.transition_covariance.value
        if self.model.observation_covariance.ndim != 3:
            observation_covariance = jnp.tile(
                self.model.observation_covariance.value, (n_observations, 1, 1)
            )
        else:
            observation_covariance = self.model.observation_covariance.value

        initial_linearization_point, linearization_points = make_linearization_points(
            linearization_points, self.propagate_first, n_observations
        )
        start_index = 0
        ell = 0.0
        if not self.propagate_first:
            # If the initial_state is specified to cover the first observation,
            # we need to perform a simple update first - in order for the gradient
            # computations to work out sequentially.
            start_index = 1
            (ell, (initial_state, _)), theta_gradient = self.update(
                initial_state,
                observations[0],
                observation_covariance[0],
                initial_linearization_point,
                initial_theta,
                options=options,
            )
            observation_covariance = observation_covariance[start_index:]
            transition_covariance = transition_covariance[start_index:]
            observations = observations[start_index:]
            if linearization_points is not None:
                linearization_points = jax.tree_map(
                    lambda x: x[1:], linearization_points
                )

        initial_linearization_point = (
            initial_linearization_point
            if initial_linearization_point is not None
            else initial_state
        )

        (ell, *_), (filtered_states, _, _, filter_gradients) = jax.lax.scan(
            self.filter_body,
            (ell, initial_state, initial_linearization_point, options, initial_theta),
            [
                observations,
                linearization_points,
                transition_covariance,
                observation_covariance,
                jnp.arange(start_index, n_observations) + options.initial_time,
            ],
        )
        if not self.propagate_first:
            filtered_states = jax.tree_map(
                lambda y, z: jnp.concatenate([y[None, ...], z], 0),
                initial_state,
                filtered_states,
            )
            filter_gradients = jax.tree_map(
                lambda x, y: jnp.vstack([jnp.expand_dims(x, 0), y]),
                theta_gradient,
                filter_gradients,
            )
        return ell, filtered_states, dict(filter_gradients=filter_gradients)


class ekf(base_filter):
    def __init__(self, model, propagate_first=False):
        propagate_linearization = LinearizationMethod(first_taylor)
        update_linearization = LinearizationMethod(first_taylor)
        super().__init__(
            model, propagate_linearization, update_linearization, propagate_first
        )


class ukf(base_filter):
    def __init__(self, model, linearization_params, propagate_first=False):
        propagate_linearization = LinearizationMethod(
            ut_slr, linearization_params.transition_parameters
        )
        update_linearization = LinearizationMethod(
            ut_slr, linearization_params.observation_parameters
        )
        super().__init__(
            model, propagate_linearization, update_linearization, propagate_first
        )


class ckf(base_filter):
    def __init__(self, model, propagate_first=False):
        propagate_linearization = LinearizationMethod(ct_slr)
        update_linearization = LinearizationMethod(ct_slr)
        super().__init__(
            model, propagate_linearization, update_linearization, propagate_first
        )
