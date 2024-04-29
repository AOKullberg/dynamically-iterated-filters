from collections import namedtuple

import jax
import objax
import jax.numpy as jnp

__all__ = ["StateSpaceModel", "LinearizationParameters", "LinearizationMethod",
            "MVNormal", "LinearizedModel"]

@jax.tree_util.register_pytree_node_class
class StateSpaceModel():
    def __init__(self, transition_function, observation_function, transition_covariance, observation_covariance):
        self.transition_function = transition_function
        self.observation_function = observation_function
        self.transition_covariance = objax.StateVar(transition_covariance)
        self.observation_covariance = objax.StateVar(observation_covariance)

    def __repr__(self):
        return "StateSpaceModel: transition_function={}, \
observation_function={}, transition_covariance={}, \
observation_covariance={}".format(self.transition_function,
                                  self.observation_function,
                                  self.transition_covariance,
                                  self.observation_covariance)
    def tree_flatten(self):
        children = (self.transition_function, self.observation_function, self.transition_covariance, self.observation_covariance)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

LinearizationMethod = namedtuple("LinearizationMethod", ["method", "parameters"], defaults=(None,))

@jax.tree_util.register_pytree_node_class
class LinearizationParameters():
    def __init__(self, transition_parameters=None, observation_parameters=None):
        if not (type(transition_parameters) is object or \
                transition_parameters is None):
            if observation_parameters is None:
                observation_parameters = transition_parameters
        elif not (type(observation_parameters) is object or \
                observation_parameters is None):
            if transition_parameters is None:
                transition_parameters = observation_parameters
        self.transition_parameters = transition_parameters
        self.observation_parameters = observation_parameters

    def __repr__(self):
        return "LinearizationParameters: transition={}, observation={}".format(self.transition_parameters, self.observation_parameters)

    def tree_flatten(self):
        children = (self.transition_parameters, self.observation_parameters)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@jax.tree_util.register_pytree_node_class
class MVNormal():
    def __init__(self, mean, chol=None, cov=None):
        self.mean = mean
        _chol = chol
        _cov = cov
        if not (type(chol) is object or type(cov) is object or type(chol) is bool):
            if chol is not None:
                _chol = chol
                if chol.ndim == 2:
                    if chol.shape[0] == chol.shape[1]:
                        _chol = chol[jnp.tril_indices(chol.shape[1])]
                elif chol.ndim == 3:
                    _chol = jax.vmap(lambda x: x[jnp.tril_indices(chol.shape[1])], 0, 0)(chol)
            elif cov is not None:
                _cov = cov
        self._chol = _chol
        self._cov = _cov

    @property
    def chol(self):
        if self._chol is None:
            if self._cov.ndim > 2:
                tmp = jax.vmap(lambda x: jnp.linalg.cholesky(x)[jnp.tril_indices(self._cov.shape[1])], 0, 0)(self._cov)
            else:
                tmp = jnp.linalg.cholesky(self._cov)
        else:
            if self._chol.ndim > 1:
                tmp = jnp.zeros((self.mean.shape[0], self.mean.shape[1], self.mean.shape[1]))
                inds = jnp.tril_indices(self.mean.shape[1])
                tmp = tmp.at[:, inds[0], inds[1]].set(self._chol)
            else:
                tmp = jnp.zeros((self.mean.size, self.mean.size))
                tmp = tmp.at[jnp.tril_indices(self.mean.size)].set(self._chol)
        return tmp

    @property
    def cov(self):
        if self._cov is not None:
            return self._cov
        if self._chol is None:
            cov = self._cov
        else:
            if self._chol.ndim > 1:
                cov = jax.vmap(lambda x: jnp.atleast_2d(jnp.dot(x, x.T)), 0, 0)(self.chol)
            else:
                cov = jnp.atleast_2d(jnp.dot(self.chol, self.chol.T))
        return cov

    def __repr__(self):
        if self._chol is None:
            return "MVNormal(mean={}, cov={})".format(self.mean, self._cov)
        return "MVNormal(mean={}, chol={})".format(self.mean, self._chol)

    def tree_flatten(self):
        children = (self.mean, self._chol, self._cov)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

LinearizedModel = namedtuple("LinearizedModel", ["state_jac", "param_jac",
                                                "offset", "linearization_error"])
