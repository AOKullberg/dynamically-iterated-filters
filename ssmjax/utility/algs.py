from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from ..types import MVNormal

__all__ = ["make_linearization_points",
            "combine_mean"]

def combine_mean(one, two, alpha):
    return one.mean + alpha*(two.mean - one.mean)

def make_linearization_points(linearization_points: MVNormal,
                            propagate_first: bool,
                            n_observations: int) -> MVNormal:
    if linearization_points is not None:
        initial_linearization_point = jax.tree_map(lambda x: x[0],
                                        linearization_points)
        if n_observations == 1:
            linearization_points = None
            if propagate_first:
                linearization_points = jax.tree_map(lambda x: x[1:],
                                                linearization_points)
    else:
        initial_linearization_point = linearization_points = None
    return initial_linearization_point, linearization_points
