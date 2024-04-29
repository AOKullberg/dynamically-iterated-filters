from collections import namedtuple

import numpy as np
import jax.numpy as jnp

def combine_namedtuples(tuple_a, tuple_b, name):
    # Safe concatenation of namedtuples
    combined_fields = (tuple_a._field_defaults | tuple_b._field_defaults)
    return namedtuple(name, (combined_fields.keys()),
                    defaults=(combined_fields).values())

def extend_namedtuple(tuple_a, new_field_defaults, name):
    combined_fields = (tuple_a._field_defaults | new_field_defaults)
    return namedtuple(name, (combined_fields.keys()),
                    defaults=(combined_fields).values())

Options = namedtuple("Options",
                    ["initial_time"],
                    defaults=[0])

IterationOptions = extend_namedtuple(Options, dict(max_iter=10, gamma=1e-5),
                                    "IterationOptions")

OptimizationOptions = extend_namedtuple(Options, dict(grad_weights=jnp.ones((3,)),
                        learning_rate=1e-2, iterations=50), "OptimizationOptions")

IteratedOptimizationOptions = combine_namedtuples(IterationOptions, \
                            OptimizationOptions, "IteratedOptimizationOptions")

LineSearchOptions = namedtuple("LineSearchOptions", ["gamma", "beta"],
                                defaults=[0.25, 0.9])
LineSearchIterationOptions = combine_namedtuples(namedtuple("LineSearchIterationOptions",
                                ["linesearch"], defaults=[LineSearchOptions()]),
                                IterationOptions, "LineSearchIterationOptions")