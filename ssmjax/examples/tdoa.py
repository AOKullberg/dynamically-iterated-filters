import jax
import jax.numpy as jnp
import numpy as np
import json
from scipy.linalg import block_diag
from ssmjax.types import StateSpaceModel, MVNormal

def build_model(T, q1, q2, s2):
    Q = np.array([[q1*T**3/3, q1*T**2/2, 0, 0, 0],
                 [q1*T**2/2, q1*T, 0, 0, 0],
                 [0, 0, q1*T**3/3, q1*T**2/2, 0],
                 [0, 0, q1*T**2/2, q1*T, 0],
                 [0, 0, 0, 0, q2*T]])
    Q = np.array([[q1*T**3/3, 0, q1*T**2/2, 0, 0],
                 [0, q1*T**3/3, 0, q1*T**2/2, 0],
                 [q1*T**2/2, 0, q1*T, 0, 0],
                 [0, q1*T**2/2, 0, q1*T, 0],
                 [0, 0, 0, 0, q2*T]])

    def transition_function(x, *args, **kwargs):
        #px, vx, py, vy, delta = x
        px, py, vx, vy, delta = x
        delta = jax.lax.cond(delta == 0, lambda x: 1e-16, lambda x: x, delta)
        dT = delta*T
        pxk = px + jnp.sin(dT)/delta*vx - (1-jnp.cos(dT))/delta*vy
        vxk = jnp.cos(dT)*vx - jnp.sin(dT)*vy
        pyk = py + (1-jnp.cos(dT))/delta*vx + jnp.sin(dT)/delta*vy
        vyk = jnp.sin(dT)*vx + jnp.cos(dT)*vy
        deltak = delta
        xk = jnp.vstack([pxk, pyk, vxk, vyk, deltak])
        return xk.squeeze()

    def observation_function(x, theta, **kwargs):
        px, py = jnp.atleast_2d(x)[:, :2].T
        dist = jnp.sqrt(jnp.sum((jnp.vstack([px, py])[:, :, None] - theta.reshape(2, 1, -1))**2, axis=0))
        dr = jnp.diff(dist, axis=1)
        return dr.squeeze()
    
    cov = jnp.diag(-s2[1:-1], k=1)
    R = jnp.diag(jnp.convolve(jnp.ones(2,), s2, 'valid')) + cov + cov.T

    transition_covariance = Q
    observation_covariance = R

    return StateSpaceModel(transition_function, observation_function,
                        transition_covariance, observation_covariance)

T = .5
q1 = .1
q2 = 1e-2

def build_prior(file, P0=np.identity(5)):
    with open(file, 'r') as f:
        data = json.load(f)
    if data.get('initial_state', None) is not None:
        x0 = np.array(data['initial_state'])
    else:
        x0 = np.zeros((2,))
    x0 = np.hstack([x0, np.array([0.6, 0, 0])])
    return MVNormal(x0, cov=P0)

v = 343