import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
from ssmjax.types import StateSpaceModel, MVNormal

def build_model(T, q1, q2, s2):
    Q = np.array([[q1*T**3/3, q1*T**2/2, 0, 0, 0],
                 [q1*T**2/2, q1*T, 0, 0, 0],
                 [0, 0, q1*T**3/3, q1*T**2/2, 0],
                 [0, 0, q1*T**2/2, q1*T, 0],
                 [0, 0, 0, 0, q2*T]])
    R = s2*np.identity(2)
    def transition_function(x, *args, **kwargs):
        px, vx, py, vy, delta = x
        dT = delta*T
        pxk = px + jnp.sin(dT)/delta*vx - (1-jnp.cos(dT))/delta*vy
        vxk = jnp.cos(dT)*vx - jnp.sin(dT)*vy
        pyk = py + (1-jnp.cos(dT))/delta*vx + jnp.sin(dT)/delta*vy
        vyk = jnp.sin(dT)*vx + jnp.cos(dT)*vy
        deltak = delta
        xk = jnp.vstack([pxk, vxk, pyk, vyk, deltak])
        return xk.squeeze()

    def observation_function(x, *args, **kwargs):
        return jnp.vstack([x[0],x[2]]).squeeze()

    transition_covariance = Q
    observation_covariance = R

    # def transition_covariance(k):
    #     return Q
    # def observation_covariance(k):
    #     return R

    return StateSpaceModel(transition_function, observation_function,
                        transition_covariance, observation_covariance)

T = .5
q1 = .5
q2 = 1e-2
s2 = 10**2
# def observation_function(x, *args, **kwargs):
#     return jnp.vstack([x[1],x[3]]).squeeze()

K = 130

x0 = np.array([130, 35, -20, 20, -4*np.pi/180])
s2px = 5
s2vx = 5
s2py = 5#*1e4
s2vy = 5#10
s2delta = 1e-2
P0 = np.diag([s2px, s2vx, s2py, s2vy, s2delta])

def trajectories(K, initial_state, model, rkey):
    # Trajectories are always the same.
    # rkey = jax.random.PRNGKey(13)
    L = 20
    # nx = model.transition_covariance(0).shape[0]
    nx = model.transition_covariance.shape[0]
    x = np.zeros((K, nx, L))
    rkey, subkey = jax.random.split(rkey)
    x[0] = jax.random.multivariate_normal(subkey, mean=initial_state.mean, cov=initial_state.cov, shape=(L,)).T
    for k in range(1, K):
        rkey, subkey = jax.random.split(rkey)
        # x[k] = model.transition_function(x[k-1]) + jax.random.multivariate_normal(subkey, mean=np.zeros((nx,)), cov=model.transition_covariance(k), shape=(L,)).T
        x[k] = model.transition_function(x[k-1]) + jax.random.multivariate_normal(subkey, mean=np.zeros((nx,)), cov=model.transition_covariance, shape=(L,)).T
    return x, rkey

def generate_data(rkey, nsim=5, model=build_model(T, q1, q2, s2), initial_state=MVNormal(x0, cov=P0)):
    x, rkey = trajectories(K, initial_state, model, rkey)
    x = np.tile(x, nsim)
    y = []
    # ny = model.observation_covariance(0).shape[0]
    ny = model.observation_covariance.shape[0]
    for xi in jnp.rollaxis(x, 2):
        rkey, subkey = jax.random.split(rkey)
        # y.append(model.observation_function(xi.T).T + jax.random.multivariate_normal(subkey, mean=np.zeros((ny,)), cov=model.observation_covariance(0), shape=(K,)))
        y.append(model.observation_function(xi.T).T + jax.random.multivariate_normal(subkey, mean=np.zeros((ny,)), cov=model.observation_covariance, shape=(K,)))
    y = np.stack(y, axis=2)
    return jnp.rollaxis(x, 2), jnp.rollaxis(y, 2), rkey
