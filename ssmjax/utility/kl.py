import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import scipy as sp

@jax.jit
def positive_definite_solve(a, b):
    factors = jax.scipy.linalg.cho_factor(a)
    def solve(matvec, x):
        return jax.scipy.linalg.cho_solve(factors, x)
    matvec = partial(jnp.dot, a)
    return jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True)

@jax.jit
def cholinv(X):
    L = jnp.linalg.cholesky(X)
    u = jax.scipy.linalg.solve_triangular(L, jnp.identity(X.shape[0]), lower=True)
    Xinv = jax.scipy.linalg.solve_triangular(L.T, u, lower=False)
    return L, Xinv

def fill_lower_diag(a, n):
    # n = (jnp.sqrt(len(a)*2)).astype(int)
    out = jnp.zeros((n,n),dtype=jnp.float64)
    out = out.at[jnp.tril_indices(n)].set(a)
    return out
fill_lower_diag = jax.jit(fill_lower_diag, static_argnums=(1,))

@jax.jit
def mvnormal_kl(mvn1, mvn2):
    if mvn1.cov is None or mvn2.cov is None:
        return 0.
    # KL divergence between two multivariate Gaussians
    # S2inv = jnp.linalg.inv(S2)
    sigma2L, sigma2inv = cholinv(mvn2.cov)
    diff = mvn2.mean-mvn1.mean
    # Compute the log determinants
    # S2L = jnp.linalg.cholesky(S2)
    sigma1L = jnp.linalg.cholesky(mvn1.cov)

    logdetS2 = 2*jnp.sum(jnp.log(sigma2L.diagonal()))
    logdetS1 = 2*jnp.sum(jnp.log(sigma1L.diagonal()))

    tr_term = jnp.trace(sigma2inv@mvn1.cov)
    det_term = logdetS2-logdetS1
    quad_term = diff.T@sigma2inv@diff
    # Abs since 32-bit floats at times cause the computed KL to be negative.
    return jnp.abs(1/2*(tr_term+det_term+quad_term-mvn1.cov.shape[0])).squeeze()

@jax.jit
def mvnormal_kl_inv(mu1, sigma1, mu2, sigma2inv):
    # Second Gaussian is parametrized by precision matrix rather than cov. matrix
    # Compute the log determinants
    sigma2Linv = jnp.linalg.cholesky(sigma2inv)
    # This will not be the exact Cholesky decomp of the non-inveted S2, but it will approximate the log-det quite well
    sigma2L = jax.scipy.linalg.solve_triangular(sigma2Linv, jnp.identity(sigma2inv.shape[0]), lower=True)
    sigma1L = jnp.linalg.cholesky(sigma1)
    logdetS2 = 2*jnp.sum(jnp.log(sigma2L.diagonal()))
    logdetS1 = 2*jnp.sum(jnp.log(sigma1L.diagonal()))
    return 1/2*(logdetS2-logdetS1+jnp.trace(sigma2inv@sigma1)+(mu2-mu1).T@sigma2inv@(mu2-mu1))
