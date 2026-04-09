from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
import basix
import os
import jax
import jax.numpy as jnp
from jax import jacfwd
from functools import partial
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

@jax.jit
def compute_elastic_tangent_jax(E, nu):
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    lam = K - 2.0/3.0 * G
    C = jnp.zeros((6, 6))
    C = C.at[0, 0].set(lam + 2*G)
    C = C.at[1, 1].set(lam + 2*G)
    C = C.at[2, 2].set(lam + 2*G)
    C = C.at[3, 3].set(G)
    C = C.at[4, 4].set(G)
    C = C.at[5, 5].set(G)
    C = C.at[0, 1].set(lam)
    C = C.at[0, 2].set(lam)
    C = C.at[1, 0].set(lam)
    C = C.at[1, 2].set(lam)
    C = C.at[2, 0].set(lam)
    C = C.at[2, 1].set(lam)
    return C