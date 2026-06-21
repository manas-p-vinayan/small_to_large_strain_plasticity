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
def strain_to_voigt(eps):
    # Engineering shear strains (factor of 2 for off-diagonals)
    return jnp.array([eps[0,0], eps[1,1], eps[2,2], 2*eps[1,2], 2*eps[0,2], 2*eps[0,1]])

@jax.jit
def stress_to_voigt(sigma):
    # Stress Voigt: no factor of 2
    return jnp.array([sigma[0,0], sigma[1,1], sigma[2,2], sigma[1,2], sigma[0,2], sigma[0,1]])

@jax.jit
def voigt_to_stress(sigma_v):
    return jnp.array([[sigma_v[0], sigma_v[5], sigma_v[4]],
                      [sigma_v[5], sigma_v[1], sigma_v[3]],
                      [sigma_v[4], sigma_v[3], sigma_v[2]]])