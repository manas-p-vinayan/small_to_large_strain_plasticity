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

def compute_B_matrix(grad_N, node_idx):
    dN_dx, dN_dy, dN_dz = grad_N[node_idx, 0], grad_N[node_idx, 1], grad_N[node_idx, 2]

    B = np.array([
        [dN_dx,   0.0,    0.0  ],  
        [  0.0, dN_dy,    0.0  ],  
        [  0.0,   0.0,  dN_dz  ],  
        [  0.0, dN_dz,  dN_dy  ],  
        [dN_dz,   0.0,  dN_dx  ],  
        [dN_dy, dN_dx,    0.0  ]   
    ])

    return B  