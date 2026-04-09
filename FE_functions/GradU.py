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

def compute_grad_u_at_qp(u_fun, cell, qp, basis_grad, V):
    cell_dofs = V.dofmap.cell_dofs(cell)
    bs = V.dofmap.index_map_bs
    u_local = np.zeros((len(cell_dofs), 3))
    for node in range(len(cell_dofs)):
        base_idx = cell_dofs[node] * bs
        for comp in range(3):
            u_local[node, comp] = u_fun.x.array[base_idx + comp]
    grad_u = np.zeros((3, 3))
    for node in range(len(cell_dofs)):
        for i in range(3):
            for j in range(3):
                grad_u[i, j] += u_local[node, i] * basis_grad[qp, node, j]
    return grad_u