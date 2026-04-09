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
from FE_functions.TensorsStr import strain_to_voigt, stress_to_voigt, voigt_to_stress
from FE_functions.BMatrix import compute_B_matrix
jax.config.update("jax_enable_x64", True)

def assemble_residual(u, sigma_q, traction, V, num_cells, num_qp, basis_grad, qp_weights,
                      right_face_dofs):
    ndof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    R = np.zeros(ndof)

    sigma_arr = sigma_q.x.array.reshape(num_cells, num_qp, 3, 3)

    for cell in range(num_cells):
        cell_dofs = V.dofmap.cell_dofs(cell)
        R_cell = np.zeros(len(cell_dofs) * 3)

        for qp in range(num_qp):
            sigma_voigt = stress_to_voigt(jnp.array(sigma_arr[cell, qp]))
            weight = qp_weights[qp]
            grad_N = basis_grad[qp, :, :]

            for i in range(len(cell_dofs)):
                B_i = compute_B_matrix(grad_N, i) 
                R_i = B_i.T @ np.array(sigma_voigt) * weight
                for comp in range(3):
                    R_cell[i*3 + comp] += R_i[comp]

        # Assemble into global
        for i, dof in enumerate(cell_dofs):
            for comp in range(3):
                R[dof * 3 + comp] += R_cell[i*3 + comp]

    num_right = len(right_face_dofs)
    for dof in right_face_dofs:
        R[dof * 3 + 0] -= traction[0] / num_right  # x-direction traction
        # R[dof * 3 + 1] -= traction[1] / num_right  # y-direction
        # R[dof * 3 + 2] -= traction[2] / num_right  # z-direction  

    return R