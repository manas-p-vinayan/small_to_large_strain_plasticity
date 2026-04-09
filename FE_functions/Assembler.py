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
from FE_functions.ElasticModulus import compute_elastic_tangent_jax
from FE_functions.BMatrix import compute_B_matrix
jax.config.update("jax_enable_x64", True)


class ConsistentTangentAssembler:

    def __init__(self, V, num_cells, num_qp, basis_grad, qp_weights, E, nu):
        self.V = V
        self.num_cells = num_cells
        self.num_qp = num_qp
        self.basis_grad = basis_grad
        self.qp_weights = qp_weights
        self.dofmap = V.dofmap
        self.bs = V.dofmap.index_map_bs
        self.num_nodes = basis_grad.shape[1]

   
        C_elastic = np.array(compute_elastic_tangent_jax(E, nu))
        self.C_tangent = np.tile(C_elastic, (num_cells, num_qp, 1, 1)) #4D tensor for the tangent

    def assemble_stiffness(self):
        ndof = self.V.dofmap.index_map.size_local * self.bs
        K = np.zeros((ndof, ndof))

        for cell in range(self.num_cells):
            cell_dofs = self.dofmap.cell_dofs(cell)
            K_cell = np.zeros((self.num_nodes * 3, self.num_nodes * 3))

            for qp in range(self.num_qp):
                C = self.C_tangent[cell, qp]  
                weight = self.qp_weights[qp]
                grad_N = self.basis_grad[qp, :, :]

                for i in range(self.num_nodes):
                    B_i = compute_B_matrix(grad_N, i)

                    for j in range(self.num_nodes):
                        B_j = compute_B_matrix(grad_N, j)

                        K_ij = B_i.T @ C @ B_j * weight

                        for comp_i in range(3):
                            for comp_j in range(3):
                                K_cell[i*3 + comp_i, j*3 + comp_j] += K_ij[comp_i, comp_j]

            # Assemble into global matrix
            for i, dof_i in enumerate(cell_dofs):
                for j, dof_j in enumerate(cell_dofs):
                    for comp_i in range(3):
                        for comp_j in range(3):
                            row = dof_i * self.bs + comp_i
                            col = dof_j * self.bs + comp_j
                            K[row, col] += K_cell[i*3 + comp_i, j*3 + comp_j]

        return K

    def update_tangent(self, cell, qp, C_new):
        self.C_tangent[cell, qp] = np.array(C_new)  


# assembler = ConsistentTangentAssembler(V, num_cells, num_qp, basis_grad, qp_weights)