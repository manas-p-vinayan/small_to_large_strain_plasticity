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

from FE_functions.ElasticModulus import compute_elastic_tangent_jax
from FE_functions.TensorsStr import strain_to_voigt, stress_to_voigt, voigt_to_stress
from FE_functions.RadialReturn import radial_return_jax, constitutive_update_with_tangent
from FE_functions.BMatrix import compute_B_matrix
from FE_functions.Assembler import ConsistentTangentAssembler
from FE_functions.GradU import compute_grad_u_at_qp
from FE_functions.Residual import assemble_residual

def update_history_and_tangents(u, u_old, sigma_q, eps_p_q, Y_q,alpha_q, assembler,
                                num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
                                update_u_old=True, strain='small', hardening = 'linear_isotropic'):
    
    sigma_arr_old = sigma_q.x.array.reshape(num_cells, num_qp, 3, 3).copy()
    eps_p_arr_old = eps_p_q.x.array.reshape(num_cells, num_qp).copy()
    Y_arr_old     = Y_q.x.array.reshape(num_cells, num_qp).copy()
    alpha_arr_old = alpha_q.x.array.reshape(num_cells, num_qp, 6).copy()

    sigma_new = np.zeros_like(sigma_arr_old)
    eps_p_new = np.zeros_like(eps_p_arr_old)
    Y_new     = np.zeros_like(Y_arr_old)
    alpha_new = np.zeros_like(alpha_arr_old)

    for cell in range(num_cells):
        for qp in range(num_qp):
            grad_u     = compute_grad_u_at_qp(u, cell, qp, basis_grad, V)
            grad_u_old = compute_grad_u_at_qp(u_old, cell, qp, basis_grad, V)
            sigma_n_v   = stress_to_voigt(jnp.array(sigma_arr_old[cell, qp]))
            alpha_n_v   = jnp.array(alpha_arr_old[cell, qp])

            if strain == 'small':
                #print("Small strain history update")
                delta_eps   = 0.5 * ((grad_u - grad_u_old) + (grad_u - grad_u_old).T)
                delta_eps_v = strain_to_voigt(jnp.array(delta_eps))

                sigma_v, eps_p, Y, alpha_new_v, C_tan = constitutive_update_with_tangent(
                    delta_eps_v, sigma_n_v,
                    float(eps_p_arr_old[cell, qp]),
                    float(Y_arr_old[cell, qp]),
                    E, nu, h, Y_init, Y_inf, delta, alpha_n_v=alpha_n_v, hardening = hardening
                )

            elif strain == 'large1':
                #print("Large strain history update")
                I         = np.eye(3)
                F_n       = I + np.array(grad_u_old)
                J_n       = np.linalg.det(F_n)
                delta_grad = np.array(grad_u) - np.array(grad_u_old)
                delta_eps  = 0.5*(delta_grad + delta_grad.T) + 0.5*(delta_grad.T @ delta_grad)
                delta_eps_v = strain_to_voigt(jnp.array(delta_eps))
                #sigma_n_v   = stress_to_voigt(jnp.array(sigma_arr_old[cell, qp]))

                sigma_v, eps_p, Y, alpha_new_v, C_tan = constitutive_update_with_tangent(
                    delta_eps_v, sigma_n_v,
                    float(eps_p_arr_old[cell, qp]),
                    float(Y_arr_old[cell, qp]),
                    E, nu, h, Y_init, Y_inf, delta, alpha_n_v=alpha_n_v,
                    F_n=jnp.array(F_n), J_n=float(J_n), strain='large1', hardening = hardening  # ← just pass these
                )

            sigma_new[cell, qp] = np.array(voigt_to_stress(sigma_v))
            eps_p_new[cell, qp] = float(eps_p)
            Y_new[cell, qp]     = float(Y)
            alpha_new[cell, qp, :] = np.array(alpha_new_v)
            assembler.update_tangent(cell, qp, C_tan) #The tangent update happens here, which is to be used while updating the stiffness matrix

    sigma_q.x.array[:] = sigma_new.flatten()
    eps_p_q.x.array[:] = eps_p_new.flatten()
    Y_q.x.array[:]     = Y_new.flatten()
    alpha_q.x.array[:] = alpha_new.flatten()

    if update_u_old:
        u_old.x.array[:] = u.x.array[:]