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
from FE_functions.RadialReturn import radial_return_jax
from FE_functions.BMatrix import compute_B_matrix
from FE_functions.Assembler import ConsistentTangentAssembler
from FE_functions.GradU import compute_grad_u_at_qp
from FE_functions.Residual import assemble_residual
from FE_functions.HistoryUpd_SS import update_history_and_tangents

def Newton_Solver_FullTangent(V, u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
                              num_iterations, traction_increment, num_cells, num_qp,
                              basis_grad, qp_weights, right_face_dofs,
                              bc_dofs, E, nu, Y0, h, Y_init, Y_inf, delta, cumulative_traction, strain='small', hardening = 'linear_isotropic'):

    traction_vec = np.array([cumulative_traction, 0, 0.0])
    print(f"\nNewton (cumulative_traction={cumulative_traction:.1f}):")

    # Snapshot the converged state
    sigma_converged = sigma_q.x.array.copy()
    eps_p_converged = eps_p_q.x.array.copy()
    Y_converged     = Y_q.x.array.copy()
    alpha_converged = alpha_q.x.array.copy()
    u_old_converged = u_old.x.array.copy()

    converged = False
    du_norm0  = None

    for k in range(num_iterations):
        # Restoring reference state before every material update
        sigma_q.x.array[:]  = sigma_converged   
        eps_p_q.x.array[:]  = eps_p_converged   
        Y_q.x.array[:]      = Y_converged        
        alpha_q.x.array[:]  = alpha_converged    
        u_old.x.array[:]    = u_old_converged    

        update_history_and_tangents(
            u, u_old, sigma_q, eps_p_q, Y_q,alpha_q, assembler,
            num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
            update_u_old=False, strain = strain, hardening = hardening
        )

        K = assembler.assemble_stiffness()
        R = assemble_residual(u, sigma_q, traction_vec, V, num_cells, num_qp,
                              basis_grad, qp_weights, right_face_dofs)

        for dof in bc_dofs:
            for comp in range(3):
                row = dof * 3 + comp
                R[row]    = 0.0
                K[row, :] = 0.0
                K[:, row] = 0.0
                K[row, row] = 1.0

        try:
            du = np.linalg.solve(K, -R)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(K, -R, rcond=None)[0]

        u.x.array[:] += du

        du_norm = np.linalg.norm(du)
        if k == 0:
            du_norm0 = max(du_norm, 1e-10)

        rel_du = du_norm / du_norm0
        print(f"    Iter {k}: |du|={du_norm:.3e}, rel={rel_du:.3e}")

        if rel_du < 1e-8 and k > 0:
            print(f"Converged in {k+1} iterations!")
            converged = True
            break

    if not converged:
        print(f"WARNING: Newton solver did not converge after {num_iterations} iterations!")

    # Restore reference state one more time, then do the official history update that advances u_old to the new converged position
    # 
    sigma_q.x.array[:]  = sigma_converged
    eps_p_q.x.array[:]  = eps_p_converged
    Y_q.x.array[:]      = Y_converged
    alpha_q.x.array[:] = alpha_converged
    u_old.x.array[:]    = u_old_converged

    update_history_and_tangents(
        u, u_old, sigma_q, eps_p_q, Y_q,alpha_q, assembler,
        num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
        update_u_old=True, strain = strain, hardening = hardening   #update_u_old=True- updating to the new converged state
    )
    return u
