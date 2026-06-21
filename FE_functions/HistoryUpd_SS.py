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

from LargeStrains.ElasticModulus import compute_elastic_tangent_jax
from LargeStrains.TensorsStr     import strain_to_voigt, stress_to_voigt, voigt_to_stress
from LargeStrains.RadialReturn   import radial_return_jax, constitutive_update_with_tangent
from LargeStrains.BMatrix        import compute_B_matrix
from LargeStrains.Assembler      import ConsistentTangentAssembler
from LargeStrains.GradU          import compute_grad_u_at_qp
from LargeStrains.Residual       import assemble_residual


def update_history_and_tangents(u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
                                num_cells, num_qp, basis_grad, V,
                                E, nu, Y0, h, Y_init, Y_inf, delta,
                                update_u_old=True,
                                strain='small',
                                hardening='linear_isotropic',
                                solver='classical',
                                nn_update_fn=None,
                                lstm_states=None):

    sigma_arr_old = sigma_q.x.array.reshape(num_cells, num_qp, 3, 3).copy()
    eps_p_arr_old = eps_p_q.x.array.reshape(num_cells, num_qp).copy()
    Y_arr_old     = Y_q.x.array.reshape(num_cells, num_qp).copy()
    alpha_arr_old = alpha_q.x.array.reshape(num_cells, num_qp, 6).copy()

    sigma_new = np.zeros_like(sigma_arr_old)
    eps_p_new = np.zeros_like(eps_p_arr_old)
    Y_new     = np.zeros_like(Y_arr_old)
    alpha_new = np.zeros_like(alpha_arr_old)
    I3        = np.eye(3)

    for cell in range(num_cells):
        for qp in range(num_qp):
            grad_u     = compute_grad_u_at_qp(u, cell, qp, basis_grad, V)
            grad_u_old = compute_grad_u_at_qp(u_old, cell, qp, basis_grad, V)
            sigma_n_v  = stress_to_voigt(jnp.array(sigma_arr_old[cell, qp]))
            alpha_n_v  = jnp.array(alpha_arr_old[cell, qp])

            if strain == 'small':
                # ── Standard small-strain increment ─────────────────────────
                delta_eps   = 0.5 * ((grad_u - grad_u_old) + (grad_u - grad_u_old).T)
                delta_eps_v = strain_to_voigt(jnp.array(delta_eps))

                sigma_v, eps_p, Y, alpha_new_v, C_tan = constitutive_update_with_tangent(
                    delta_eps_v, sigma_n_v,
                    float(eps_p_arr_old[cell, qp]),
                    float(Y_arr_old[cell, qp]),
                    E, nu, h, Y_init, Y_inf, delta,
                    alpha_n_v=alpha_n_v,
                    strain='small', hardening=hardening
                )

            elif strain == 'large1':
                # ─── Paper's Algorithm 1 — kinematics (shared by all solvers) ──
                F_n   = jnp.array(I3 + grad_u_old)
                F_np1 = jnp.array(I3 + grad_u)
                Lambda   = F_np1 @ jnp.linalg.inv(F_n)
                J_Lambda = jnp.linalg.det(Lambda)
                grad_du_n = Lambda - jnp.eye(3)
                delta_eps   = 0.5 * (grad_du_n + grad_du_n.T
                                     + grad_du_n.T @ grad_du_n)
                delta_eps_v = strain_to_voigt(delta_eps)

                if solver == 'lstm':
                    # ── LSTM path: predict Δγ, reconstruct σ analytically ───
                    if lstm_states is None:
                        raise ValueError(
                            "solver='lstm' requires lstm_states — pass the dict "
                            "from make_lstm_states() and keep it across load steps."
                        )
                    sigma_v, eps_p, Y, alpha_new_v, C_tan = nn_update_fn(
                        delta_eps_v, sigma_n_v,
                        float(eps_p_arr_old[cell, qp]),
                        float(Y_arr_old[cell, qp]),
                        float(J_Lambda),
                        alpha_n_v, cell, qp, lstm_states, Lambda
                    )

                elif solver == 'nn':
                    # ── MLP path (NNConstitutive) ───────────────────────────
                    if nn_update_fn is None:
                        raise ValueError(
                            "solver='nn' requires nn_update_fn — pass the "
                            "function from make_nn_constitutive_update()."
                        )
                    sigma_v, eps_p, Y, alpha_new_v, C_tan = nn_update_fn(
                        delta_eps_v, sigma_n_v,
                        float(eps_p_arr_old[cell, qp]),
                        float(Y_arr_old[cell, qp]),
                        float(J_Lambda),
                        alpha_n_v=alpha_n_v
                    )

                else:
                    # ── Classical path (default) ────────────────────────────
                    sigma_v, eps_p, Y, alpha_new_v, C_tan = constitutive_update_with_tangent(
                        delta_eps_v, sigma_n_v,
                        float(eps_p_arr_old[cell, qp]),
                        float(Y_arr_old[cell, qp]),
                        E, nu, h, Y_init, Y_inf, delta,
                        alpha_n_v = alpha_n_v,
                        Lambda    = Lambda,
                        J_Lambda  = float(J_Lambda),
                        F_n       = F_n,
                        J_n       = float(jnp.linalg.det(F_n)),
                        strain    = 'large1',
                        hardening = hardening
                    )

            else:
                raise ValueError(f"Unknown strain mode: {strain!r}")

            sigma_new[cell, qp]    = np.array(voigt_to_stress(sigma_v))
            eps_p_new[cell, qp]    = float(eps_p)
            Y_new[cell, qp]        = float(Y)
            alpha_new[cell, qp, :] = np.array(alpha_new_v)
            assembler.update_tangent(cell, qp, C_tan)

    sigma_q.x.array[:] = sigma_new.flatten()
    eps_p_q.x.array[:] = eps_p_new.flatten()
    Y_q.x.array[:]     = Y_new.flatten()
    alpha_q.x.array[:] = alpha_new.flatten()

    if update_u_old:
        u_old.x.array[:] = u.x.array[:]