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
from functools import partial
jax.config.update("jax_enable_x64", True)

from FE_functions.TensorsStr import strain_to_voigt, stress_to_voigt, voigt_to_stress

@partial(jax.jit, static_argnames=('strain', 'hardening'))
def radial_return_jax(delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta, alpha_n_v=None,
                      F_n=None, J_n=None, strain='small', hardening = 'linear_isotropic'):
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    lam = K - 2.0/3.0 * G

    alpha_n_v = jnp.zeros(6) if alpha_n_v is None else alpha_n_v

    # ── Build elastic incremental stress C:Δε ──────────────────────────
    delta_sigma_v = jnp.zeros(6)
    delta_sigma_v = delta_sigma_v.at[0].set((lam + 2*G)*delta_eps_v[0] + lam*(delta_eps_v[1] + delta_eps_v[2]))
    delta_sigma_v = delta_sigma_v.at[1].set((lam + 2*G)*delta_eps_v[1] + lam*(delta_eps_v[0] + delta_eps_v[2]))
    delta_sigma_v = delta_sigma_v.at[2].set((lam + 2*G)*delta_eps_v[2] + lam*(delta_eps_v[0] + delta_eps_v[1]))
    delta_sigma_v = delta_sigma_v.at[3].set(2*G*delta_eps_v[3])
    delta_sigma_v = delta_sigma_v.at[4].set(2*G*delta_eps_v[4])
    delta_sigma_v = delta_sigma_v.at[5].set(2*G*delta_eps_v[5])

    # ── Build trial stress depending on strain mode ─────────────────────
    if strain == 'small':
        sigma_trial_v = sigma_n_v + delta_sigma_v

    elif strain == 'large1':
        sigma_n_mat       = voigt_to_stress(sigma_n_v)           # 3x3
        delta_sigma_mat   = voigt_to_stress(delta_sigma_v)       # 3x3
        pushed            = (1.0 / J_n) * F_n @ (sigma_n_mat + delta_sigma_mat) @ F_n.T
        sigma_trial_v     = stress_to_voigt(pushed)

    # ── Yield check and return mapping ──────────────────
    p_trial   = (sigma_trial_v[0] + sigma_trial_v[1] + sigma_trial_v[2]) / 3.0
    s_trial_v = sigma_trial_v.at[0].add(-p_trial).at[1].add(-p_trial).at[2].add(-p_trial)

    eta_v   = s_trial_v - alpha_n_v #For isotropic hardening, alpha_n_v = 0, so eta_v = s_trial_v. For kinematic hardening, alpha_n_v is the backstress.
    seq_trial = jnp.sqrt(1.5 * (
        eta_v[0]**2 + eta_v[1]**2 + eta_v[2]**2 +
        2*(eta_v[3]**2 + eta_v[4]**2 + eta_v[5]**2)
    ))



    f_trial = seq_trial - Y_n

    def elastic_return(_):
        return sigma_trial_v, eps_p_n, Y_n, alpha_n_v

    #--------------------Linear Isotropic hardening-------------------------------------
    def plastic_return(_):
        if hardening == 'linear_isotropic':
            print("Running linear isotropic hardening return mapping......")
            dgamma      = f_trial / (3.0*G + h)
            n_dir_v     = s_trial_v / seq_trial
            s_new_v     = s_trial_v - 2.0*G*dgamma*n_dir_v
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + jnp.sqrt(2.0/3.0) * dgamma
            Y_new       = Y_n + h * jnp.sqrt(2.0/3.0) * dgamma
            alpha_new_v = alpha_n_v 
            return sigma_new_v, eps_p_new, Y_new, alpha_new_v

    #--------------------Non-linear Isotropic hardening------------------------------------
        elif hardening == 'nonlinear_isotropic': #Voce model
            print("Running non-linear isotropic hardening return mapping......")
            def Y_voce(ep):
                return Y_init + (Y_inf - Y_init) * (1.0 - jnp.exp(-delta * ep))

            def dY_voce(ep):
                return (Y_inf - Y_init) * delta * jnp.exp(-delta * ep)

            def newton_step(_, dgamma):
                ep_new = eps_p_n + jnp.sqrt(2.0/3.0) * dgamma
                f      = seq_trial - 3.0*G*dgamma - Y_voce(ep_new)
                df     = -3.0*G - dY_voce(ep_new) * jnp.sqrt(2.0/3.0)
                return dgamma - f/df

            # Initial guess
            dgamma_init = f_trial / (3.0*G + dY_voce(eps_p_n) * jnp.sqrt(2.0/3.0))
            dgamma = jax.lax.fori_loop(0, 50, newton_step, dgamma_init)

            n_dir_v     = s_trial_v / seq_trial
            s_new_v     = s_trial_v - 2.0*G*dgamma*n_dir_v
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + jnp.sqrt(2.0/3.0) * dgamma
            Y_new       = Y_voce(eps_p_new)
            alpha_new_v = alpha_n_v 
            return sigma_new_v, eps_p_new, Y_new, alpha_new_v

        #--------------------Linear Kinematic hardening-------------------------------------
        elif hardening == 'kinematic': #Armstrong–Frederick rule for nonlinear kinematic hardening
            print("Running non-linear kinematic hardening return mapping......")
            C_af     = h
            gamma_af = delta

            def newton_step_kin(_, dgamma):
                denom = 1.0 + gamma_af * dgamma
                f     = seq_trial - 3.0*G*dgamma - (2.0/3.0)*C_af*(dgamma/denom) - Y_init
                df    = -3.0*G - (2.0/3.0)*C_af*(1.0/denom**2)
                return dgamma - f/df

            dgamma_init = f_trial / (3.0*G + (2.0/3.0)*C_af)
            dgamma      = jax.lax.fori_loop(0, 50, newton_step_kin, dgamma_init)

            denom       = 1.0 + gamma_af * dgamma
            n_dir_v     = eta_v / seq_trial
            s_new_v     = s_trial_v - 2.0*G*dgamma*n_dir_v
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + jnp.sqrt(2.0/3.0) * dgamma   # scalar eps_p unchanged
            Y_new       = Y_n                                      # no isotropic hardening
            alpha_new_v = (alpha_n_v + (2.0/3.0)*C_af*dgamma*n_dir_v )/ denom  #Check this equation
            return sigma_new_v, eps_p_new, Y_new, alpha_new_v
            

        # return sigma_new_v, eps_p_new, Y_new, alpha_new_v

    return jax.lax.cond(f_trial <= 0.0, elastic_return, plastic_return, operand=None)

@partial(jax.jit, static_argnames=('strain', 'hardening'))
def constitutive_update_with_tangent(delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta, alpha_n_v=None,
                                      F_n=None, J_n=None, strain='small', hardening = 'linear_isotropic'):
    sigma_new_v, eps_p_new, Y_new, alpha_new_v = radial_return_jax(
        delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta, alpha_n_v=alpha_n_v,
        F_n=F_n, J_n=J_n, strain=strain, hardening = hardening
    )

    def stress_func(deps):
        sig, _, _, _ = radial_return_jax(deps, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta, alpha_n_v=alpha_n_v,
                                       F_n=F_n, J_n=J_n, strain=strain, hardening = hardening)
        return sig

    C_tangent = jacfwd(stress_func)(delta_eps_v)
    return sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent