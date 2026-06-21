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

from LargeStrains.TensorsStr import strain_to_voigt, stress_to_voigt, voigt_to_stress


@partial(jax.jit, static_argnames=('strain', 'hardening'))
def radial_return_jax(delta_eps_v, sigma_n_v, eps_p_n, Y_n,
                      E, nu, h, Y_init, Y_inf, delta,
                      alpha_n_v=None,
                      F_n=None, J_n=None,
                      Lambda=None, J_Lambda=None,
                      strain='small',
                      hardening='linear_isotropic'):

    G   = E / (2.0 * (1.0 + nu))
    K   = E / (3.0 * (1.0 - 2.0 * nu))
    lam = K - 2.0/3.0 * G

    alpha_n_v = jnp.zeros(6) if alpha_n_v is None else alpha_n_v

    # ── Build elastic incremental stress in Ω_n: Δσ = C : Δε ───────────────
    delta_sigma_v = jnp.zeros(6)
    delta_sigma_v = delta_sigma_v.at[0].set((lam + 2*G)*delta_eps_v[0]
                                            + lam*(delta_eps_v[1] + delta_eps_v[2]))
    delta_sigma_v = delta_sigma_v.at[1].set((lam + 2*G)*delta_eps_v[1]
                                            + lam*(delta_eps_v[0] + delta_eps_v[2]))
    delta_sigma_v = delta_sigma_v.at[2].set((lam + 2*G)*delta_eps_v[2]
                                            + lam*(delta_eps_v[0] + delta_eps_v[1]))
    delta_sigma_v = delta_sigma_v.at[3].set(G*delta_eps_v[3])
    delta_sigma_v = delta_sigma_v.at[4].set(G*delta_eps_v[4])
    delta_sigma_v = delta_sigma_v.at[5].set(G*delta_eps_v[5])

    # ── Trial stress depending on strain mode ──────────────────────────────
    if strain == 'small':
        sigma_trial_v = sigma_n_v + delta_sigma_v

    elif strain == 'large1':
        sigma_n_mat     = voigt_to_stress(sigma_n_v)
        delta_sigma_mat = voigt_to_stress(delta_sigma_v)
        pushed = (1.0 / J_Lambda) * Lambda @ (sigma_n_mat + delta_sigma_mat) @ Lambda.T
        sigma_trial_v = stress_to_voigt(pushed)

    # ── Yield check & return mapping ───────────────────────────────────────
    p_trial   = (sigma_trial_v[0] + sigma_trial_v[1] + sigma_trial_v[2]) / 3.0
    s_trial_v = sigma_trial_v.at[0].add(-p_trial).at[1].add(-p_trial).at[2].add(-p_trial)

    eta_v = s_trial_v - alpha_n_v
    seq_trial = jnp.sqrt(1.5 * (
        eta_v[0]**2 + eta_v[1]**2 + eta_v[2]**2 +
        2*(eta_v[3]**2 + eta_v[4]**2 + eta_v[5]**2)
    ))
    f_trial = seq_trial - Y_n
    #print("f_trial =", f_trial)

    def elastic_return(_):
        return sigma_trial_v, eps_p_n, Y_n, alpha_n_v

    def plastic_return(_):

        # ─── Linear isotropic hardening ──────────────
        if hardening == 'linear_isotropic':
            dgamma      = f_trial / (3.0*G + h)
            factor      = 1.0 - 3.0*G*dgamma / seq_trial
            s_new_v     = s_trial_v * factor
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + dgamma
            Y_new       = Y_n + h * dgamma
            return sigma_new_v, eps_p_new, Y_new, alpha_n_v

        # ─── Nonlinear isotropic hardening (Voce) ──────────────────────────
        elif hardening == 'nonlinear_isotropic':
            def Y_voce(ep):
                return Y_init + (Y_inf - Y_init) * (1.0 - jnp.exp(-delta * ep))
            # def dY_voce(ep):
            #     return jax.grad(Y_voce)

            dY_voce = jax.grad(Y_voce) 

            # Newton-Raphson to find Δγ such that σ_eq_trial − 3G·Δγ = Y(ε_p+Δγ)
            def newton_step(_, dgamma):
                ep_new = eps_p_n + dgamma
                f      = seq_trial - 3.0*G*dgamma - Y_voce(ep_new)
                df     = -3.0*G - dY_voce(ep_new)
                return dgamma - f/df

            dgamma_init = f_trial / (3.0*G + dY_voce(eps_p_n))
            dgamma      = jax.lax.fori_loop(0, 50, newton_step, dgamma_init)

            factor      = 1.0 - 3.0*G*dgamma / seq_trial
            s_new_v     = s_trial_v * factor
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + dgamma
            Y_new       = Y_voce(eps_p_new)
            return sigma_new_v, eps_p_new, Y_new, alpha_n_v

        # ─── Kinematic hardening (Armstrong-Frederick) ─────────────────────
        elif hardening == 'kinematic':
            C_af, gamma_af = h, delta
            def newton_step_kin(_, dgamma):
                denom = 1.0 + gamma_af * dgamma
                f     = seq_trial - 3.0*G*dgamma - C_af*dgamma/denom - Y_init
                df    = -3.0*G - C_af/denom**2
                return dgamma - f/df

            dgamma_init = f_trial / (3.0*G + C_af)
            dgamma      = jax.lax.fori_loop(0, 50, newton_step_kin, dgamma_init)

            denom       = 1.0 + gamma_af * dgamma
            n_v         = (1.5 / seq_trial) * eta_v
            s_new_v     = s_trial_v - 2.0*G*dgamma*n_v
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            alpha_new_v = (alpha_n_v + (2.0/3.0)*C_af*dgamma*n_v) / denom
            eps_p_new   = eps_p_n + dgamma
            Y_new       = Y_n  
            return sigma_new_v, eps_p_new, Y_new, alpha_new_v

    return jax.lax.cond(f_trial <= 0.0, elastic_return, plastic_return, operand=None)


@partial(jax.jit, static_argnames=('strain', 'hardening'))
def constitutive_update_with_tangent(delta_eps_v, sigma_n_v, eps_p_n, Y_n,
                                     E, nu, h, Y_init, Y_inf, delta,
                                     alpha_n_v=None,
                                     F_n=None, J_n=None,
                                     Lambda=None, J_Lambda=None,
                                     strain='small',
                                     hardening='linear_isotropic'):
    sigma_new_v, eps_p_new, Y_new, alpha_new_v = radial_return_jax(
        delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
        alpha_n_v=alpha_n_v, F_n=F_n, J_n=J_n,
        Lambda=Lambda, J_Lambda=J_Lambda,
        strain=strain, hardening=hardening
    )

    def stress_func(deps):
        sig, _, _, _ = radial_return_jax(
            deps, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            alpha_n_v=alpha_n_v, F_n=F_n, J_n=J_n,
            Lambda=Lambda, J_Lambda=J_Lambda,
            strain=strain, hardening=hardening
        )
        return sig

    C_tangent = jacfwd(stress_func)(delta_eps_v)
    return sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent