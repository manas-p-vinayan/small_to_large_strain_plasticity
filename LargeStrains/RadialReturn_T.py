import time
import numpy as np
from dolfinx import mesh, fem
from mpi4py import MPI
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

ENABLE_TIMING = True

_timing_state = {
    'call_count'     : 0,
    'first_call_done': False,
    'compile_time'   : None,
    'times_radial'   : [],
    'times_tangent'  : [],
}

def reset_timing():
    _timing_state['call_count']      = 0
    _timing_state['first_call_done'] = False
    _timing_state['compile_time']    = None
    _timing_state['times_radial']    = []
    _timing_state['times_tangent']   = []

def print_timing_summary(label=""):
    s = _timing_state
    if not s['times_radial']:
        return
    r = np.array(s['times_radial']) * 1e6
    t = np.array(s['times_tangent']) * 1e6
    total = r + t
    print(f"\n  ── Timing summary {label} ──")
    if s['compile_time'] is not None:
        print(f"    JAX compile (first call)  : {s['compile_time']*1000:.1f} ms")
    print(f"    Calls (post-compile)      : {len(r)}")
    print(f"    radial_return  avg/min/max: {r.mean():.1f} / {r.min():.1f} / {r.max():.1f} µs")
    print(f"    jacfwd tangent avg/min/max: {t.mean():.1f} / {t.min():.1f} / {t.max():.1f} µs")
    print(f"    total per QP   avg/min/max: {total.mean():.1f} / {total.min():.1f} / {total.max():.1f} µs")


# ── Toggle: set True to print dgamma at every plastic QP ──────────────────
PRINT_DGAMMA = False   # ← change to True to activate


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

    if strain == 'small':
        sigma_trial_v = sigma_n_v + delta_sigma_v
    elif strain == 'large1':
        sigma_n_mat     = voigt_to_stress(sigma_n_v)
        delta_sigma_mat = voigt_to_stress(delta_sigma_v)
        pushed = (1.0 / J_Lambda) * Lambda @ (sigma_n_mat + delta_sigma_mat) @ Lambda.T
        sigma_trial_v = stress_to_voigt(pushed)

    p_trial   = (sigma_trial_v[0] + sigma_trial_v[1] + sigma_trial_v[2]) / 3.0
    s_trial_v = sigma_trial_v.at[0].add(-p_trial).at[1].add(-p_trial).at[2].add(-p_trial)

    eta_v = s_trial_v - alpha_n_v
    seq_trial = jnp.sqrt(1.5 * (
        eta_v[0]**2 + eta_v[1]**2 + eta_v[2]**2 +
        2*(eta_v[3]**2 + eta_v[4]**2 + eta_v[5]**2)
    ))
    f_trial = seq_trial - Y_n

    def elastic_return(_):
        # ── Print for elastic QPs (dgamma = 0) ────────────────────────────
        jax.debug.callback(
            lambda f: print(f"    [elastic]  f_trial={float(f):.6e}  dgamma=0"),
            f_trial
        ) if PRINT_DGAMMA else None
        return sigma_trial_v, eps_p_n, Y_n, alpha_n_v

    def plastic_return(_):

        if hardening == 'linear_isotropic':
            dgamma  = f_trial / (3.0*G + h)
            # ── CORRECT: jax.debug.print inside JIT ───────────────────────
            jax.debug.print(
                "    [plastic / linear_iso]  f_trial={f:.6e}  dgamma={d:.6e}  Y_n={y:.4f}  Y_new={yn:.4f}",
                f=f_trial, d=dgamma, y=Y_n, yn=Y_n + h*dgamma
            )
            factor      = 1.0 - 3.0*G*dgamma / seq_trial
            s_new_v     = s_trial_v * factor
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + dgamma
            Y_new       = Y_n + h * dgamma
            return sigma_new_v, eps_p_new, Y_new, alpha_n_v

        elif hardening == 'nonlinear_isotropic':
            def Y_voce(ep):
                return Y_init + (Y_inf - Y_init) * (1.0 - jnp.exp(-delta * ep))
            dY_voce = jax.grad(Y_voce)

            def newton_step(_, dgamma):
                ep_new = eps_p_n + dgamma
                f      = seq_trial - 3.0*G*dgamma - Y_voce(ep_new)
                df     = -3.0*G - dY_voce(ep_new)
                return dgamma - f/df

            dgamma_init = f_trial / (3.0*G + dY_voce(eps_p_n))
            dgamma      = jax.lax.fori_loop(0, 50, newton_step, dgamma_init)
            jax.debug.print(
                "    [plastic / voce]        f_trial={f:.6e}  dgamma={d:.6e}  Y_new={yn:.4f}",
                f=f_trial, d=dgamma, yn=Y_voce(eps_p_n + dgamma)
            )
            factor      = 1.0 - 3.0*G*dgamma / seq_trial
            s_new_v     = s_trial_v * factor
            sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
            eps_p_new   = eps_p_n + dgamma
            Y_new       = Y_voce(eps_p_new)
            return sigma_new_v, eps_p_new, Y_new, alpha_n_v

        elif hardening == 'kinematic':
            C_af, gamma_af = h, delta

            def newton_step_kin(_, dgamma):
                denom = 1.0 + gamma_af * dgamma
                f     = seq_trial - 3.0*G*dgamma - C_af*dgamma/denom - Y_init
                df    = -3.0*G - C_af/denom**2
                return dgamma - f/df

            dgamma_init = f_trial / (3.0*G + C_af)
            dgamma      = jax.lax.fori_loop(0, 50, newton_step_kin, dgamma_init)
            jax.debug.print(
                "    [plastic / kinematic]   f_trial={f:.6e}  dgamma={d:.6e}",
                f=f_trial, d=dgamma
            )
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


def constitutive_update_timed(delta_eps_v, sigma_n_v, eps_p_n, Y_n,
                               E, nu, h, Y_init, Y_inf, delta,
                               alpha_n_v=None,
                               F_n=None, J_n=None,
                               Lambda=None, J_Lambda=None,
                               strain='small',
                               hardening='linear_isotropic'):
    s = _timing_state
    s['call_count'] += 1

    common_kwargs = dict(
        alpha_n_v=alpha_n_v, F_n=F_n, J_n=J_n,
        Lambda=Lambda, J_Lambda=J_Lambda,
        strain=strain, hardening=hardening
    )

    if not ENABLE_TIMING:
        return constitutive_update_with_tangent(
            delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            **common_kwargs
        )

    if not s['first_call_done']:
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        result = radial_return_jax(
            delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            **common_kwargs
        )
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        radial_compile_ms = (t1 - t0) * 1000

        t0 = time.perf_counter()
        full_result = constitutive_update_with_tangent(
            delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            **common_kwargs
        )
        jax.block_until_ready(full_result)
        t1 = time.perf_counter()
        tangent_compile_ms = (t1 - t0) * 1000

        t_total_end = time.perf_counter()
        s['compile_time'] = t_total_end - t_total_start
        s['first_call_done'] = True

        print(f"\n  ┌─ JAX compilation (QP call #{s['call_count']}) ─────────────────────")
        print(f"  │  radial_return_jax compile : {radial_compile_ms:.1f} ms")
        print(f"  │  jacfwd tangent compile    : {tangent_compile_ms:.1f} ms")
        print(f"  │  total compile time        : {s['compile_time']*1000:.1f} ms")
        print(f"  └────────────────────────────────────────────────────────")
        return full_result

    else:
        t0 = time.perf_counter()
        _ = radial_return_jax(
            delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            **common_kwargs
        )
        jax.block_until_ready(_)
        t1 = time.perf_counter()
        t_radial = t1 - t0
        s['times_radial'].append(t_radial)

        t0 = time.perf_counter()
        result = constitutive_update_with_tangent(
            delta_eps_v, sigma_n_v, eps_p_n, Y_n, E, nu, h, Y_init, Y_inf, delta,
            **common_kwargs
        )
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        t_full = t1 - t0
        t_tangent = t_full - t_radial
        s['times_tangent'].append(t_tangent)

        if s['call_count'] <= 5 or s['call_count'] % 50 == 0:
            print(f"  QP #{s['call_count']:4d}:  "
                  f"radial={t_radial*1e6:.1f} µs  "
                  f"tangent={t_tangent*1e6:.1f} µs  "
                  f"total={t_full*1e6:.1f} µs")

        return result