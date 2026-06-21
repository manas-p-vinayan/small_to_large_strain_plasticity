# ═══════════════════════════════════════════════════════════════════════════
#  RUN ANY TEST  —  paste as a notebook cell
#
#  Just edit the CONFIG block, run the cell, get plots.
#  Reuses your existing globals: V, domain, u, u_old, sigma_q, eps_p_q,
#  Y_q, alpha_q, num_cells, num_qp, basis_grad, qp_weights,
#  E, nu, Y0, h, Y_init, Y_inf, delta, ConsistentTangentAssembler.
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
from generic_test_driver import build_test_setup
from Newton_Generic import Newton_Solver_Generic


# ── CONFIG ─────────────────────────────────────────────────────────────────
TEST       = 'biaxial'           # 'uniaxial' | 'compression' | 'biaxial' | 'shear'
P_MAX      = 400.0               # max traction (MPa) — for shear: max γ (e.g. 0.05)
NUM_STEPS  = 20
STRAIN     = 'small'             # 'small' | 'large1'
HARDENING  = 'linear_isotropic'  # 'linear_isotropic' | 'nonlinear_isotropic' | 'kinematic'
NEWTON_IT  = 50
# ───────────────────────────────────────────────────────────────────────────


def reset_state():
    u.x.array[:]       = 0.0
    u_old.x.array[:]   = 0.0
    sigma_q.x.array[:] = 0.0
    eps_p_q.x.array[:] = 0.0
    Y_q.x.array[:]     = Y0
    alpha_q.x.array[:] = 0.0


def run_test(test_type, P_max, num_steps, strain, hardening):
    setup = build_test_setup(test_type, V, domain, P_max, num_steps)
    print(f"\n{'='*70}")
    print(f"  {setup['label'].upper()}   "
          f"| strain={strain} | hardening={hardening}")
    print(f"{'='*70}")

    reset_state()
    assembler = ConsistentTangentAssembler(
        V, num_cells, num_qp, basis_grad, qp_weights, E, nu)

    sxx_h, syy_h, szz_h = [], [], []
    sxy_h, sxz_h, syz_h = [], [], []
    eps_p_h, ux_h       = [], []

    for n, step in enumerate(setup['load_path']):
        print(f"\n── step {n+1}/{num_steps} ──")
        Newton_Solver_Generic(
            V, u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
            NEWTON_IT, num_cells, num_qp, basis_grad, qp_weights,
            setup['primary_face_dofs'], setup['bc_per_component'],
            E, nu, Y0, h, Y_init, Y_inf, delta,
            cumulative_traction   = step['cumulative_traction'],
            traction_vec_override = step['traction_vec_override'],
            extra_tractions       = step['extra_tractions'],
            prescribed_values     = step['prescribed_values'],
            strain=strain, hardening=hardening,
        )

        sigma_arr = sigma_q.x.array.reshape(num_cells, num_qp, 3, 3)
        sxx_h.append(np.mean(sigma_arr[:, :, 0, 0]))
        syy_h.append(np.mean(sigma_arr[:, :, 1, 1]))
        szz_h.append(np.mean(sigma_arr[:, :, 2, 2]))
        sxy_h.append(np.mean(sigma_arr[:, :, 0, 1]))
        sxz_h.append(np.mean(sigma_arr[:, :, 0, 2]))
        syz_h.append(np.mean(sigma_arr[:, :, 1, 2]))
        eps_p_h.append(np.max(eps_p_q.x.array))
        ux_h.append(np.max(u.x.array.reshape(-1, 3)[:, 0]))

    # Prepend zero state for plotting
    return dict(
        label      = setup['label'],
        norm_load  = setup['norm_loads'],
        sxx        = np.array([0.0] + sxx_h),
        syy        = np.array([0.0] + syy_h),
        szz        = np.array([0.0] + szz_h),
        sxy        = np.array([0.0] + sxy_h),
        sxz        = np.array([0.0] + sxz_h),
        syz        = np.array([0.0] + syz_h),
        eps_p      = np.array([0.0] + eps_p_h),
        ux         = np.array([0.0] + ux_h),
    )


def plot_result(res, test_type):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{res['label']}  —  {STRAIN} strain, {HARDENING}",
                 fontsize=12, fontweight='bold')

    axes[0].plot(res['norm_load'], res['sxx']/E, '-o', label=r'$\sigma_{xx}/E$')
    axes[0].plot(res['norm_load'], res['syy']/E, '-s', label=r'$\sigma_{yy}/E$')
    axes[0].plot(res['norm_load'], res['szz']/E, '-^', label=r'$\sigma_{zz}/E$')
    axes[0].set_xlabel('normalised load')
    axes[0].set_ylabel(r'$\sigma_{ii}/E$')
    axes[0].set_title('Normal stresses')
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(res['norm_load'], res['sxy']/E, '-o', label=r'$\sigma_{xy}/E$')
    axes[1].plot(res['norm_load'], res['sxz']/E, '-s', label=r'$\sigma_{xz}/E$')
    axes[1].plot(res['norm_load'], res['syz']/E, '-^', label=r'$\sigma_{yz}/E$')
    axes[1].set_xlabel('normalised load')
    axes[1].set_ylabel(r'$\sigma_{ij}/E$')
    axes[1].set_title('Shear stresses')
    axes[1].grid(alpha=0.3); axes[1].legend()

    axes[2].plot(res['norm_load'], res['eps_p'], '-o', color='tab:red')
    axes[2].set_xlabel('normalised load')
    axes[2].set_ylabel(r'$\bar\varepsilon^p$')
    axes[2].set_title('Equivalent plastic strain')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'result_{test_type}_{STRAIN}_{HARDENING}.png',
                dpi=180, bbox_inches='tight')
    plt.show()


# ── Run ────────────────────────────────────────────────────────────────────
res = run_test(TEST, P_MAX, NUM_STEPS, STRAIN, HARDENING)
plot_result(res, TEST)

# Sanity print
print("\nFinal state:")
print(f"  sigma_xx  = {res['sxx'][-1]:>12.4f} MPa")
print(f"  sigma_yy  = {res['syy'][-1]:>12.4f} MPa")
print(f"  sigma_zz  = {res['szz'][-1]:>12.4f} MPa")
print(f"  sigma_xy  = {res['sxy'][-1]:>12.4f} MPa")
print(f"  eps_p_max = {res['eps_p'][-1]:>12.6e}")
print(f"  u_x_max   = {res['ux'][-1]:>12.6e}")
