import numpy as np
from LargeStrains.Residual import assemble_residual
from LargeStrains.HistoryUpd_SS import update_history_and_tangents


def Newton_Solver_Generic(
        V, u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
        num_iterations, num_cells, num_qp,
        basis_grad, qp_weights, primary_face_dofs,
        bc_per_component,            
        E, nu, Y0, h, Y_init, Y_inf, delta,
        cumulative_traction,
        traction_vec_override=None,
        extra_tractions=None,
        prescribed_values=None,         
        strain='small', hardening='linear_isotropic',
        tol=1e-8,
        solver='classical',      # 'classical' | 'nn' | 'lstm'
        nn_update_fn=None,       # from make_nn_constitutive_update() or make_lstm_constitutive_update()
        lstm_states=None):       # dict from make_lstm_states() — only needed for solver='lstm'
    """
    Returns the converged Function `u`.
    Boundary conditions are applied by row/col elimination per component.
    Non-zero prescribed displacements are imposed by setting u = value
    BEFORE the first iteration and zeroing the corresponding rows in
    every subsequent iteration (so the increment in those DOFs stays 0).
    """

    traction_vec = (traction_vec_override
                    if traction_vec_override is not None
                    else np.array([cumulative_traction, 0.0, 0.0]))

    print(f"\nNewton  (cumulative load = {cumulative_traction:.4g}):")

    # ── Snapshot converged state for restart on each iteration ─────────────
    sigma_converged = sigma_q.x.array.copy()
    eps_p_converged = eps_p_q.x.array.copy()
    Y_converged     = Y_q.x.array.copy()
    alpha_converged = alpha_q.x.array.copy()
    u_old_converged = u_old.x.array.copy()

    # eps_p at the START of this load step (for elastic / plastic status)
    eps_p_step_start = eps_p_converged.copy()

    # ── Impose prescribed (non-zero) displacements as initial guess ────────
    if prescribed_values is not None:
        for dof_array, comp, value in prescribed_values:
            for dof in dof_array:
                u.x.array[dof * 3 + comp] = value

    # ── Build a flat list of constrained (row) indices ─────────────────────
    constrained_rows = []
    for dof_array, comp in bc_per_component:
        for dof in dof_array:
            constrained_rows.append(dof * 3 + comp)
    if prescribed_values is not None:
        for dof_array, comp, _ in prescribed_values:
            for dof in dof_array:
                constrained_rows.append(dof * 3 + comp)
    constrained_rows = np.unique(np.array(constrained_rows, dtype=np.int64))

    converged = False
    du_norm0  = None

    for k in range(num_iterations):
        # restore reference state before each material update
        sigma_q.x.array[:] = sigma_converged
        eps_p_q.x.array[:] = eps_p_converged
        Y_q.x.array[:]     = Y_converged
        alpha_q.x.array[:] = alpha_converged
        u_old.x.array[:]   = u_old_converged

        update_history_and_tangents(
            u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
            num_cells, num_qp, basis_grad, V, E, nu, Y0, h,
            Y_init, Y_inf, delta,
            update_u_old=False, strain=strain, hardening=hardening,
            solver=solver, nn_update_fn=nn_update_fn, lstm_states=lstm_states
        )

        K = assembler.assemble_stiffness()
        R = assemble_residual(u, sigma_q, traction_vec, V,
                              num_cells, num_qp, basis_grad, qp_weights,
                              primary_face_dofs,
                              extra_tractions=extra_tractions)

        # ── Apply BCs (homogeneous + prescribed) by row/col elimination ────
        for row in constrained_rows:
            R[row]      = 0.0
            K[row, :]   = 0.0
            K[:, row]   = 0.0
            K[row, row] = 1.0

        try:
            du = np.linalg.solve(K, -R)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(K, -R, rcond=None)[0]

        # safety: enforce zero increment on constrained dofs
        du[constrained_rows] = 0.0

        u.x.array[:] += du

        du_norm = np.linalg.norm(du)
        if k == 0:
            du_norm0 = max(du_norm, 1e-10)
        rel_du = du_norm / du_norm0
        print(f"   iter {k:2d}:  |du|={du_norm:.3e}   rel={rel_du:.3e}")

        if rel_du < tol and k > 0:
            print(f" Converged in {k+1} iterations")
            converged = True
            break

    if not converged:
        print(f" Newton did NOT converge after {num_iterations} iterations")

    # ── Final history advance: restore, then update u_old to converged u ──
    sigma_q.x.array[:] = sigma_converged
    eps_p_q.x.array[:] = eps_p_converged
    Y_q.x.array[:]     = Y_converged
    alpha_q.x.array[:] = alpha_converged
    u_old.x.array[:]   = u_old_converged

    update_history_and_tangents(
        u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
        num_cells, num_qp, basis_grad, V, E, nu, Y0, h,
        Y_init, Y_inf, delta,
        update_u_old=True, strain=strain, hardening=hardening,
        solver=solver, nn_update_fn=nn_update_fn, lstm_states=lstm_states
    )

    # ── ELASTIC / PLASTIC status report ─────────────────────────────────────
    eps_p_final     = eps_p_q.x.array
    delta_eps_p     = eps_p_final - eps_p_step_start        
    plastic_thresh  = 1e-12

    n_plastic_now   = int(np.sum(delta_eps_p > plastic_thresh))   
    n_plastic_total = int(np.sum(eps_p_final  > plastic_thresh))  
    n_qp_total      = eps_p_final.size

    if n_plastic_now == 0:
        status = "ELASTIC"
    elif n_plastic_now == n_qp_total:
        status = "PLASTIC (all QPs yielding)"
    else:
        status = f"PLASTIC ({n_plastic_now}/{n_qp_total} QPs yielding)"

    print(f"   status: {status}")
    print(f"   max eps_p (cumulative) = {np.max(eps_p_final):.6e}   "
          f"max Δeps_p (this step) = {np.max(delta_eps_p):.6e}   "
          f"max Y = {np.max(Y_q.x.array):.4f}")

    return u