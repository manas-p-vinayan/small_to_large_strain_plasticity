import numpy as np
from LargeStrains.HistoryUpd_SS import update_history_and_tangents
from LargeStrains.Residual import assemble_residual


# =============================================================================
# Arc-Length (Riks) Solver
# =============================================================================
# Theory:
#   Standard Newton: find u such that R(u, λ) = F_int(u) - λ·F_ext = 0
#   Arc-length adds a constraint: the step (Δu, Δλ) lies on a sphere of
#   radius Δs in (u, λ) space:
#
#       ||Δu||² + (Δλ · ψ · ||F_ext||)² = Δs²
#
#   where ψ is a scaling parameter (ψ=1 → cylindrical, ψ=0 → displacement).
#   This allows the solver to traverse limit points and snap-through.
#
# Algorithm (Crisfield 1981):
#   Predictor:  Δu_t = K⁻¹ · F_ext,   Δλ = ±Δs / sqrt(Δu_t·Δu_t + ψ²||F_ext||²)
#   Corrector:  solve 2×2 system for dλ at each Newton iteration
# =============================================================================


def arc_length_solver(V, u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
                      num_iterations, num_cells, num_qp,
                      basis_grad, qp_weights, right_face_dofs,
                      bc_dofs, E, nu, Y0, h, Y_init, Y_inf, delta,
                      # Arc-length specific parameters
                      lam,              # current load factor (scalar, modified in-place via list [lam])
                      delta_s,          # arc-length radius for this step
                      F_ext_ref,        # reference external force vector (unit traction direction)
                      psi=1.0,          # arc-length scaling: 1=spherical, 0=cylindrical
                      strain='small', hardening='linear_isotropic',
                      verbose=True):
    """
    One arc-length step starting from converged state (u, lam[0]).

    Parameters
    ----------
    lam : list[float]
        Single-element list holding current load factor. Modified in-place.
        Load applied = lam[0] * F_ext_ref.
    delta_s : float
        Arc-length (step radius). Adapt this between steps for efficiency.
    F_ext_ref : np.ndarray
        External force vector for unit load factor (traction=1).
        Shape: (n_dofs,). BC rows should already be zeroed.
    psi : float
        Arc-length scaling parameter (default 1.0 = spherical arc-length).

    Returns
    -------
    converged : bool
    lam[0] is updated to the new load factor on convergence.
    """

    # ── Snapshot converged state ──────────────────────────────────────────────
    sigma_conv  = sigma_q.x.array.copy()
    eps_p_conv  = eps_p_q.x.array.copy()
    Y_conv      = Y_q.x.array.copy()
    alpha_conv  = alpha_q.x.array.copy()
    u_old_conv  = u_old.x.array.copy()
    u_conv      = u.x.array.copy()
    lam_conv    = lam[0]

    lam_norm_sq = (psi * np.linalg.norm(F_ext_ref)) ** 2   # ψ²||F_ext||²

    # ── Predictor: tangent direction ──────────────────────────────────────────
    # Restore to converged state and compute tangent stiffness
    _restore(sigma_q, eps_p_q, Y_q, alpha_q, u_old,
             sigma_conv, eps_p_conv, Y_conv, alpha_conv, u_old_conv)
    u.x.array[:] = u_conv

    update_history_and_tangents(
        u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
        num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
        update_u_old=False, strain=strain, hardening=hardening
    )
    K = assembler.assemble_stiffness()
    _apply_bc_to_K(K, bc_dofs)

    # Tangent displacement for unit load: du_t = K⁻¹ · F_ext
    du_t = np.linalg.solve(K, F_ext_ref)

    # Predictor load increment (sign chosen to move forward on path)
    denom = np.dot(du_t, du_t) + lam_norm_sq
    delta_lam = delta_s / np.sqrt(denom)

    # Sign: keep moving in the same direction as previous step
    # (on first call, positive = increasing load)
    # The caller can flip sign of delta_s to reverse direction.

    # Predictor displacement and load factor
    delta_u = delta_lam * du_t
    u.x.array[:]  = u_conv + delta_u
    lam[0]        = lam_conv + delta_lam

    if verbose:
        print(f"\n  [Riks Predictor] Δλ={delta_lam:+.4e}, λ={lam[0]:.4f}")

    # ── Corrector iterations ──────────────────────────────────────────────────
    converged = False
    delta_u_total = delta_u.copy()   # accumulated Δu from predictor start
    delta_lam_total = delta_lam      # accumulated Δλ from predictor start

    for k in range(num_iterations):
        # Restore reference material state, update with current u
        _restore(sigma_q, eps_p_q, Y_q, alpha_q, u_old,
                 sigma_conv, eps_p_conv, Y_conv, alpha_conv, u_old_conv)

        update_history_and_tangents(
            u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
            num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
            update_u_old=False, strain=strain, hardening=hardening
        )

        K = assembler.assemble_stiffness()

        # Residual at current (u, λ):  R = F_int - λ·F_ext
        R = _residual_arc(u, sigma_q, lam[0], F_ext_ref, V,
                          num_cells, num_qp, basis_grad, qp_weights,
                          right_face_dofs, bc_dofs)
        R_norm = np.linalg.norm(R)

        _apply_bc_to_K(K, bc_dofs)

        # Solve two systems (Batoz-Dhatt decomposition):
        #   K · du_R = -R          (residual correction)
        #   K · du_F =  F_ext      (load correction direction)
        du_R = np.linalg.solve(K, -R)
        du_F = np.linalg.solve(K,  F_ext_ref)

        # Arc-length constraint:
        #   (Δu + du_R + dλ·du_F)·(Δu + du_R + dλ·du_F) + (Δλ+dλ)²·ψ²||F||² = Δs²
        # Expand and collect terms in dλ:
        #   a·dλ² + b·dλ + c = 0
        a = np.dot(du_F, du_F) + lam_norm_sq
        b = 2.0 * (np.dot(delta_u_total + du_R, du_F)
                   + delta_lam_total * lam_norm_sq)
        c = (np.dot(delta_u_total + du_R, delta_u_total + du_R)
             + delta_lam_total**2 * lam_norm_sq
             - delta_s**2)

        disc = b**2 - 4.0 * a * c
        if disc < 0:
            if verbose:
                print(f"    Iter {k}: negative discriminant — reducing arc-length")
            converged = False
            break

        sqrt_disc = np.sqrt(disc)
        dλ1 = (-b + sqrt_disc) / (2.0 * a)
        dλ2 = (-b - sqrt_disc) / (2.0 * a)

        # Choose root that maintains forward progress (dot product criterion)
        du1 = du_R + dλ1 * du_F
        du2 = du_R + dλ2 * du_F
        dot1 = np.dot(delta_u_total, du1) + delta_lam_total * dλ1 * lam_norm_sq
        dot2 = np.dot(delta_u_total, du2) + delta_lam_total * dλ2 * lam_norm_sq

        if dot1 >= dot2:
            dλ = dλ1;  du_corr = du1
        else:
            dλ = dλ2;  du_corr = du2

        # Update
        u.x.array[:]     += du_corr
        lam[0]           += dλ
        delta_u_total    += du_corr
        delta_lam_total  += dλ

        du_norm  = np.linalg.norm(du_corr)
        R_new    = np.linalg.norm(_residual_arc(
            u, sigma_q, lam[0], F_ext_ref, V,
            num_cells, num_qp, basis_grad, qp_weights,
            right_face_dofs, bc_dofs))

        if verbose:
            print(f"    Iter {k}: |R|={R_norm:.3e}→{R_new:.3e}, "
                  f"|du|={du_norm:.3e}, λ={lam[0]:.4f}, Δλ={dλ:+.4e}")

        if R_new < 1e-10 * max(np.linalg.norm(F_ext_ref), 1.0) or du_norm < 1e-12:
            if verbose:
                print(f"  Converged in {k+1} corrector iterations! λ={lam[0]:.6f}")
            converged = True
            break

    # ── Final history update ──────────────────────────────────────────────────
    if converged:
        _restore(sigma_q, eps_p_q, Y_q, alpha_q, u_old,
                 sigma_conv, eps_p_conv, Y_conv, alpha_conv, u_old_conv)
        update_history_and_tangents(
            u, u_old, sigma_q, eps_p_q, Y_q, alpha_q, assembler,
            num_cells, num_qp, basis_grad, V, E, nu, Y0, h, Y_init, Y_inf, delta,
            update_u_old=True, strain=strain, hardening=hardening
        )
    else:
        # Restore to converged state on failure
        if verbose:
            print(f"  WARNING: Arc-length did not converge after {num_iterations} iterations!")
        _restore(sigma_q, eps_p_q, Y_q, alpha_q, u_old,
                 sigma_conv, eps_p_conv, Y_conv, alpha_conv, u_old_conv)
        u.x.array[:] = u_conv
        lam[0]       = lam_conv

    return converged


# =============================================================================
# Arc-length step-size adaptation
# =============================================================================

def adapt_arc_length(delta_s, n_iter_last, n_iter_target=5,
                     delta_s_min=1e-6, delta_s_max=1.0):
    """
    Irons-Razzaque adaptation: grow/shrink arc-length based on iteration count.
        δs_new = δs * sqrt(n_target / n_actual)
    """
    factor = np.sqrt(n_iter_target / max(n_iter_last, 1))
    delta_s_new = np.clip(delta_s * factor, delta_s_min, delta_s_max)
    return delta_s_new


# =============================================================================
# Helper: build F_ext_ref (unit traction load vector)
# =============================================================================

def build_F_ext_ref(traction_direction, V, num_cells, num_qp,
                    basis_grad, qp_weights, right_face_dofs, bc_dofs,
                    u_dummy, sigma_q_zero):
    """
    Build the external force vector for unit load factor.
    traction_direction: np.array([tx, ty, tz]) — direction only, magnitude=1
    """
    from LargeStrains.Residual import assemble_residual
    # assemble_residual returns F_int - F_ext; we want just F_ext (with zero stress)
    # Set sigma to zero temporarily
    sigma_backup = sigma_q_zero.x.array.copy()
    sigma_q_zero.x.array[:] = 0.0

    # R = -F_ext when sigma=0 and u=0
    neg_F_ext = assemble_residual(
        u_dummy, sigma_q_zero, traction_direction, V,
        num_cells, num_qp, basis_grad, qp_weights, right_face_dofs
    )
    F_ext = -neg_F_ext   # F_ext = traction integral

    sigma_q_zero.x.array[:] = sigma_backup

    # Zero BC rows
    for dof in bc_dofs:
        for comp in range(3):
            F_ext[dof * 3 + comp] = 0.0

    return F_ext


# =============================================================================
# Internal helpers
# =============================================================================

def _restore(sigma_q, eps_p_q, Y_q, alpha_q, u_old,
             sigma_conv, eps_p_conv, Y_conv, alpha_conv, u_old_conv):
    sigma_q.x.array[:]  = sigma_conv
    eps_p_q.x.array[:]  = eps_p_conv
    Y_q.x.array[:]      = Y_conv
    alpha_q.x.array[:]  = alpha_conv
    u_old.x.array[:]    = u_old_conv


def _apply_bc_to_K(K, bc_dofs):
    for dof in bc_dofs:
        for comp in range(3):
            row = dof * 3 + comp
            K[row, :]   = 0.0
            K[:, row]   = 0.0
            K[row, row] = 1.0


def _residual_arc(u, sigma_q, lam, F_ext_ref, V,
                  num_cells, num_qp, basis_grad, qp_weights,
                  right_face_dofs, bc_dofs):
    """R = F_int(u) - λ·F_ext_ref, with BC rows zeroed."""
    # assemble_residual computes F_int - F_ext(traction_vec)
    # so pass traction = lam * unit_direction is wrong here;
    # instead compute: R = assemble_residual(u, sigma, lam*t_ref) 
    # but assemble_residual takes a traction_vec directly.
    # We need F_int - lam*F_ext_ref.
    # Since assemble_residual returns F_int - F_ext(t_vec),
    # and F_ext(t_vec) = F_ext_ref when |t_vec|=1,
    # we can't directly use it with scaling. So we compute manually:
    from LargeStrains.Residual import assemble_residual
    import numpy as np

    # Get F_int by passing zero traction (assemble_residual = F_int - 0 = F_int)
    # Actually simpler: just use lam * unit traction vector
    # assemble_residual(u, sigma, lam * t_hat) = F_int - lam*F_ext_ref  ✓
    # But we stored F_ext_ref = integral of t_hat, not t_hat itself.
    # The cleanest approach: R = -(assemble_residual with zero traction) + lam*F_ext_ref... 
    # 
    # Let's be precise. assemble_residual returns:
    #   R_code = F_int(u, sigma) - F_ext(traction_vec)
    # where F_ext(traction_vec) = integral of N^T * traction_vec on right face.
    # 
    # We want: R_arc = F_int - lam * F_ext_ref
    # So:      R_arc = R_code(traction=0) + lam * F_ext_ref_from_code
    # But F_ext_ref was built as -R_code(sigma=0, u=0, traction=t_ref).
    # This gets circular. Simplest fix: just call with lam as the traction magnitude
    # and use t_hat = F_ext_ref / ||F_ext_ref|| ... 
    #
    # CLEANEST: store t_ref_vec (the actual traction direction vector [tx,ty,tz])
    # and call assemble_residual with lam * t_ref_vec.
    # We do this by storing traction_ref on the solver closure.
    # For now, return the raw residual using the trick below.

    # This function is called with F_ext_ref already computed.
    # We compute F_int separately:
    R_zero_traction = assemble_residual(
        u, sigma_q, np.zeros(3), V, num_cells, num_qp,
        basis_grad, qp_weights, right_face_dofs
    )
    # R_zero_traction = F_int(u, sigma) - 0 = F_int
    R_arc = R_zero_traction - lam * F_ext_ref

    for dof in bc_dofs:
        for comp in range(3):
            R_arc[dof * 3 + comp] = 0.0

    return R_arc
