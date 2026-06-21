# =============================================================================
#  LSTMConstitutive.py
#
#  LSTM-based constitutive surrogate following the REIIS pattern:
#    "Physics-Based Self-Learning Recurrent Neural Network enhanced time
#     integration scheme for computing viscoplastic structural finite element
#     response" — Tandale, Bamer, Markert, Stoffel (CMAME 2022)
#
#  Adaptation for rate-independent J2 plasticity, large strain (mode 'large1'):
#
#  REIIS original          │  This implementation
#  ────────────────────────┼──────────────────────────────────────────────────
#  Output: Δε̇ᵖ (scalar)  │  Output: Δγ  (scalar consistency parameter)
#  Input:  [σ̂',X̂,k,K]   │  Input:  [σ̂_n(6), εₚ_n(1), Ŷ_n(1), Δε(6), J_Λ(1)]
#  2 LSTM × 128 units     │  2 LSTM × 64 units
#  Sigmoid output          │  Softplus output  (enforces Δγ ≥ 0)
#  Viscoplastic ODE loss   │  Yield consistency: f = seq_trial−(3G+h)Δγ−Y_n = 0
#  Per-QP state [S1,S2]   │  Per-QP (H,C) arrays persist across load steps
#  400 sequences           │  400 sequences × 20 steps each
#  Curriculum [2..128]     │  Curriculum [2, 4, 8, 16, 32, 64]
#
#  σ_new is RECONSTRUCTED ANALYTICALLY from Δγ (exact radial return formulas).
#  Consistent tangent C = dσ_new/dΔε via jacfwd tracing through LSTM + recon.
#
#  Layout:
#    1. LSTM primitives  (cell, step, init)
#    2. Analytic reconstruction  σ(Δγ, Δε, σ_n, …)
#    3. Physics loss  (yield consistency, Kuhn-Tucker)
#    4. Sequence data generation
#    5. Training with curriculum learning
#    6. Drop-in wrapper (same signature as constitutive_update_with_tangent)
# =============================================================================

import jax
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LSTM primitives
# ─────────────────────────────────────────────────────────────────────────────

def init_lstm_params(key, input_size, hidden_size, num_layers=2):
    """
    Returns params dict:
        params['lstm'] : list of (W_ih, W_hh, b) per layer
        params['dense']: (W, b) — hidden → scalar Δγ
    Initialised with uniform ±1/√fan_in (standard LSTM init).
    """
    params = {'lstm': [], 'dense': None}

    for layer in range(num_layers):
        key, sk1, sk2 = jax.random.split(key, 3)
        in_sz  = input_size if layer == 0 else hidden_size
        scale  = 1.0 / jnp.sqrt(in_sz)
        W_ih   = jax.random.uniform(sk1, (4 * hidden_size, in_sz),
                                    minval=-scale, maxval=scale)
        W_hh   = jax.random.uniform(sk2, (4 * hidden_size, hidden_size),
                                    minval=-scale, maxval=scale)
        b      = jnp.zeros(4 * hidden_size)
        params['lstm'].append((W_ih, W_hh, b))

    # Dense output: hidden_size → 1  (Δγ via softplus)
    key, sk = jax.random.split(key)
    scale_d = 1.0 / jnp.sqrt(hidden_size)
    W_d = jax.random.uniform(sk, (1, hidden_size), minval=-scale_d, maxval=scale_d)
    b_d = jnp.zeros(1)
    params['dense'] = (W_d, b_d)

    return params


def lstm_cell(W_ih, W_hh, b, h, c, x):
    """
    Single LSTM cell step.
      h, c : (hidden_size,)
      x    : (input_size,)
    Returns h_new, c_new — both (hidden_size,).
    """
    gates = W_ih @ x + W_hh @ h + b          # (4·hidden,)
    i, f, g, o = jnp.split(gates, 4, axis=0)
    i = jax.nn.sigmoid(i)                     # input  gate
    f = jax.nn.sigmoid(f)                     # forget gate
    g = jnp.tanh(g)                           # cell   gate
    o = jax.nn.sigmoid(o)                     # output gate
    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return h_new, c_new


def lstm_forward_step(params, H, C, x):
    """
    One time-step through all LSTM layers + dense output.

    H, C : (num_layers, hidden_size) — hidden/cell state for all layers
    x    : (input_size,)

    Returns:
        dgamma : scalar ≥ 0   (Δγ prediction via softplus)
        H_new  : (num_layers, hidden_size)
        C_new  : (num_layers, hidden_size)
    """
    current = x
    H_rows, C_rows = [], []

    for i, (W_ih, W_hh, b) in enumerate(params['lstm']):
        h_new, c_new = lstm_cell(W_ih, W_hh, b, H[i], C[i], current)
        H_rows.append(h_new)
        C_rows.append(c_new)
        current = h_new                        # feed into next layer

    H_new = jnp.stack(H_rows)                 # (num_layers, hidden_size)
    C_new = jnp.stack(C_rows)

    W_d, b_d = params['dense']
    dgamma = jax.nn.softplus(W_d @ current + b_d)[0]   # scalar, ≥ 0 by softplus
    return dgamma, H_new, C_new


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Analytic reconstruction of σ_new, εₚ_new, Y_new from predicted Δγ
#
#  Identical to the classical radial return formulas (Simo-Hughes Box 3.2)
#  applied with the LSTM-predicted Δγ instead of the closed-form value.
#  This means kinematics (push-forward by Λ) and yield check are exact;
#  only the corrector scalar Δγ comes from the network.
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_dgamma(delta_eps_v, sigma_n_v, eps_p_n, Y_n,
                             E, nu, h_hard,
                             Lambda, J_Lambda,
                             dgamma):
    """
    Given Δγ (from LSTM), reconstruct the full updated state analytically.
    Works for large strain mode 'large1'.

    Returns: sigma_new_v (6), eps_p_new (scalar), Y_new (scalar)
    """
    from FE_functions.TensorsStr import strain_to_voigt, stress_to_voigt, voigt_to_stress

    G   = E / (2.0 * (1.0 + nu))
    K   = E / (3.0 * (1.0 - 2.0 * nu))
    lam = K - 2.0 / 3.0 * G

    # Elastic stress increment C:Δε
    ds = jnp.zeros(6)
    ds = ds.at[0].set((lam+2*G)*delta_eps_v[0] + lam*(delta_eps_v[1]+delta_eps_v[2]))
    ds = ds.at[1].set((lam+2*G)*delta_eps_v[1] + lam*(delta_eps_v[0]+delta_eps_v[2]))
    ds = ds.at[2].set((lam+2*G)*delta_eps_v[2] + lam*(delta_eps_v[0]+delta_eps_v[1]))
    ds = ds.at[3].set(G * delta_eps_v[3])
    ds = ds.at[4].set(G * delta_eps_v[4])
    ds = ds.at[5].set(G * delta_eps_v[5])

    # Large-strain push-forward (Eq. 19, Rodriguez-Ferran & Huerta 1996)
    sigma_n_mat  = voigt_to_stress(sigma_n_v)
    ds_mat       = voigt_to_stress(ds)
    pushed       = (1.0 / J_Lambda) * Lambda @ (sigma_n_mat + ds_mat) @ Lambda.T
    sigma_trial_v = stress_to_voigt(pushed)

    # Deviatoric / pressure split
    p_trial   = (sigma_trial_v[0] + sigma_trial_v[1] + sigma_trial_v[2]) / 3.0
    s_trial_v = sigma_trial_v.at[0].add(-p_trial).at[1].add(-p_trial).at[2].add(-p_trial)

    seq_trial = jnp.sqrt(1.5 * (
        s_trial_v[0]**2 + s_trial_v[1]**2 + s_trial_v[2]**2 +
        2*(s_trial_v[3]**2 + s_trial_v[4]**2 + s_trial_v[5]**2)
    ) + 1e-14)

    # Radial return with predicted Δγ (Simo-Hughes Box 3.2)
    factor      = 1.0 - 3.0 * G * dgamma / (seq_trial + 1e-14)
    s_new_v     = s_trial_v * factor
    sigma_new_v = s_new_v.at[0].add(p_trial).at[1].add(p_trial).at[2].add(p_trial)
    eps_p_new   = eps_p_n + dgamma
    Y_new       = Y_n + h_hard * dgamma

    return sigma_new_v, eps_p_new, Y_new, seq_trial


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Physics loss — yield consistency + Kuhn-Tucker conditions
#
#  Directly analogous to REIIS physics loss (constitutive ODE residual):
#    REIIS:   L_phys = ||f_viscoplastic(Δε̇ᵖ_pred)||²
#    Here:    L_phys = ||f_yield(Δγ_pred)||²  + Kuhn-Tucker penalties
#
#  Yield consistency after return mapping (linear isotropic hardening):
#    f = seq_trial − (3G + h)·Δγ − Y_n  →  should be ≤ 0
#    For plastic step: f = 0 exactly
#    For elastic step: Δγ = 0, f = seq_trial − Y_n ≤ 0
# ─────────────────────────────────────────────────────────────────────────────

def physics_loss_dgamma(dgamma, seq_trial, Y_n, G, h_hard):
    """
    Kuhn-Tucker-based physics loss on predicted Δγ.

    Three terms:
    (a) Yield admissibility: stress must not exceed yield after return
    (b) Non-negativity of Δγ
    (c) Complementarity: if elastic (f_trial ≤ 0), Δγ should be 0
    """
    # Residual of scalar consistency equation
    f_res  = seq_trial - (3.0 * G + h_hard) * dgamma - Y_n

    # (a) admissibility: f_res should be ≤ 0 after return
    L_adm  = jnp.maximum(f_res, 0.0) ** 2

    # (b) non-negativity
    L_nn   = jnp.maximum(-dgamma, 0.0) ** 2

    # (c) complementarity: Δγ · max(f_res, 0) = 0
    L_comp = (dgamma * jnp.maximum(f_res, 0.0)) ** 2

    return L_adm + L_nn + L_comp


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sequence data generation
#
#  REIIS: 400 training + 100 test sequences.
#  Here:  N_seq sequences of T_seq steps each.
#
#  Each sequence = a random continuous loading path starting from a random
#  initial state. At every step the classical radial_return_jax is called
#  to get the ground-truth Δγ.
#
#  Stored per step:
#    x     (15): normalisation-ready LSTM input
#    dgamma (1): true Δγ from classical solver
#    seq_trial : for physics loss (not fed to LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequence_data(N_seq, T_seq, E, nu, Y0, h_hard,
                            Y_init, Y_inf, delta_voce,
                            hardening='linear_isotropic', seed=42):
    """
    Returns:
        X_seq     : (N_seq, T_seq, 15)  — LSTM inputs  (un-normalised)
        DG_seq    : (N_seq, T_seq)      — true Δγ per step
        ST_seq    : (N_seq, T_seq)      — seq_trial per step (for physics loss)
    """
    from FE_functions.RadialReturn import radial_return_jax

    rng = np.random.RandomState(seed)
    G   = E / (2.0 * (1.0 + nu))

    X_all, DG_all, ST_all = [], [], []

    for seq in range(N_seq):
        # ── Random initial state ─────────────────────────────────────────
        eps_p = float(rng.uniform(0.0, 0.05))
        Y     = float(Y0 + h_hard * eps_p)

        # Random stress inside yield surface (50–90 % of Y)
        s_dir = rng.randn(6); s_dir[3:] *= 0.5
        seq_dir = np.sqrt(1.5 * (s_dir[0]**2+s_dir[1]**2+s_dir[2]**2
                                  + 2*(s_dir[3]**2+s_dir[4]**2+s_dir[5]**2)))
        sigma_v = jnp.array(s_dir / (seq_dir+1e-10) * rng.uniform(0.5, 0.9) * Y,
                             dtype=jnp.float64)

        X_seq_list, DG_list, ST_list = [], [], []
        eps_yield = Y0 / E

        for step in range(T_seq):
            # ── Strain increment for this step ───────────────────────────
            delta_eps_v = jnp.array(rng.randn(6) * eps_yield * 1.5,
                                     dtype=jnp.float64)

            # ── Random incremental deformation gradient ──────────────────
            perturb  = rng.randn(3, 3) * 0.10
            Lambda   = jnp.array(np.eye(3) + perturb, dtype=jnp.float64)
            J_Lambda = float(jnp.linalg.det(Lambda))
            if J_Lambda <= 0.2:
                Lambda   = jnp.eye(3, dtype=jnp.float64)
                J_Lambda = 1.0

            # ── Classical solution ────────────────────────────────────────
            sigma_new_v, eps_p_new, Y_new, _ = radial_return_jax(
                delta_eps_v, sigma_v, eps_p, Y,
                E, nu, h_hard, Y_init, Y_inf, delta_voce,
                Lambda=Lambda, J_Lambda=J_Lambda,
                strain='large1', hardening=hardening
            )

            dgamma_true = float(eps_p_new) - eps_p    # Δγ = Δεₚ for Simo convention

            # ── Compute seq_trial for physics loss ────────────────────────
            # (reuse reconstruction function for seq_trial, dgamma_true=0 gives trial state)
            _, _, _, seq_trial_val = reconstruct_from_dgamma(
                delta_eps_v, sigma_v, eps_p, Y,
                E, nu, h_hard, Lambda, J_Lambda,
                jnp.array(0.0)          # dgamma=0 → trial state
            )

            # ── Build LSTM input ──────────────────────────────────────────
            inp = jnp.concatenate([
                sigma_v,                                  # 6
                jnp.array([eps_p, Y, J_Lambda]),          # 3
                delta_eps_v                               # 6
            ])                                            # → 15

            X_seq_list.append(np.array(inp))
            DG_list.append(dgamma_true)
            ST_list.append(float(seq_trial_val))

            # ── Advance state for next step ───────────────────────────────
            sigma_v = sigma_new_v
            eps_p   = float(eps_p_new)
            Y       = float(Y_new)

        X_all.append(X_seq_list)          # (T_seq, 15)
        DG_all.append(DG_list)            # (T_seq,)
        ST_all.append(ST_list)            # (T_seq,)

    X_seq  = jnp.array(X_all,  dtype=jnp.float64)   # (N, T, 15)
    DG_seq = jnp.array(DG_all, dtype=jnp.float64)   # (N, T)
    ST_seq = jnp.array(ST_all, dtype=jnp.float64)   # (N, T)
    return X_seq, DG_seq, ST_seq


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_seq_normalisation(X_seq):
    """X_seq: (N, T, input_size)  →  per-feature mean and std."""
    flat   = X_seq.reshape(-1, X_seq.shape[-1])
    x_mean = flat.mean(axis=0)
    x_std  = flat.std(axis=0) + 1e-8
    return x_mean, x_std


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training with curriculum learning
#
#  REIIS curriculum: batch sizes [2, 4, 8, 16, 32, 64, 128], Adam lr=0.001.
#  Here:             batch sizes [2, 4, 8, 16, 32, 64],      Adam lr=0.001.
#
#  At each curriculum stage, the model trains for n_epochs_per_stage epochs
#  on mini-batches of that size before moving to the next.
#
#  Loss per sequence: mean over T steps of (data + lambda_physics * physics).
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm(X_seq_raw, DG_seq, ST_seq,
               E, nu, h_hard,
               hidden_size=64, num_layers=2,
               curriculum_batches=(2, 4, 8, 16, 32, 64),
               n_epochs_per_stage=500,
               lr=1e-3,
               lambda_physics=0.1,
               seed=0):
    """
    Trains the LSTM following the REIIS curriculum pattern.

    Returns:
        params, x_mean, x_std, hidden_size, num_layers
    """
    G = E / (2.0 * (1.0 + nu))

    x_mean, x_std = compute_seq_normalisation(X_seq_raw)
    X_seq = (X_seq_raw - x_mean) / x_std     # (N, T, 15)
    N_seq, T_seq, input_size = X_seq.shape

    key    = jax.random.PRNGKey(seed)
    params = init_lstm_params(key, input_size, hidden_size, num_layers)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    H0 = jnp.zeros((num_layers, hidden_size))
    C0 = jnp.zeros((num_layers, hidden_size))

    # ── Loss for a single sequence ────────────────────────────────────────────
    def sequence_loss(params, x_seq, dg_seq, st_seq):
        """
        x_seq : (T, 15)  — normalised
        dg_seq: (T,)     — true Δγ
        st_seq: (T,)     — seq_trial (for physics loss)
        """
        def scan_fn(carry, inputs):
            H, C = carry
            x_t, dg_true, seq_tr = inputs
            dgamma, H_new, C_new = lstm_forward_step(params, H, C, x_t)

            L_data  = (dgamma - dg_true) ** 2
            L_phys  = physics_loss_dgamma(dgamma, seq_tr,
                                          x_t[7] * x_std[7] + x_mean[7],  # Y_n denormed
                                          G, h_hard)
            step_loss = L_data + lambda_physics * L_phys
            return (H_new, C_new), step_loss

        _, step_losses = jax.lax.scan(scan_fn, (H0, C0),
                                       (x_seq, dg_seq, st_seq))
        return jnp.mean(step_losses)

    # ── Batched loss over a mini-batch of sequences ───────────────────────────
    @jax.jit
    def batch_loss(params, x_batch, dg_batch, st_batch):
        losses = jax.vmap(lambda x, d, s: sequence_loss(params, x, d, s))(
            x_batch, dg_batch, st_batch
        )
        return jnp.mean(losses)

    @jax.jit
    def step_fn(params, opt_state, x_b, dg_b, st_b):
        loss, grads = jax.value_and_grad(batch_loss)(params, x_b, dg_b, st_b)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # ── Curriculum training ───────────────────────────────────────────────────
    print(f"\nTraining LSTM  layers={num_layers} × hidden={hidden_size}")
    print(f"  {N_seq} sequences × {T_seq} steps  |  λ_phys={lambda_physics}")
    print(f"  Curriculum batches: {curriculum_batches}")
    print("─" * 60)

    rng = np.random.RandomState(seed)
    epoch_global = 0

    for batch_size in curriculum_batches:
        print(f"\n  ▶ Batch size = {batch_size}  ({n_epochs_per_stage} epochs)")
        for epoch in range(n_epochs_per_stage):
            idx = rng.permutation(N_seq)
            for start in range(0, N_seq, batch_size):
                b = idx[start : start + batch_size]
                if len(b) == 0:
                    continue
                params, opt_state, loss = step_fn(
                    params, opt_state,
                    X_seq[b], DG_seq[b], ST_seq[b]
                )
            epoch_global += 1
            if (epoch + 1) % 100 == 0:
                print(f"    epoch {epoch_global:5d}  loss = {loss:.4e}")

    print("─" * 60)
    print("Training complete.")
    return params, x_mean, x_std, hidden_size, num_layers


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Drop-in wrapper for HistoryUpd_SS
#
#  Returns a callable that manages per-QP hidden state and has the same
#  return signature as constitutive_update_with_tangent:
#
#    sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent
#
#  CRITICAL: consistent tangent via jacfwd traces through BOTH the LSTM
#  (Δγ as function of Δε) AND the analytic reconstruction (σ(Δγ, Δε)),
#  giving the exact elasto-plastic tangent automatically.
#
#  Per-QP state management:
#    lstm_states = {
#        'h': np.zeros((num_cells, num_qp, num_layers, hidden_size)),
#        'c': np.zeros((num_cells, num_qp, num_layers, hidden_size))
#    }
#  This dict is updated IN-PLACE by the returned function.
# ─────────────────────────────────────────────────────────────────────────────

def make_lstm_constitutive_update(params, x_mean, x_std,
                                   hidden_size, num_layers,
                                   E, nu, h_hard, Y_init, Y_inf, delta_voce):
    """
    Returns nn_update_fn(delta_eps_v, sigma_n_v, eps_p_n, Y_n, J_Lambda,
                         alpha_n_v, cell, qp, lstm_states)
    """
    _xm, _xs = x_mean, x_std

    def _build_input(delta_eps_v, sigma_n_v, eps_p_n, Y_n, J_Lambda):
        """Build and normalise the 15-feature input vector."""
        raw = jnp.concatenate([
            sigma_n_v,                               # 6
            jnp.array([eps_p_n, Y_n, J_Lambda]),     # 3
            delta_eps_v                              # 6
        ])
        return (raw - _xm) / _xs

    def lstm_update_fn(delta_eps_v, sigma_n_v, eps_p_n, Y_n, J_Lambda,
                       alpha_n_v, cell, qp, lstm_states, Lambda):
        """
        Full LSTM-based constitutive update for one Gauss point.

        lstm_states : dict with 'h' and 'c' arrays, shape
                      (num_cells, num_qp, num_layers, hidden_size).
                      Updated IN-PLACE after each call.

        Returns: sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent
        """
        # ── Retrieve per-QP hidden state ─────────────────────────────────
        H = jnp.array(lstm_states['h'][cell, qp])   # (num_layers, hidden_size)
        C_st = jnp.array(lstm_states['c'][cell, qp])

        # ── LSTM forward pass → Δγ ───────────────────────────────────────
        x_norm = _build_input(delta_eps_v, sigma_n_v,
                               float(eps_p_n), float(Y_n), float(J_Lambda))
        dgamma, H_new, C_new = lstm_forward_step(params, H, C_st, x_norm)

        # ── Analytic reconstruction ──────────────────────────────────────
        sigma_new_v, eps_p_new, Y_new, _ = reconstruct_from_dgamma(
            delta_eps_v, sigma_n_v, float(eps_p_n), float(Y_n),
            E, nu, h_hard, Lambda, float(J_Lambda), dgamma
        )

        # ── Consistent tangent: jacfwd traces LSTM + reconstruction ──────
        def stress_only(deps):
            x_n = _build_input(deps, sigma_n_v,
                                float(eps_p_n), float(Y_n), float(J_Lambda))
            dg, _, _ = lstm_forward_step(params, H, C_st, x_n)
            sig, _, _, _ = reconstruct_from_dgamma(
                deps, sigma_n_v, float(eps_p_n), float(Y_n),
                E, nu, h_hard, Lambda, float(J_Lambda), dg
            )
            return sig

        C_tangent = jacfwd(stress_only)(delta_eps_v)

        # ── Update per-QP hidden state in-place ──────────────────────────
        lstm_states['h'][cell, qp] = np.array(H_new)
        lstm_states['c'][cell, qp] = np.array(C_new)

        alpha_new_v = jnp.zeros(6) if alpha_n_v is None else alpha_n_v
        return sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent

    return lstm_update_fn


def make_lstm_states(num_cells, num_qp, num_layers, hidden_size):
    """
    Initialise the per-QP LSTM hidden state arrays (all zeros).
    Call this once before the simulation loop.

    Returns: dict {'h': array, 'c': array}  —  shape (num_cells, num_qp, num_layers, hidden_size)
    """
    return {
        'h': np.zeros((num_cells, num_qp, num_layers, hidden_size)),
        'c': np.zeros((num_cells, num_qp, num_layers, hidden_size))
    }
