# Differentiable Large-Strain Continuum Mechanics
## Algorithmic Extensions of Small-Strain FEM using JAX and DOLFINx

> Implementing and comparing two stress update algorithms for large-strain solid mechanics, with automatic differentiation via JAX for consistent algorithmic tangents, built on top of DOLFINx.

---

## Table of Contents

- [Overview](#overview)
- [Background and Motivation](#background-and-motivation)
- [Theoretical Formulation](#theoretical-formulation)
  - [Small Strain Baseline](#small-strain-baseline)
  - [Large Strain — First Algorithm (Bathe et al.)](#large-strain--first-algorithm-bathe-et-al)
  - [Stress Push-Forward](#stress-push-forward)
- [Hardening Models](#hardening-models)
- [Repository Structure](#repository-structure)
- [Implementation Details](#implementation-details)
- [Paper Validation](#paper-validation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [References](#references)

---

## Overview

This project implements a **differentiable finite element framework** for elastoplastic analysis that bridges small-strain and large-strain formulations. Starting from a standard small-strain FEM code, the framework incrementally introduces the geometric nonlinearity needed for large deformation analysis, following the methodology of Rodríguez-Ferran and Huerta (1997).

**Key features:**

- Two stress update algorithms: small strain and large strain (First Algorithm — Bathe et al. 1975)
- Automatic consistent tangent computation using `jax.jacfwd` — no hand-derived algorithmic tangents required
- Three hardening models: linear isotropic, nonlinear isotropic (Voce), and nonlinear kinematic (Armstrong–Frederick)
- Backtracking line search for robust Newton convergence in the plastic regime
- Built on DOLFINx (FEniCSx) with JAX for GPU-acceleratable constitutive updates
- Single Gauss point paper validation notebook against analytical solutions

---

## Background and Motivation

Classical small-strain FEM assumes that deformations are small enough that the reference and deformed configurations are indistinguishable. For large deformation problems — metal forming, rubber elasticity, soft tissue mechanics — this assumption breaks down.

The key challenge in extending a small-strain code to handle large strains is the **stress update**: Cauchy stress tensors defined at different configurations cannot be directly added or subtracted. A proper push-forward transformation is needed to bring all quantities to a common configuration before operations can be performed.

This project follows the approach of Rodríguez-Ferran and Huerta (1997), which shows that:

> *"Very few features must be added to a code with small-strain and nonlinear material behavior to enable its use for large-strain analysis."*

Specifically, two additional operations are required:
1. Updating the mesh configuration
2. Computing incremental deformation gradients from nodal shape functions

---

## Theoretical Formulation

### Small Strain Baseline

In the small strain setting, the strain increment at each Gauss point is computed as the symmetrized gradient of the displacement increment:

```
Δε = ½ [(∇Δu) + (∇Δu)ᵀ]
```

The trial stress is built directly in the reference configuration:

```
σ_trial = σₙ + C : Δε
```

where `C` is the fourth-order elastic tangent tensor. A radial return mapping then checks the yield condition and corrects the stress if plastic flow occurs.

### Large Strain — First Algorithm (Bathe et al.)

The First Algorithm uses the **incremental Lagrange strain tensor**, which includes the quadratic terms necessary for large deformation objectivity (Eq. 14 in the paper):

```
Δε = ½ { [∂(Δu)/∂(ⁿx)] + [∂(Δu)/∂(ⁿx)]ᵀ + [∂(Δu)/∂(ⁿx)]ᵀ [∂(Δu)/∂(ⁿx)] }
       \_____linear part_____/                   \________quadratic part________/
```

This strain is then used to compute the elastic trial stress increment `C:Δε` in the reference configuration, which is subsequently **pushed forward** to the current configuration before the yield check.

### Stress Push-Forward

The central operation distinguishing small strain from large strain is the **push-forward Piola transformation** (Eq. 16 in the paper):

```
σ_trial = ⁿJ⁻¹ · ⁿF · (σₙ + C:Δε) · ⁿFᵀ
```

where:
- `ⁿF = I + ∇uₙ` is the deformation gradient at the start of the step
- `ⁿJ = det(ⁿF)` is the Jacobian
- `σₙ` is the converged Cauchy stress from the previous step

This transformation maps the second Piola-Kirchhoff stress to the Cauchy stress in the current configuration, ensuring **incremental objectivity** — rigid body rotations produce no spurious stresses.

The full stress update pipeline is:

```
σₙ (converged, step n)
        │
        ▼
Compute ⁿF, ⁿJ from grad(u_old)
        │
        ▼
Compute Δε (with quadratic terms)
        │
        ▼
C:Δε  (elastic incremental stress)
        │
        ▼
Push-forward:  σ_trial = J⁻¹ F (σₙ + C:Δε) Fᵀ
        │
        ▼
Yield check:  f = ‖s_trial - α‖ - Y
        │
   ┌────┴────┐
   │elastic  │plastic
   ▼         ▼
σ_trial   Radial return mapping
              │
              ▼
          σ_{n+1}, εᵖ_{n+1}, Y_{n+1}, α_{n+1}
```

---

## Hardening Models

All hardening models share the same radial return framework. The yield function in general form is:

```
f = ‖s - α‖_vM - Y(εᵖ)
```

where `α` is the back stress (zero for isotropic models) and `Y` is the current yield stress.

### 1. Linear Isotropic Hardening

The simplest model. Yield stress grows linearly with equivalent plastic strain:

```
Y_{n+1} = Yₙ + h · √(2/3) · Δγ
```

Closed-form plastic multiplier:

```
Δγ = f_trial / (3G + h)
```

### 2. Nonlinear Isotropic Hardening — Voce Model

Yield stress saturates exponentially:

```
Y(εᵖ) = Y_init + (Y_inf - Y_init)(1 - exp(-δ·εᵖ))
```

The plastic multiplier is found via Newton iteration (using `jax.lax.fori_loop` for JAX compatibility):

```
f(Δγ) = ‖η‖ - 3G·Δγ - Y(εᵖₙ + √(2/3)·Δγ) = 0
```

The saturation behavior causes the stress-strain curve to plateau, which is physically meaningful for materials that exhaust their hardening capacity.

### 3. Nonlinear Kinematic Hardening — Armstrong–Frederick Rule

Models the Bauschinger effect through an evolving back stress `α`:

```
dα = (2/3)·C_af·dεᵖ - γ_af·α·dεᵖ_eq
```

Implicit integration gives the update rule:

```
α_{n+1} = (αₙ + (2/3)·C_af·Δγ·n) / (1 + γ_af·Δγ)
```

where `n = η/‖η‖` is the flow direction in shifted deviatoric stress space. The plastic multiplier satisfies:

```
f(Δγ) = ‖η‖ - 3G·Δγ - (2/3)·C_af·Δγ/(1 + γ_af·Δγ) - Y_init = 0
```

The back stress is stored separately as a 6-component Voigt vector per quadrature point.

---

## Repository Structure

```
Small_to_large_strain_plasticity/
│
├── Small_to_large_strain_plasticity/    # Core FEM package (LargeStrains)
│   ├── RadialReturn.py                  # Constitutive update + JAX tangent
│   ├── HistoryUpd_SS.py                 # History update (sigma, eps_p, Y, alpha)
│   ├── NewtonUpd.py                     # Newton solver with backtracking line search
│   ├── Assembler.py                     # Consistent tangent stiffness assembler
│   ├── Residual.py                      # Internal force residual assembly
│   ├── GradU.py                         # Gradient of displacement at Gauss points
│   ├── BMatrix.py                       # B-matrix (strain-displacement operator)
│   ├── TensorsStr.py                    # Voigt ↔ tensor conversion utilities
│   └── ElasticModulus.py               # Elastic tangent tensor (JAX)
│
├── FE_functions/                        # Auxiliary FE utilities
│
├── dolfinx_materials/                   # DOLFINx material interface experiments
│
├── main.ipynb                           # Main simulation notebook
├── Paper_Validation_V2.ipynb           # Single-QP validation vs. analytical solutions
└── README.md
```

---

## Implementation Details

### Automatic Differentiation for Consistent Tangents

The algorithmic tangent `C_tan = dσ/dε` is computed automatically using JAX forward-mode differentiation:

```python
@partial(jax.jit, static_argnames=('strain', 'hardening'))
def constitutive_update_with_tangent(delta_eps_v, sigma_n_v, ...):
    sigma_new_v, eps_p_new, Y_new, alpha_new_v = radial_return_jax(...)

    def stress_func(deps):
        sig, _, _, _ = radial_return_jax(deps, sigma_n_v, ...)
        return sig

    C_tangent = jacfwd(stress_func)(delta_eps_v)
    return sigma_new_v, eps_p_new, Y_new, alpha_new_v, C_tangent
```

This eliminates the need for hand-derived consistent tangents across all hardening models and strain formulations — a significant reduction in error-prone manual derivation.

### Newton Solver with Backtracking Line Search

Pure Newton iteration without step control diverges in the deep plastic regime. A backtracking line search with the Armijo sufficient decrease condition is implemented:

```
Accept step α if:  ‖R(u + α·du)‖ ≤ (1 - c·α) · ‖R(u)‖

Starting from α = 1.0, reduce by factor ρ = 0.5 until condition is met.
```

This stabilises convergence particularly for large load steps with significant plastic flow.

### State Management in Newton Iterations

A critical implementation detail: all history variables (`σ`, `εᵖ`, `Y`, `α`, `u_old`) are **snapshotted at the start of each load step** and restored before every Newton iteration. This ensures the incremental strain `Δε = ε(u_current) - ε(u_step_start)` is always computed consistently:

```python
sigma_converged = sigma_q.x.array.copy()  # snapshot
...
for k in range(num_iterations):
    sigma_q.x.array[:] = sigma_converged  # restore every iteration
    update_history_and_tangents(...)       # recompute from consistent base
```

---

## Paper Validation

The notebook `Paper_Validation_V2.ipynb` validates the stress update algorithms against the analytical solutions provided in Rodríguez-Ferran and Huerta (1997) at a **single Gauss point**, bypassing the full FEM assembly.

Three deformation paths are tested:

| Test | Description | Analytical Solution |
|------|-------------|---------------------|
| Extension & Compression | Uniaxial extension in x, compression in y | `σ_xx = E(t + t²/2)`, `σ_yy = 0` |
| Dilatation | Biaxial extension, no shear | `σ_xx = σ_yy = E·ln(1+t)` |
| Extension & Rotation | Uniaxial + superposed rigid rotation | `σ_xx = Et·cos²(2πt)`, `σ_yy = Et·sin²(2πt)` |

The First Algorithm (Bathe et al.) is first-order accurate in time (slope ~1 on log-log error plots), consistent with the theoretical analysis in the paper. The error decreases as `O(Δt)` with increasing number of increments.

---

## Results

### Small vs Large Strain Comparison

For the uniaxial tension test on a unit cube (hexahedral mesh), the small and large strain formulations agree closely at small deformation levels, then diverge as finite strain effects become significant:

- **Elastic regime**: Both formulations give identical results (the quadratic term in Δε is negligible)
- **Plastic regime**: The push-forward transformation redistributes stress via the deformation gradient, causing the large strain response to differ from small strain — this difference is physically meaningful and grows with accumulated deformation

### Convergence Behavior

Newton convergence rate reflects the quality of the consistent tangent:

- **Elastic steps**: Quadratic convergence (2 iterations to machine precision) — the JAX-computed tangent is exact
- **Elastoplastic steps**: 3–5 iterations with superlinear convergence — the algorithmic tangent accounts for the yield surface curvature
- **Large strain with deep plasticity**: Requires line search for steps near the yield limit; without line search, pure Newton can diverge exponentially

### Hardening Model Comparison

| Model | Behaviour | Key Parameters |
|-------|-----------|----------------|
| Linear isotropic | Unlimited linear hardening | `h` (hardening modulus) |
| Voce (nonlinear isotropic) | Saturating hardening | `Y_inf` (saturation stress), `δ` (saturation rate) |
| Armstrong–Frederick (kinematic) | Bauschinger effect, ratcheting control | `C_af` (kinematic modulus), `γ_af` (recall parameter) |

---

## Installation

### Prerequisites

- Python 3.10+
- [DOLFINx (FEniCSx)](https://fenicsproject.org/) — recommended via conda or Docker
- JAX with 64-bit precision enabled

### Environment Setup

```bash
# Using conda (recommended)
conda create -n fenicsx-jax python=3.10
conda activate fenicsx-jax

# Install FEniCSx
conda install -c conda-forge fenics-dolfinx

# Install JAX (CPU)
pip install jax[cpu]

# Clone the repository
git clone https://github.com/MANAS-P-VINAYAN/Small_to_large_strain_plasticity.git
cd Small_to_large_strain_plasticity
```

### Docker (alternative)

```bash
docker pull dolfinx/dolfinx:stable
docker run -it -v $(pwd):/home/user dolfinx/dolfinx:stable
pip install jax[cpu]
```

---

## Usage

### Running the Main Simulation

Open `main.ipynb` and configure the material parameters at the top:

```python
E     = 210000.0   # Young's modulus (MPa)
nu    = 0.3        # Poisson's ratio
Y0    = 250.0      # Initial yield stress (MPa)
h     = 1000.0     # Hardening modulus (linear isotropic)
Y_inf = 500.0      # Saturation yield stress (Voce model)
delta = 10.0       # Saturation rate (Voce model)
```

Select the strain formulation and hardening model:

```python
u = Newton_Solver_FullTangent(
    ...,
    strain       = 'small',             # or 'large1'
    hardening    = 'linear_isotropic'   # or 'nonlinear_isotropic', 'kinematic'
)
```

### Running the Paper Validation

Open `Paper_Validation_V2.ipynb`. This notebook runs the three benchmark deformation paths at a single quadrature point and compares the computed stress evolution against the analytical solutions from Rodríguez-Ferran and Huerta (1997).

---

## Key Design Decisions

**Why JAX for constitutive updates?**
The consistent algorithmic tangent is critical for Newton convergence but is notoriously error-prone to derive analytically, especially for complex hardening models. `jax.jacfwd` computes the exact derivative of the stress return mapping with respect to strain increment automatically, for all hardening models, in both small and large strain.

**Why `jax.lax.fori_loop` for Newton iterations inside the return mapping?**
Python `for` loops with `break` are not JAX-traceable. `fori_loop` runs a fixed number of iterations but is fully compatible with JIT compilation and `jacfwd`. The convergence behaviour is equivalent to a fixed-iteration Newton loop, which is standard practice in computational plasticity.

**Why snapshot-and-restore in the Newton solver?**
Without restoring `σₙ` and `u_old` at the start of each Newton iteration, the incremental strain `Δε` would accumulate across iterations, leading to double-counting of plastic strain and inconsistent stress states. The snapshot ensures each Newton iteration sees the same reference state.

**Why separate `alpha_q` storage for the back stress?**
Storing the back stress (6-component tensor per quadrature point) separately from the scalar equivalent plastic strain prevents confusion between isotropic and kinematic models and allows both to coexist in the same codebase cleanly.

---

## References

- Rodríguez-Ferran, A. and Huerta, A. (1997). *Comparing Two Algorithms to Add Large Strains to a Small-Strain FE Code.* ASCE Journal of Engineering Mechanics.
- Bathe, K.J., Ramm, E., and Wilson, E.L. (1975). *Finite element formulations for large deformation dynamic analysis.* Int. J. Numer. Methods Engrg., 9, 353–386.
- Pinsky, P.M., Ortiz, M., and Pister, K.S. (1983). *Numerical integration of rate constitutive equations in finite deformation analysis.* Comp. Appl. Mech. Engrg., 40, 137–158.
- Simo, J.C. and Hughes, T.J.R. (1998). *Computational Inelasticity.* Springer.
- Chaboche, J.L. (1986). *Time-independent constitutive theories for cyclic plasticity.* Int. J. Plasticity, 2(2), 149–188.

---

## License

MIT License. See `LICENSE` for details.

---

*Built with [DOLFINx](https://fenicsproject.org/) · [JAX](https://github.com/google/jax) · [basix](https://github.com/FEniCS/basix)*

