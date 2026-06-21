# ═══════════════════════════════════════════════════════════════════════════
#  GENERIC TEST DRIVER  —  full set of tests, including extension+rotation
#
#  Test types:
#    Traction-driven:
#      'uniaxial', 'compression', 'biaxial', 'shear_traction'
#    Displacement-driven (free lateral faces):
#      'uniaxial_disp', 'compression_disp', 'biaxial_disp', 'shear'
#    Paper-matching (Rodriguez-Ferran & Huerta 1996, all dofs prescribed):
#      'paper_ext_comp'    — Extension+Compression  (§4.1, Fig. 3)
#      'paper_dilatation'  — Biaxial expansion      (§4.2, Fig. 6)
#      'paper_uniaxial'    — Uniaxial, y/z clamped  (§4.3, Eq. 36)
#      'paper_ext_rot'     — Extension+Rotation     (§4.7, Fig. 9, Eq. 45)
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
from dolfinx import fem, mesh as msh
from mpi4py import MPI


def get_face_dofs(V, domain):
    tdim = domain.topology.dim
    fdim = tdim - 1
    def on_face(coord, val):
        return lambda x: np.isclose(x[coord], val)
    faces = {
        'x0': on_face(0, 0.0), 'x1': on_face(0, 1.0),
        'y0': on_face(1, 0.0), 'y1': on_face(1, 1.0),
        'z0': on_face(2, 0.0), 'z1': on_face(2, 1.0),
    }
    out = {}
    for name, locator in faces.items():
        facets = msh.locate_entities_boundary(domain, fdim, locator)
        out[name] = fem.locate_dofs_topological(V, fdim, facets)
    return out


def get_corner_dof(V, domain, corner):
    coords = V.tabulate_dof_coordinates()
    target = np.array(corner, dtype=float)
    distances = np.linalg.norm(coords - target, axis=1)
    return int(np.argmin(distances))


def get_all_dofs_with_coords(V):
    """Return (dof_indices, coords) for ALL dofs in the function space."""
    coords = V.tabulate_dof_coordinates()
    n_dofs = coords.shape[0]
    return np.arange(n_dofs), coords


def build_test_setup(test_type, V, domain, P_max, num_steps):
    fd = get_face_dofs(V, domain)
    roller_bcs = [(fd['x0'], 0), (fd['y0'], 1), (fd['z0'], 2)]
    norm_loads = np.array([0.0] + [(n + 1) / num_steps for n in range(num_steps)])

    # ════════════════════════════════════════════════════════════════════
    #  Traction-driven
    # ════════════════════════════════════════════════════════════════════
    if test_type == 'uniaxial':
        load_path = []
        for n in range(num_steps):
            t_cum = P_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : t_cum,
                'traction_vec_override' : np.array([t_cum, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : None,
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Uniaxial Tension (traction)',
                    control='traction', face_dofs=fd)

    elif test_type == 'compression':
        load_path = []
        for n in range(num_steps):
            t_cum = -P_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : t_cum,
                'traction_vec_override' : np.array([t_cum, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : None,
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Uniaxial Compression (traction)',
                    control='traction', face_dofs=fd)

    elif test_type == 'biaxial':
        load_path = []
        for n in range(num_steps):
            t_cum = P_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : t_cum,
                'traction_vec_override' : np.array([t_cum, 0.0, 0.0]),
                'extra_tractions'       : [(fd['y1'], np.array([0.0, t_cum, 0.0]))],
                'prescribed_values'     : None,
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Biaxial Tension (traction)',
                    control='traction', face_dofs=fd)

    elif test_type == 'shear_traction':
        c000 = get_corner_dof(V, domain, (0.0, 0.0, 0.0))
        c100 = get_corner_dof(V, domain, (1.0, 0.0, 0.0))
        c010 = get_corner_dof(V, domain, (0.0, 1.0, 0.0))
        anchor_bcs = [
            (np.array([c000]), 0), (np.array([c000]), 1), (np.array([c000]), 2),
            (np.array([c100]), 1), (np.array([c100]), 2),
            (np.array([c010]), 2),
        ]
        load_path = []
        for n in range(num_steps):
            tau = P_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : tau,
                'traction_vec_override' : np.array([tau, 0.0, 0.0]),
                'extra_tractions'       : [(fd['x1'], np.array([0.0, 0.0, tau]))],
                'prescribed_values'     : None,
            })
        return dict(bc_per_component=anchor_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['z1'],
                    label='Simple Shear (traction)',
                    control='traction', face_dofs=fd)

    # ════════════════════════════════════════════════════════════════════
    #  Displacement-driven (free lateral faces)
    # ════════════════════════════════════════════════════════════════════
    elif test_type == 'uniaxial_disp':
        eps_max = P_max
        load_path = []
        for n in range(num_steps):
            eps = eps_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : eps,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : [(fd['x1'], 0, eps)],
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Uniaxial Tension (displacement, free lateral)',
                    control='displacement', face_dofs=fd)

    elif test_type == 'compression_disp':
        eps_max = P_max
        load_path = []
        for n in range(num_steps):
            eps = -eps_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : eps,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : [(fd['x1'], 0, eps)],
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Uniaxial Compression (displacement, free lateral)',
                    control='displacement', face_dofs=fd)

    elif test_type == 'biaxial_disp':
        eps_max = P_max
        load_path = []
        for n in range(num_steps):
            eps = eps_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : eps,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : [(fd['x1'], 0, eps), (fd['y1'], 1, eps)],
            })
        return dict(bc_per_component=roller_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Biaxial Tension (displacement, free z)',
                    control='displacement', face_dofs=fd)

    elif test_type == 'shear':
        gamma_max = P_max
        load_path = []
        for n in range(num_steps):
            gamma = gamma_max * (n + 1) / num_steps
            load_path.append({
                'cumulative_traction'   : gamma,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : [
                    (fd['z1'], 0, gamma),
                    (fd['z1'], 1, 0.0),
                    (fd['z1'], 2, 0.0),
                ],
            })
        shear_bcs = [(fd['z0'], 0), (fd['z0'], 1), (fd['z0'], 2)]
        return dict(bc_per_component=shear_bcs, load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Simple Shear (displacement)',
                    control='displacement', face_dofs=fd)

    # ════════════════════════════════════════════════════════════════════
    #  Paper-matching tests (all dofs prescribed)
    # ════════════════════════════════════════════════════════════════════

    elif test_type == 'paper_ext_comp':
        # x = X(1+t), y = Y/(1+t), z = Z   (paper §4.1, Eq. 39)
        t_max = P_max
        load_path = []
        for n in range(num_steps):
            t = t_max * (n + 1) / num_steps
            uy_at_y1 = -t / (1.0 + t)
            prescribed = [
                (fd['x0'], 0, 0.0), (fd['x1'], 0, t),
                (fd['y0'], 1, 0.0), (fd['y1'], 1, uy_at_y1),
                (fd['z0'], 2, 0.0), (fd['z1'], 2, 0.0),
            ]
            load_path.append({
                'cumulative_traction'   : t,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : prescribed,
            })
        return dict(bc_per_component=[], load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Extension + Compression',
                    control='displacement', face_dofs=fd)

    elif test_type == 'paper_dilatation':
        # x = X(1+t), y = Y(1+t), z = Z   (paper §4.2)
        t_max = P_max
        load_path = []
        for n in range(num_steps):
            t = t_max * (n + 1) / num_steps
            prescribed = [
                (fd['x0'], 0, 0.0), (fd['x1'], 0, t),
                (fd['y0'], 1, 0.0), (fd['y1'], 1, t),
                (fd['z0'], 2, 0.0), (fd['z1'], 2, 0.0),
            ]
            load_path.append({
                'cumulative_traction'   : t,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : prescribed,
            })
        return dict(bc_per_component=[], load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Dilatation',
                    control='displacement', face_dofs=fd)

    elif test_type == 'paper_uniaxial':
        # x = X(1+t), y = Y, z = Z   (paper §4.3, Eq. 36; y, z clamped)
        t_max = P_max
        load_path = []
        for n in range(num_steps):
            t = t_max * (n + 1) / num_steps
            prescribed = [
                (fd['x0'], 0, 0.0), (fd['x1'], 0, t),
                (fd['y0'], 1, 0.0), (fd['y1'], 1, 0.0),
                (fd['z0'], 2, 0.0), (fd['z1'], 2, 0.0),
            ]
            load_path.append({
                'cumulative_traction'   : t,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : prescribed,
            })
        return dict(bc_per_component=[], load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Uniaxial extension (y,z clamped)',
                    control='displacement', face_dofs=fd)

    elif test_type == 'paper_ext_rot':
        # ── Paper §4.7 (Eq. 45, Fig. 9): Extension + Rotation ────────────
        #
        #    x(t) = X(1+t) cos(2πt) − Y sin(2πt)
        #    y(t) = X(1+t) sin(2πt) + Y cos(2πt)
        #    z(t) = Z
        #
        # Each node gets its OWN displacement values based on its (X,Y,Z)
        # reference coordinates. We prescribe ALL three components on
        # EVERY node (full kinematic control).
        #
        # Analytical solution (paper Eq. 47, in spatial frame):
        #    σ_xx(t) = E·t·cos²(2πt)
        #    σ_xy(t) = E·t·sin(2πt)·cos(2πt)
        #    σ_yy(t) = E·t·sin²(2πt)
        #
        # NOTE: this test exercises incremental objectivity — the rigid
        # rotation should produce no spurious stress, only the extension
        # part should stress the material.

        all_dofs, all_coords = get_all_dofs_with_coords(V)

        t_max = P_max
        load_path = []
        for n in range(num_steps):
            t = t_max * (n + 1) / num_steps
            cos = np.cos(2*np.pi*t)
            sin = np.sin(2*np.pi*t)

            # For EACH dof, compute required (u_x, u_y, u_z)
            prescribed = []
            for dof, X in zip(all_dofs, all_coords):
                # imposed motion
                x_new = X[0]*(1+t)*cos - X[1]*sin
                y_new = X[0]*(1+t)*sin + X[1]*cos
                z_new = X[2]
                # displacement = new position − reference
                u_x = x_new - X[0]
                u_y = y_new - X[1]
                u_z = z_new - X[2]
                # prescribe each component for this single dof
                prescribed.append((np.array([dof]), 0, float(u_x)))
                prescribed.append((np.array([dof]), 1, float(u_y)))
                prescribed.append((np.array([dof]), 2, float(u_z)))

            load_path.append({
                'cumulative_traction'   : t,
                'traction_vec_override' : np.array([0.0, 0.0, 0.0]),
                'extra_tractions'       : None,
                'prescribed_values'     : prescribed,
            })

        return dict(bc_per_component=[], load_path=load_path,
                    norm_loads=norm_loads, primary_face_dofs=fd['x1'],
                    label='Extension + Rotation (paper §4.7)',
                    control='displacement', face_dofs=fd)

    else:
        raise ValueError(
            f"Unknown test_type: {test_type!r}.\n"
            f"  Traction-driven  : 'uniaxial', 'compression', 'biaxial', 'shear_traction'\n"
            f"  Displacement     : 'uniaxial_disp', 'compression_disp', 'biaxial_disp', 'shear'\n"
            f"  Paper-matching   : 'paper_ext_comp', 'paper_dilatation', 'paper_uniaxial',\n"
            f"                     'paper_ext_rot'"
        )