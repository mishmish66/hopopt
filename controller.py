from physics import foot_J_func

from einops import einsum, rearrange

import numpy as np
import casadi as ca

import sys
import os


def optimize_traj(init_q, init_qd, ref_traj, stance, params, dt=0.005):
    # Unpack params
    m_b, I, m_l, g = params
    # Find the first stance phase
    stance_start = np.min(np.argwhere(stance.squeeze()))
    stance_end = (
        np.min(np.argwhere(~stance[stance_start:].squeeze())) + stance_start + 1
    )
    # Grab just the first stance phases
    stance = np.zeros_like(stance.squeeze())
    stance[stance_start:stance_end] = True
    n_stance_control_times = np.sum(stance)

    # Extract x and ys from ref_traj
    (
        thetas,
        xs,
        ys,
        theta_dotx,
        x_dots,
        y_dots,
    ) = ref_traj.T
    x_ref = rearrange(ref_traj[stance], "t i -> (t i)")
    x_0 = np.concatenate([init_q[0:3], init_qd[0:3]])
    ls = np.zeros_like(xs)
    qs = np.stack([thetas, xs, ys, ls], axis=-1)

    # Construct A_qp and B_qp
    eye = np.eye(3)
    zero = np.zeros((3, 3))
    A = rearrange(
        np.stack(
            [
                [eye, eye * dt],
                [zero, eye],
            ]
        ),
        "br bc i j -> (br i) (bc j)",
        **{"br": 2, "bc": 2},
    )

    m_inv = np.diag([1 / I, 1 / (m_b + m_l), 1 / (m_b + m_l)])
    M_inv = rearrange(
        np.stack(
            [
                [zero, zero],
                [zero, m_inv],
            ]
        ),
        "br bc i j -> (br i) (bc j)",
        **{"br": 2, "bc": 2},
    )

    jacs = np.stack([foot_J_func(q).T[0:3] for q in qs[stance]])

    J = np.concatenate(
        [
            np.broadcast_to(
                np.zeros([3, 2]),
                jacs.shape,
            ),
            jacs,
        ],
        axis=1,
    )

    bs = np.concatenate([einsum(M_inv, J, "i j, t j k -> t i k")]) * dt

    A_qp_stack = np.stack(
        [np.linalg.matrix_power(A, i) for i in range(n_stance_control_times)]
    )
    A_qp = rearrange(A_qp_stack, "p i j -> (p i) j")
    A_qp_pad = np.concatenate([np.zeros([1, 6, 6]), A_qp_stack])

    A_ind_mat = np.clip(
        np.arange(n_stance_control_times, 0, -1)[None, :]
        - np.arange(n_stance_control_times, 0, -1)[:, None],
        0,
        None,
    )

    A_qp_mat = A_qp_pad[A_ind_mat]

    B_qp = einsum(A_qp_mat, bs, "bi bj i j, bj j k -> bi bj i k")
    B_qp = rearrange(B_qp, "bi bj i j -> (bi i) (bj j)")

    # Weight Matrix
    # TODO: Tune this
    L = np.kron(
        np.diag(np.arange(n_stance_control_times)),
        np.diag(np.array([0.0, 1.0, 1.0, 0.0, 250.0, 250.0])),
    )
    K = np.eye(n_stance_control_times * 2) * 1e-2

    H = 2 * (B_qp.T @ L @ B_qp + K)
    g = 2 * B_qp.T @ L @ (A_qp @ x_0 - x_ref)

    # Constraints
    c_single = np.array(
        [
            [0, 1],  # No pulling constraint
            [-1.0, 0.4],  # Friction cone constraint
            [1.0, 0.4],  # Friction cone constraint
        ]
    )
    c_lower_single = np.array(
        [
            [0.05],  # No pulling constraint
            [0],  # Friction cone constraint
            [0],  # Friction cone constraint
        ]
    )

    C = np.kron(np.eye(n_stance_control_times), c_single)
    c_lower = np.tile(c_lower_single, (n_stance_control_times, 1))

    # Silence casadi
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")
    # Define optimization variables
    U = ca.MX.sym("U", n_stance_control_times * 2)

    # Solve QP
    H_ca = ca.DM(H)
    g_ca = ca.DM(g)
    C_ca = ca.DM(C)
    c_lower_ca = ca.DM(c_lower)

    # Cost
    cost = 0.5 * ca.mtimes(U.T, ca.mtimes(H_ca, U)) + ca.mtimes(g_ca.T, U)

    # Solver
    qp = {"x": U, "f": cost, "g": ca.vertcat(C_ca @ U)}
    opts = {"print_time": 0, "printLevel": "none", "verbose": False}
    solver = ca.qpsol("solver", "qpoases", qp, opts)

    # Solve the QP
    res = solver(lbg=c_lower_ca, ubg=ca.inf)
    U_opt = res["x"]

    # Un-silence casadi
    sys.stdout = stdout
    sys.stderr = stderr

    return U_opt
