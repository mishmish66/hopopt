import numpy as np
import dill

from einops import einsum

A_func = dill.load(open("funcs/A_func.pkl", "rb"))  # Mass matrix
b_func = dill.load(open("funcs/b_func.pkl", "rb"))  # Coriolis and gravity terms
Q_func = dill.load(open("funcs/Q_func.pkl", "rb"))  # Generalized forces
foot_pos_func = dill.load(open("funcs/foot_pos_func.pkl", "rb"))  # Foot position
foot_vel_func = dill.load(open("funcs/foot_vel_func.pkl", "rb"))  # Foot velocity
foot_J_func = dill.load(open("funcs/foot_J_func.pkl", "rb"))  # Foot Jacobian


def contact(q, qd, p, coeff_rest=0.0, coeff_frict=0.6):
    # Check for contact
    foot_pos = foot_pos_func(q)
    foot_vel = foot_vel_func(q, qd)

    if foot_pos[1] < 0 and foot_vel[1] < 0:
        # Contact

        J_contact = foot_J_func(q)
        M = A_func(q, qd, p)
        M_inv = np.linalg.inv(M)
        contact_mass_inv = einsum(
            J_contact,
            M_inv,
            J_contact.T,
            "i j, j k, k l -> i l",
        )
        contact_mass = np.linalg.inv(contact_mass_inv)

        # Solve for the impulse

        m_vel_t, m_vel_n = foot_vel.squeeze()
        p_vel_n_des = -coeff_rest * m_vel_n

        imp = np.zeros([2, 1])
        iter = 0

        resids = []

        while iter < 16:
            p_vel = contact_mass_inv @ imp + foot_vel
            p_vel_n = p_vel[1]
            p_vel_t = p_vel[0]

            # Update normal impulse
            n_resid = contact_mass @ np.array([[0], p_vel_n_des - p_vel_n])
            imp += n_resid / 4

            # Update tangential impulse (friction)
            t_resid = contact_mass @ np.array([0 - p_vel_t, [0]])
            imp += t_resid / 4

            # Clip to friction cone
            cone = coeff_frict * np.abs(imp[1])
            imp[0] = np.clip(imp[0], -cone, cone)
            t_clipped = np.linalg.norm(imp[0]) == cone

            overall_resid = n_resid + t_resid * t_clipped
            resids.append(overall_resid)

            n_converged = np.linalg.norm(n_resid) < 1e-6
            t_converged = np.linalg.norm(t_resid) < 1e-6

            if n_converged and (t_converged or t_clipped):
                # print("Contact solver converged in ", iter, " iterations")
                break

            iter += 1
        else:
            # print(
            #     f"Warning: contact solver did not converge, best residual index: {np.argmin(resids)}"
            # )
            pass

        delta_rd_cont = contact_mass_inv @ imp
        return J_contact.T @ delta_rd_cont, True

    return np.zeros_like(qd), False


# Define dynamics functions
def dynamics(q, qd, u, p):
    A = A_func(q, qd, p)
    b = b_func(q, qd, p)
    Q = Q_func(q, qd, u)

    qdd = np.linalg.solve(A, Q - b)

    return qdd
