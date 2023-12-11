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

        imp = np.zeros([4, 1])
        iter = 0

        resids = []

        while iter < 16:
            p_vel = J_contact @ M_inv @ imp + foot_vel
            p_vel_n = p_vel[1]
            p_vel_t = p_vel[0]

            # Update normal impulse
            delta_contact_imp_n = contact_mass[1, 1] * np.array(
                [[0], p_vel_n_des - p_vel_n]
            )
            imp += J_contact.T @ delta_contact_imp_n

            # Update tangential impulse (friction)
            delta_contact_imp_t = contact_mass[0, 0] * np.array([0 - p_vel_t, [0]])
            imp += J_contact.T @ delta_contact_imp_t

            # Clip to friction cone
            contact_imp_t, contact_imp_n = J_contact @ imp
            cone = coeff_frict * np.abs(contact_imp_n)
            clipped_contact_imp_t = np.clip(
                contact_imp_t,
                -cone,
                cone,
            )
            delta_contact_imp_t = np.array([[1.0], [0.0]]) * (
                clipped_contact_imp_t - contact_imp_t
            )
            imp += J_contact.T @ delta_contact_imp_t

            p_vel_t, p_vel_n = J_contact @ M_inv @ imp + foot_vel
            n_resid = p_vel_n_des - p_vel_n
            t_resid = 0 - p_vel_t
            t_clipped = np.allclose(np.linalg.norm(clipped_contact_imp_t), cone)

            overall_resid = n_resid + t_resid * t_clipped
            resids.append(overall_resid)

            n_converged = np.linalg.norm(n_resid) < 1e-6
            t_converged = np.linalg.norm(t_resid) < 1e-6

            if n_converged and (t_converged or t_clipped):
                # print("Contact solver converged in ", iter, " iterations")
                break

            iter += 1
        else:
            print(
                f"Warning: contact solver did not converge, best residual index: {np.argmin(resids)}"
            )
            pass

        return M_inv @ imp, True

    return np.zeros_like(qd), False


# Define dynamics functions
def dynamics(q, qd, u, p):
    A = A_func(q, qd, p)
    b = b_func(q, qd, p)
    Q = Q_func(q, qd, u)

    qdd = np.linalg.solve(A, Q - b)

    return qdd
