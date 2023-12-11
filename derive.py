import dill
from sympy import *
import numpy as np


# Define state variables
theta, x, y, l = symbols("theta x y l")
q = Matrix([theta, x, y, l])
theta_dot, x_dot, y_dot, l_dot = symbols("theta_dot x_dot y_dot l_dot")
qd = Matrix([theta_dot, x_dot, y_dot, l_dot])
theta_ddot, x_ddot, y_ddot, l_ddot = symbols("theta_ddot x_ddot y_ddot l_ddot")
qdd = Matrix([theta_ddot, x_ddot, y_ddot, l_ddot])

# Define control variables
tau, f = symbols("tau f")
u = Matrix([tau, f])

# Define system parameters
m_b, I, m_l, g = symbols("m_b I m_l, g")
p = Matrix([m_b, I, m_l, g])

# Defin POIs
r_1 = Matrix([x, y])  # COM of body
r_2 = Matrix([x + (l + 0.5) * cos(theta), y + (l + 0.5) * sin(theta)])  # COM of leg
r_3 = Matrix([x + (l + 1.0) * cos(theta), y + (l + 1.0) * sin(theta)])  # foot

g_hat = Matrix([0, 1])  # gravity direction vector


# Define utility functions
def ddt(expr):
    # Take the gradient of expr with respect to q
    # Then multiply by qd
    jacd = expr.jacobian(q) * qd + expr.jacobian(qd) * qdd
    return simplify(jacd)


Q = q.jacobian(Matrix([theta, l])) * u


# Define the Lagrangian
T = (
    1 / 2 * ddt(r_1).T * m_b * ddt(r_1)
    + 1 / 2 * ddt(r_2).T * m_l * ddt(r_2)
    + 1 / 2 * I * ddt(Matrix([theta])) ** 2
)
V = m_b * g * g_hat.T * r_1 + m_l * g * g_hat.T * r_2

L = T - V

# Use the Euler-Lagrange equation to derive the equations of motion

u_exp = ddt(L.jacobian(qd)) - L.jacobian(q).T

A = u_exp.jacobian(qdd)  # Mass matrix
b = u_exp - A * qdd  # Coriolis and gravity terms

A = simplify(A)
b = simplify(b)

# Convert to numpy functions
A_func = lambdify((q, qd, p), A, "numpy")
b_func = lambdify((q, qd, p), b, "numpy")
Q_func = lambdify((q, qd, u), Q, "numpy")
r_3_func = lambdify((q,), r_3, "numpy")
r_3_dot_func = lambdify((q, qd), ddt(r_3), "numpy")
foot_J_func = lambdify((q,), r_3.jacobian(q), "numpy")

# Save the functions
dill.dump(A_func, open("funcs/A_func.pkl", "wb"))
dill.dump(b_func, open("funcs/b_func.pkl", "wb"))
dill.dump(Q_func, open("funcs/Q_func.pkl", "wb"))
dill.dump(r_3_func, open("funcs/foot_pos_func.pkl", "wb"))
dill.dump(r_3_dot_func, open("funcs/foot_vel_func.pkl", "wb"))
dill.dump(foot_J_func, open("funcs/foot_J_func.pkl", "wb"))


# Define the dynamics
def dynamics(q, qd, u, p):
    return np.linalg.solve(A_func(q, qd, p), b_func(q, qd, p) - Q)
