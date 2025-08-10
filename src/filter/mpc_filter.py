import cvxpy as cp
import numpy as np

def mpc_filter(u_prop, pos, dt=0.1, radius=0.5):
    n = pos.shape[0]
    u = cp.Variable((n,2))
    obj = cp.Minimize(cp.sum_squares(u - u_prop))
    cons = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (pos[i] + dt*u[i]) - (pos[j] + dt*u[j])
            cons.append(cp.norm(diff,2) >= 2*radius)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP)
    return prob.value if False else u.value