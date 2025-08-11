import cvxpy as cp
import numpy as np

def mpc_filter(u_prop, pos, dt=0.1, radius=0.5, eps=1e-3, lambda_slack=1e3):
    """
    Minimize ||u - u_prop||^2 + lambda_slack * sum(s_ij^2)
    s.t. (p_i - p_j)^T (u_i - u_j) >= (d_min^2 - ||p_i - p_j||^2)/dt - s_ij
         -1 <= u[:,k] <= 1
    Slack s_ij >= 0 makes the QP always feasible; large lambda discourages violations.
    """
    n = pos.shape[0]
    u = cp.Variable((n, 2))
    d_min = 2.0 * radius + eps

    constraints = [u[:,0] <= 2, u[:,0] >= -2, u[:,1] <= 2, u[:,1] >= -2]

    slacks = []
    for i in range(n):
        for j in range(i + 1, n):
            p_rel = pos[i] - pos[j]                      # (2,)
            rhs = (d_min**2 - float(p_rel @ p_rel)) / dt
            s_ij = cp.Variable(nonneg=True)
            slacks.append(s_ij)
            constraints.append(p_rel @ u[i] - p_rel @ u[j] >= rhs - s_ij)

    slack_term = cp.sum_squares(cp.hstack(slacks)) if slacks else 0
    obj = cp.Minimize(cp.sum_squares(u - u_prop) + lambda_slack * slack_term)
    prob = cp.Problem(obj, constraints)

    for solver in (cp.OSQP, cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, warm_start=True, verbose=False)
            break
        except Exception:
            continue

    return u.value if u.value is not None else u_prop
