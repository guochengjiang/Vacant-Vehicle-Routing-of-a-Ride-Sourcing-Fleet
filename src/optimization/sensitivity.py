from scipy.sparse import issparse
import numpy as np

def solve_sensitivities_direct(
        lambda_vec, tau, omega, M, alpha,
        Q, pi, mu, m,
        x_r, outgoing_mask,
        outgoing_dict
):



    n = len(lambda_vec)


    if issparse(pi):
        pi = pi.toarray()
    if issparse(Q):
        Q = Q.toarray()

    pi = np.asarray(pi, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    x_r = np.asarray(x_r, dtype=np.float64)
    outgoing_mask = np.asarray(outgoing_mask, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).flatten()
    m = np.asarray(m, dtype=np.float64).flatten()
    tau = np.asarray(tau, dtype=np.float64).flatten()
    omega = np.asarray(omega, dtype=np.float64).flatten()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float64).flatten()

    eps = 1e-12
    mu_safe = np.maximum(mu, eps)


    t = tau + m * omega
    w = np.dot(mu, t)
    c = (1 - m) * alpha * lambda_vec / M

    P = (1 - m)[:, np.newaxis] * pi * outgoing_mask + m[:, np.newaxis] * Q

    weighted_x = np.sum(pi * x_r * outgoing_mask, axis=1)
    dpi_dtheta = pi * (x_r - weighted_x[:, np.newaxis]) * outgoing_mask

    f0 = np.dot(mu * (1 - m), dpi_dtheta)

    Delta = Q - pi * outgoing_mask

    a = c / mu_safe
    v = mu * omega


    A_mmu = np.outer(c / mu_safe, t) - np.diag(c * w / mu_safe ** 2)
    A_mm = np.outer(a, v)
    L = np.linalg.solve(np.eye(n) - A_mm, A_mmu)


    G_matrix = (mu[:, np.newaxis] * Delta).T @ L  # shape (n, n)

    A_full = np.eye(n) - P.T - G_matrix


    A_full[0, :] = 1.0
    f0_constrained = f0.copy()
    f0_constrained[0] = 0.0


    dmu = np.linalg.solve(A_full, f0_constrained)
    dm = L @ dmu

    return dmu, dm, {'sum_dmu': np.sum(dmu)}