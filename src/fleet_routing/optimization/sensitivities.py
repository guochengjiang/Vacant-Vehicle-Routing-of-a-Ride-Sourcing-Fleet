"""Dense analytical sensitivity solver (cell 3; equations preserved)."""
import numpy as np

def solve_sensitivities_direct(
    lambda_vec, tau, omega, M, alpha,
    Q, pi, mu, m,
    x_r, outgoing_mask,
    outgoing_dict
):
    """
    Direct solver (corrected version)
    """
    from scipy.sparse import issparse
    
    n = len(lambda_vec)
    
    # Convert to dense
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
    
    # Precompute
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
    
    # Construct matrix L: dm = L @ dmu
    A_mmu = np.outer(c / mu_safe, t) - np.diag(c * w / mu_safe**2)
    A_mm = np.outer(a, v)
    L = np.linalg.solve(np.eye(n) - A_mm, A_mmu)
    
    # Construct G_matrix: G(dmu) = G_matrix @ dmu
    # G[u] = sum_s mu[s] * Delta[s,u] * dm[s] = sum_s mu[s] * Delta[s,u] * (L @ dmu)[s]
    # G[u] = sum_s sum_j mu[s] * Delta[s,u] * L[s,j] * dmu[j]
    # G_matrix[u,j] = sum_s mu[s] * Delta[s,u] * L[s,j]
    G_matrix = (mu[:, np.newaxis] * Delta).T @ L  # shape (n, n)
    
    # Equation: dmu - dmu @ P - G(dmu) = f0
    # Equivalently: dmu - P^T @ dmu - G_matrix @ dmu = f0
    # Equivalently: (I - P^T - G_matrix) @ dmu = f0
    
    A_full = np.eye(n) - P.T - G_matrix  # ← Correction: use G_matrix without transposing it
    
    # Add the constraint sum(dmu) = 0
    A_full[0, :] = 1.0
    f0_constrained = f0.copy()
    f0_constrained[0] = 0.0
    
    # Solve directly
    dmu = np.linalg.solve(A_full, f0_constrained)
    dm = L @ dmu
    
    return dmu, dm, {'sum_dmu': np.sum(dmu)}
