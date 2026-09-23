"""Optional Bellman feature experiment from cell 82; not used in the main pipeline.
Preserves the original iteration, including its return-before-final-update behavior
and hardcoded running cost 30. beta here is a DISCOUNT rate, not operating cost.
"""
import numpy as np

def evaluate_expected_fare_and_cost_matrix(
    m, pi, p, r, r_cost, tau, tau_sd, beta,
    max_iter=5000, tol=1e-6
):
    """
    Fully vectorized policy evaluation to compute expected fare and cost for each link.

    Parameters:
    - m: (n_links,) array, match probability
    - pi: (n_links, n_links) array, policy π(a|s)
    - p: (n_links, n_links) array, destination prob p(d|s)
    - r: (n_links, n_links) array, fare from s to d
    - r_cost: (n_links, n_links) array, cost from s to d
    - tau: (n_links,) array, travel time for each link
    - tau_sd: (n_links, n_links) array, shortest time from s to d
    - beta: float, discount log rate

    Returns:
    - f: (n_links,) expected fare
    - c: (n_links,) expected cost
    """
    n_links = len(m)
    f = np.zeros(n_links)
    c = np.zeros(n_links)

    discount_sd = np.exp(-beta * tau_sd)      # shape = (n_links, n_links)
    discount_links = np.exp(-beta * tau)      # shape = (n_links,)

    for it in range(max_iter):
        # Matched terms
        f_matched = np.sum(p * (r + discount_sd * f[np.newaxis, :]), axis=1)   # (n_links,)
        c_matched = np.sum(p * (r_cost + discount_sd * c[np.newaxis, :]), axis=1)

        # Unmatched terms
        f_unmatched = pi @ (discount_links * f)       # (n_links,)
        c_unmatched = pi @ (discount_links * c) # (n_links,)
        
        # Combine matched and unmatched parts
        f_new = m * f_matched + (1 - m) * f_unmatched
        c_new = 30*tau + m * c_matched + (1 - m) * c_unmatched

        # Convergence check
        delta_f = np.max(np.abs(f_new - f))
        delta_c = np.max(np.abs(c_new - c))
        print(f"Ite {it}: Δf = {delta_f:.4e}, Δc = {delta_c:.4e}")

        if delta_f < tol and delta_c < tol:
            break

        f, c = f_new, c_new

    return f, c
