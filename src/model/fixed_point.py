import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse, diags
from ..model.reward import compute_kappa, compute_R
import time

def build_P_matrix(Q, m, Pi, outgoing_dict, row_idx, col_idx):
    n = Q.shape[0]
    P = m[:, None] * Q


    if issparse(Pi):
        Pi_dense = Pi.toarray()
    else:
        Pi_dense = np.asarray(Pi)

    delta = (1 - m[row_idx]) * Pi_dense[row_idx, col_idx]

    P[row_idx, col_idx] += delta

    # delta = (1 - m[row_idx]) * Pi[row_idx, col_idx].A1

    # P[row_idx, col_idx] += delta

    return P




def compute_joint_fixed_point_final(
        lambda_vec, tau, omega, M, alpha, Q, outgoing_dict, pi,
        init_mu, init_m, row_idx, col_idx, v, c, beta,
        fixed_m_states=None,
        tol_m=1e-6, tol_mu=1e-6, max_iter=100000,
        method='hybrid',
        restart_interval=30,
        momentum=0.9,
        lr=0.5,
        phase1_tol=1e-4,
        verbose=True,
        heuristic=None
):



    n = len(lambda_vec)
    m = init_m.copy().astype(np.float64)
    mu = init_mu.copy().astype(np.float64)


    if fixed_m_states is None:
        fixed_m_states = np.array([], dtype=np.int64)
    else:
        fixed_m_states = np.asarray(fixed_m_states, dtype=np.int64)

    has_fixed_states = len(fixed_m_states) > 0

    if has_fixed_states:
        free_states_mask = np.ones(n, dtype=bool)
        free_states_mask[fixed_m_states] = False

    t0 = time.time()
    local_iter = 1


    if method in ['momentum', 'hybrid']:
        v_m = np.zeros_like(m)
        v_mu = np.zeros_like(mu)
        phase1_done = False

    for outer in range(max_iter):
        m_old = m.copy()
        mu_old = mu.copy()


        if heuristic == None:
            weighted_sum = np.dot(mu, tau + m * omega)
            safe_mu = np.where(mu > 0, mu, 1.0)
            exponent = -alpha * lambda_vec * weighted_sum / (M * safe_mu)
            m_target = 1 - np.exp(exponent)
            m_target = np.where(mu > 0, m_target, 1.0)

        ## competition-free
        elif heuristic == 'CF':
            m_target = np.where(lambda_vec > 0, 1.0, 0.0)


        if has_fixed_states:
            m_target[fixed_m_states] = 1.0


        P = build_P_matrix(Q, m_target, pi, outgoing_dict, row_idx, col_idx)


        mu_target = mu @ P


        if has_fixed_states:
            m_residual = np.max(np.abs(m_target[free_states_mask] - m_old[free_states_mask]))
        else:
            m_residual = np.max(np.abs(m_target - m_old))

        mu_residual = 0.5 * np.sum(np.abs(mu_target - mu_old))

        if method == 'msa':
            a = local_iter / (local_iter + 1)
            m = a * m_target + (1 - a) * m_old
            mu = a * mu_target + (1 - a) * mu_old

        elif method == 'momentum':
            v_m = momentum * v_m + lr * (m_target - m)
            v_mu = momentum * v_mu + lr * (mu_target - mu)
            m = m + v_m
            mu = mu + v_mu
            m = np.clip(m, 0, 1)
            mu = np.maximum(mu, 0)

        elif method == 'hybrid':
            if not phase1_done and (m_residual > phase1_tol or mu_residual > phase1_tol):
                # Phase 1: Momentum
                v_m = momentum * v_m + lr * (m_target - m)
                v_mu = momentum * v_mu + lr * (mu_target - mu)
                m = m + v_m
                mu = mu + v_mu
                m = np.clip(m, 0, 1)
                mu = np.maximum(mu, 0)
            else:
                # Phase 2: MSA
                if not phase1_done:
                    phase1_done = True
                    local_iter = 1
                    if verbose:
                        print(f"    Phase 1 完成: iter={outer}, time={time.time() - t0:.1f}s")

                a = local_iter / (local_iter + 1)
                m = a * m_target + (1 - a) * m_old
                mu = a * mu_target + (1 - a) * mu_old


        if has_fixed_states:
            m[fixed_m_states] = 1.0

        mu = mu / mu.sum()

        local_iter += 1
        if method in ['msa', 'hybrid'] and restart_interval and local_iter >= restart_interval:
            local_iter = 1


        if m_residual < tol_m and mu_residual < tol_mu:
            if verbose:
                kappa_val = compute_kappa(v, c, Q, beta)
                R_pi = compute_R(mu, tau, omega, m, kappa_val, beta)
                print(f"Converged: iter={outer}, time={time.time() - t0:.1f}s")
                print(f"  m_residual={m_residual:.2e}, mu_residual={mu_residual:.2e}, R={R_pi:.2f}")
            break

        if verbose and outer % 1000 == 0:
            kappa_val = compute_kappa(v, c, Q, beta)
            R_pi = compute_R(mu, tau, omega, m, kappa_val, beta)
            print(f"iter={outer}, m_res={m_residual:.2e}, mu_res={mu_residual:.2e}, R={R_pi:.2f}")
            if has_fixed_states:
                print(f"  fixed point m mean: {m[fixed_m_states].mean():.4f}")
                print(f"  fixed point mu sum: {mu[fixed_m_states].sum():.6f}")

    else:
        if verbose:
            print(f"Warning: not converged, reach the maximum iterations {max_iter}")
            print(f"  m_residual={m_residual:.2e}, mu_residual={mu_residual:.2e}")

    # ========================================
    # 最终验证
    # ========================================
    if verbose:
        P_final = build_P_matrix(Q, m, pi, outgoing_dict, row_idx, col_idx)
        stationarity_error = np.linalg.norm(mu @ P_final - mu)
        print(f"  Stationary error: {stationarity_error:.2e}")

        if has_fixed_states:
            print(f"  fixed point m mean: {m[fixed_m_states].mean():.4f}")
            print(f"  fixed point mu sum: {mu[fixed_m_states].sum():.6f}")

    return mu, m