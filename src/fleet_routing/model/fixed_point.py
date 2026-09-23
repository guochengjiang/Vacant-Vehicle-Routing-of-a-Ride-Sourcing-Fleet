"""Joint stationary-distribution / matching fixed point (cell 4)."""
import numpy as np
from fleet_routing.model.policy import build_P_matrix, compute_kappa, compute_R

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
    """
    Final joint fixed-point solver
    
    Parameters:
        fixed_m_states: Indices of states forced to m=1 (e.g., boundary states)
        tol_m, tol_mu: Convergence tolerances
        method: 'msa', 'momentum', 'hybrid'
        restart_interval: MSA restart interval (recommended: 20-30)
        momentum: Momentum coefficient (0.8-0.95)
        lr: Learning rate (0.3-0.7)
        phase1_tol: Phase 1 tolerance for the hybrid method
    """
    import time
    
    n = len(lambda_vec)
    m = init_m.copy().astype(np.float64)
    mu = init_mu.copy().astype(np.float64)
    
    # Handle fixed_m_states
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
    
    # Momentum variables
    if method in ['momentum', 'hybrid']:
        v_m = np.zeros_like(m)
        v_mu = np.zeros_like(mu)
        phase1_done = False
    
    for outer in range(max_iter):
        m_old = m.copy()
        mu_old = mu.copy()
        
        # ========================================
        # Step 1: Compute m_target
        # ========================================
        
        if heuristic == None:
            weighted_sum = np.dot(mu, tau + m * omega)
            safe_mu = np.where(mu > 0, mu, 1.0)
            exponent = -alpha * lambda_vec * weighted_sum / (M * safe_mu)
            m_target = 1 - np.exp(exponent)
            m_target = np.where(mu > 0, m_target, 1.0)
        
        ## competition-free
        elif heuristic == 'CF':
            m_target = np.where(lambda_vec > 0, 1.0, 0.0)
        
        # Enforce m=1 for boundary states
        if has_fixed_states:
            m_target[fixed_m_states] = 1.0
        
        # ========================================
        # Step 2: build P
        # ========================================
        P = build_P_matrix(Q, m_target, pi, outgoing_dict, row_idx, col_idx)
        
        # ========================================
        # Step 3: Compute mu_target
        # ========================================
        mu_target = mu @ P
        
        # ========================================
        # Step 4: Compute the actual residual (target - old)
        # ========================================
        if has_fixed_states:
            m_residual = np.max(np.abs(m_target[free_states_mask] - m_old[free_states_mask]))
        else:
            m_residual = np.max(np.abs(m_target - m_old))
        
        mu_residual = 0.5 * np.sum(np.abs(mu_target - mu_old))
        
        # ========================================
        # Step 5: Update m and mu
        # ========================================
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
                        print(f"    Phase 1 completed: iter={outer}, time={time.time()-t0:.1f}s")
                
                a = local_iter / (local_iter + 1)
                m = a * m_target + (1 - a) * m_old
                mu = a * mu_target + (1 - a) * mu_old
        
        # Enforce m=1 for boundary states
        if has_fixed_states:
            m[fixed_m_states] = 1.0
        
        # Normalize mu
        mu = mu / mu.sum()
        
        # Update the MSA counter
        local_iter += 1
        if method in ['msa', 'hybrid'] and restart_interval and local_iter >= restart_interval:
            local_iter = 1
        
        # ========================================
        # Step 6: Check convergence
        # ========================================
        if m_residual < tol_m and mu_residual < tol_mu:
            if verbose:
                kappa_val = compute_kappa(v, c, Q, beta)
                R_pi = compute_R(mu, tau, omega, m, kappa_val, beta)
                print(f"Converged: iter={outer}, time={time.time()-t0:.1f}s")
                print(f"  m_residual={m_residual:.2e}, mu_residual={mu_residual:.2e}, R={R_pi:.2f}")
            break
        
        if verbose and outer % 2000 == 0:
            kappa_val = compute_kappa(v, c, Q, beta)
            R_pi = compute_R(mu, tau, omega, m, kappa_val, beta)
            print(f"  Inner iter={outer}, m_res={m_residual:.2e}, mu_res={mu_residual:.2e}, R={R_pi:.4f}")
            if has_fixed_states:
                print(f"  Mean m for fixed states: {m[fixed_m_states].mean():.4f}")
                print(f"  Total mu for fixed states: {mu[fixed_m_states].sum():.6f}")
    
    else:
        if verbose:
            print(f"Warning: No convergence; maximum iterations reached: {max_iter}")
            print(f"  m_residual={m_residual:.2e}, mu_residual={mu_residual:.2e}")
    
    # ========================================
    # Final verification
    # ========================================
    if verbose:
        P_final = build_P_matrix(Q, m, pi, outgoing_dict, row_idx, col_idx)
        stationarity_error = np.linalg.norm(mu @ P_final - mu)
        print(f"  Stationarity error: {stationarity_error:.2e}")
        
        if has_fixed_states:
            print(f"  Mean m for fixed states: {m[fixed_m_states].mean():.4f}")
            print(f"  Total mu for fixed states: {mu[fixed_m_states].sum():.6f}")
    
    return mu, m
