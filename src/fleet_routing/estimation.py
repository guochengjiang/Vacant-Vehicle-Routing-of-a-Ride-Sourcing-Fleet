"""Estimate alpha from matched/unmatched trajectories (cell 11)."""
import numpy as np

def estimate_alpha_from_trajectories(trajectories, lambda_vec, tau, z_avg,
                                      bounds=(0.01, 5.0)):
    """
    Estimate the matching-function parameter alpha from simulation trajectories.
    
    Maximize LL2:
        LL2 = Σ(unmatched) log(1-m) + Σ(matched) log(m)
        m_s = 1 - exp(-alpha * lambda_s * tau_s / z_s)
    
    Parameters
    ----------
    trajectories : list of (car_id, link_id, matched_bool, time_sec)
                   or list of (link_id, matched_bool)
    lambda_vec : array, passenger arrival rates (per hour)
    tau : array, link travel times (hours)
    z_avg : array, average vacant taxi count per link
    bounds : tuple, search bounds for alpha
    
    Returns
    -------
    alpha_hat : float
    ll2_value : float
    """
    from scipy.optimize import minimize_scalar
    
    if len(trajectories) == 0:
        raise ValueError("Cannot estimate alpha from empty trajectories")

    # Support both 4-tuples and 2-tuples
    if len(trajectories[0]) == 4:
        links = np.array([t[1] for t in trajectories])
        matched = np.array([t[2] for t in trajectories], dtype=float)
    else:
        links = np.array([t[0] for t in trajectories])
        matched = np.array([t[1] for t in trajectories], dtype=float)
    
    # lambda * tau / z
    safe_z = np.maximum(z_avg[links], 1e-10)
    ratio = lambda_vec[links] * tau[links] / safe_z
    
    # Keep only transitions with lambda > 0
    valid = lambda_vec[links] > 0
    ratio_valid = ratio[valid]
    matched_valid = matched[valid]
    
    if len(ratio_valid) == 0:
        print('Warning: no valid transitions')
        return 1.0, 0.0
    
    def neg_LL2(alpha):
        m = 1 - np.exp(-alpha * ratio_valid)
        m = np.clip(m, 1e-10, 1 - 1e-10)
        ll = np.sum(matched_valid * np.log(m) + (1 - matched_valid) * np.log(1 - m))
        return -ll
    
    result = minimize_scalar(neg_LL2, bounds=bounds, method='bounded')
    alpha_hat = result.x
    ll2_value = -result.fun
    
    m_est = 1 - np.exp(-alpha_hat * ratio_valid)
    print(f'Alpha estimation: α̂ = {alpha_hat:.4f}')
    print(f'  LL2 = {ll2_value:.2f}')
    print(f'  Observed match rate: {matched_valid.mean():.4f}')
    print(f'  Estimated m mean (λ>0): {m_est.mean():.4f}')
    print(f'  N transitions: {len(matched_valid)} (λ>0) / {len(matched)} (total)')
    
    return alpha_hat, ll2_value
