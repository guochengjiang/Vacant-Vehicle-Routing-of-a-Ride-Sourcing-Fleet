"""Link-based policies, transition matrix, and average reward (notebook cells 5, 9)."""
import numpy as np
from scipy.sparse import coo_matrix, issparse

def extract_state_action_pairs(outgoing_dict):
    row_idx, col_idx = [], []
    for s, actions in outgoing_dict.items():
        row_idx.extend([s] * len(actions))
        col_idx.extend(actions)
    return np.array(row_idx), np.array(col_idx)

def compute_softmax_policy_by_action_features(action_features, theta, n_links, row_idx, col_idx):

    vals = sum(theta[i] * f[col_idx] for i, f in enumerate(action_features))


    sort_idx = np.argsort(row_idx)
    row_idx_sorted = row_idx[sort_idx]
    col_idx_sorted = col_idx[sort_idx]
    vals_sorted = vals[sort_idx]


    pi_data = np.zeros_like(vals_sorted)
    i = 0
    while i < len(row_idx_sorted):
        s = row_idx_sorted[i]
        j = i
        while j < len(row_idx_sorted) and row_idx_sorted[j] == s:
            j += 1
        scores = vals_sorted[i:j]
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        probs = exp_scores / np.sum(exp_scores)
        pi_data[i:j] = probs
        i = j


    pi_sparse = coo_matrix((pi_data, (row_idx_sorted, col_idx_sorted)), shape=(n_links, n_links)).tocsr()
    return pi_sparse


def build_P_matrix(Q, m, Pi, outgoing_dict, row_idx, col_idx):

    n = Q.shape[0]
    P = m[:, None] * Q  # baseline: m_s * q_su  correct!
    
    #Pi_dense = Pi.toarray()
    #delta = (1 - m[row_idx]) * Pi_dense[row_idx, col_idx]
    # Convert to a dense matrix
    if issparse(Pi):
        Pi_dense = Pi.toarray()
    else:
        Pi_dense = np.asarray(Pi)
        
    delta = (1 - m[row_idx]) * Pi_dense[row_idx, col_idx]
    
    P[row_idx, col_idx] += delta
    
    #delta = (1 - m[row_idx]) * Pi[row_idx, col_idx].A1  # .A1 makes it 1D

    #P[row_idx, col_idx] += delta


    return P



def compute_kappa(v, c, q, beta):
    return np.sum((v - beta * c) * q, axis=1)

def compute_R(mu, tau, omega, m, kappa, beta):
    num = np.sum(mu * (-beta * tau + kappa * m))
    denom = np.sum(mu * (tau + omega * m))
    return num / denom if denom > 0 else -np.inf


def make_uniform_policy(outgoing_dict, n_links):
    """Random Walk: uniform over actions"""
    pi = np.zeros((n_links, n_links))
    for s, actions in outgoing_dict.items():
        prob = 1.0 / len(actions)
        for a in actions:
            pi[s, a] = prob
    return pi


def make_hotspot_policy(outgoing_dict, n_links, lambda_vec, temperature=1.0):
    """Hotspot: prefer high demand links"""
    pi = np.zeros((n_links, n_links))
    for s, actions in outgoing_dict.items():
        weights = np.array([lambda_vec[a] for a in actions])
        weights = weights - weights.max()
        weights = np.exp(temperature * weights)
        weights = weights / weights.sum()
        for i, a in enumerate(actions):
            pi[s, a] = weights[i]
    return pi
