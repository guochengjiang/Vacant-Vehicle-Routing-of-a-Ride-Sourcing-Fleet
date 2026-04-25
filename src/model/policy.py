import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse, diags

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