"""Read the original Input filenames; no data is loaded at import time."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class NetworkData:
    c: np.ndarray
    v: np.ndarray
    Q: np.ndarray
    lambda_vec: np.ndarray
    tau: np.ndarray
    omega: np.ndarray
    features: np.ndarray
    init_dist: np.ndarray
    outgoing_links: dict

    @property
    def n_links(self):
        return len(self.lambda_vec)

    def outgoing_mask(self):
        mask = np.zeros((self.n_links, self.n_links), dtype=float)
        for s, actions in self.outgoing_links.items():
            mask[s, actions] = 1.0
        return mask


def load_network(input_dir):
    input_dir = Path(input_dir)
    filenames = {k: f'{k}_final.npy' for k in
                 ('c', 'v', 'Q', 'lambda_vec', 'tau', 'omega', 'init_dist')}
    filenames['features'] = 'feature_profit_final.npy'
    required = list(filenames.values()) + ['outgoing_links_final.pkl']
    missing = [f for f in required if not (input_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing files in {input_dir}:\n' + '\n'.join(missing))
    arrays = {k: np.load(input_dir / f, allow_pickle=False) for k, f in filenames.items()}
    arrays['features'] = np.atleast_2d(arrays['features'])
    data = NetworkData(**arrays, outgoing_links=pd.read_pickle(input_dir / 'outgoing_links_final.pkl'))
    n = data.n_links
    for k in ('lambda_vec', 'tau', 'omega', 'init_dist'):
        x = getattr(data, k)
        if x.shape != (n,) or not np.all(np.isfinite(x)) or np.any(x < 0):
            raise ValueError(f'{k} must be a finite nonnegative ({n},) vector')
    for k in ('Q', 'c', 'v'):
        x = getattr(data, k)
        if x.shape != (n, n) or not np.all(np.isfinite(x)):
            raise ValueError(f'{k} must be a finite ({n}, {n}) matrix')
    if np.any(data.Q < 0) or np.any(data.c < 0):
        raise ValueError('Q and c must be nonnegative')
    if data.features.shape[1] != n or not np.all(np.isfinite(data.features)):
        raise ValueError('features must have shape (n_features, n_links) and be finite')
    if np.any(data.tau <= 0) or not np.isclose(data.init_dist.sum(), 1):
        raise ValueError('tau must be positive and init_dist must sum to 1')
    if set(data.outgoing_links) != set(range(n)):
        raise ValueError('outgoing_links keys must cover all link indices 0..n-1')
    for s, actions in data.outgoing_links.items():
        if len(actions) == 0 or any(not isinstance(a, (int, np.integer)) or a < 0 or a >= n for a in actions):
            raise ValueError(f'Invalid/empty action set at link {s}; explicitly resolve boundary states')
        if len(set(actions)) != len(actions):
            raise ValueError(f'Duplicate outgoing actions at link {s}')
    if not np.allclose(data.Q.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Q rows must sum to 1; inspect boundary rows before running')
    return data
