"""Manual warm-start snapshots. These do not resume BFGS's inverse Hessian."""
from pathlib import Path
import numpy as np
from fleet_routing.results_io import save_result, load_result

from fleet_routing.config import OUTPUT_DIR
DEFAULT_DIR = OUTPUT_DIR / 'checkpoints'


def save_checkpoint(optimizer, filename='checkpoint.pkl', save_dir=DEFAULT_DIR):
    return save_result({
        'theta_hist': optimizer.theta_hist, 'R_hist': optimizer.R_hist,
        'state_hist': optimizer.state_hist,
        'mu_last': optimizer.mu_last, 'm_last': optimizer.m_last,
    }, Path(save_dir) / filename)


def load_checkpoint(optimizer, filename='checkpoint.pkl', save_dir=DEFAULT_DIR):
    data = load_result(Path(save_dir) / filename)
    optimizer.theta_hist = data['theta_hist']
    optimizer.R_hist = data['R_hist']
    optimizer.state_hist = data.get('state_hist', [])
    optimizer.mu_last = data.get('mu_last', optimizer.init_mu.copy())
    optimizer.m_last = data.get('m_last', optimizer.init_m.copy())
    optimizer.clear_cache()
    if not optimizer.theta_hist:
        raise ValueError('Checkpoint contains no theta; there is no restart point')
    return np.asarray(optimizer.theta_hist[-1]).copy()
