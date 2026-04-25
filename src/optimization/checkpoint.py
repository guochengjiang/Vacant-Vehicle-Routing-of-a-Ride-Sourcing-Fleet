import os
import pickle
from pathlib import Path


def save_checkpoint(self, filename='checkpoint.pkl', save_dir='results/checkpoints'):

    project_root = Path(__file__).parent.parent.parent
    save_dir = project_root / 'results' / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / filename

    with open(filepath, 'wb') as f:
        pickle.dump({
            'theta_hist': self.theta_hist,
            'R_hist': self.R_hist,
            'state_hist': self.state_hist,
            'mu_last': self.mu_last,
            'm_last': self.m_last
        }, f)
    #print(f"Checkpoint saved to {filepath}")


def load_checkpoint(self, filename='checkpoint.pkl', save_dir='results/checkpoints'):
    filepath = Path(save_dir) / filename

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    self.theta_hist = data['theta_hist']
    self.R_hist = data['R_hist']
    self.state_hist = data.get('state_hist', [])
    self.mu_last = data.get('mu_last', self.init_mu.copy())
    self.m_last = data.get('m_last', self.init_m.copy())

    print(f"Loaded {len(self.theta_hist)} steps from {filepath}")
    print(f"Last theta: {self.theta_hist[-1]}")
    print(f"Last R: {self.R_hist[-1]}")

    return self.theta_hist[-1]