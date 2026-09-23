"""Edit experiment settings here, then run one of the run_*.py scripts."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / 'data' / 'shanghai'
OUTPUT_DIR = PROJECT_ROOT / 'results'


@dataclass
class ExperimentConfig:
    M: int = 5000
    # Demand scaling is independent of M. For M=10000 with 5000's demand,
    # change M only and leave demand_reference_fleet=5000.
    demand_reference_fleet: int = 5000
    demand_base_fleet: int = 12000
    # Set a known alpha here OR run run_estimation.py first.
    alpha: float | None = None
    beta: float = 30.0
    heuristic: str | None = None  # None = optimal; 'CF' = competition-free
    fixed_point_method: str = 'msa'
    fixed_point_maxiter: int = 300000
    tol_m: float = 1e-6
    tol_mu: float = 1e-6
    restart_interval: int = 30
    maxiter: int = 1000
    gtol: float = 1e-5
    sim_hours: float = 12.0  # Includes warmup, as in the notebook
    warmup_hours: float = 2.0
    tick_duration: float = 20.0
    n_runs: int = 10
    seed: int | None = 2026
    save_trajectories: bool = True

    @property
    def tag(self):
        mode = 'CF' if self.heuristic == 'CF' else 'optimal'
        return f'M{self.M}_demand{self.demand_reference_fleet}_{mode}'

    @property
    def calibration_file(self):
        return OUTPUT_DIR / f'calibration_M{self.M}_demand{self.demand_reference_fleet}.pkl'

    @property
    def optimization_file(self):
        return OUTPUT_DIR / f'optimization_{self.tag}.pkl'

    def scaled_demand(self, network):
        return network.lambda_vec * (self.demand_reference_fleet / self.demand_base_fleet)

