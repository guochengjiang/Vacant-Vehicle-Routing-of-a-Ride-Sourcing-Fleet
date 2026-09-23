"""Original small-network experiment settings; no feature normalization."""
from fleet_routing.config import OUTPUT_DIR as RESULTS_ROOT

NETWORK_SEED = 2025
DEMAND_SEED = 2025
M = 1000
ALPHA = 0.8
TOL_M = 1e-10
TOL_MU = 1e-10
FIXED_POINT_MAXITER = 200000
THETA_INIT = [0.0]
BFGS_MAXITER = 100
BFGS_GTOL = 1e-5
NELDER_MEAD_MAXITER = 500
CHECK_GRADIENT = True
# The uploaded old script also included numerical BFGS between the two solvers.
# Default False follows the requested BFGS -> Nelder-Mead sequence.
RUN_NUMERICAL_BFGS = False
NUMERICAL_MAXITER = 500
NUMERICAL_GTOL = 1e-2
OUTPUT_DIR = RESULTS_ROOT / 'small_network'
