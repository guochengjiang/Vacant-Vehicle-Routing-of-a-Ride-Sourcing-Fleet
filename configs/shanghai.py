"""Shanghai experiment settings. Edit CONFIG and file selections here."""
from fleet_routing.config import ExperimentConfig, INPUT_DIR, OUTPUT_DIR

CONFIG = ExperimentConfig()
# To reuse M5000 calibration for M10000, explicitly point this at the M5000 file.
CALIBRATION_FILE = CONFIG.calibration_file
# Simulation modes: 'optimized', 'random_walk', 'hotspot_low', 'hotspot_high'.
SIMULATION_POLICY = 'optimized'
OPTIMIZATION_FILE = CONFIG.optimization_file

# Generation output and estimation input are separate to protect legacy files.
GENERATED_TRAJECTORY_FILE = OUTPUT_DIR / (
    f'trajectories_M{CONFIG.M}_demand{CONFIG.demand_reference_fleet}.pkl'
)
#TRAJECTORY_FILE = GENERATED_TRAJECTORY_FILE
# To estimate from an old notebook file, replace the line above with:
TRAJECTORY_FILE = OUTPUT_DIR / 'simulation_data_M5000_Jun9_iter0.pkl'

# Manual theta must use the same feature normalization as optimization.
MANUAL_THETA = [54.8567]  # Replace with result.x values.
MANUAL_NAME = "manual_theta"
FEATURE_REFERENCE_FILE = None  # Optional optimization file containing normalized_features.
