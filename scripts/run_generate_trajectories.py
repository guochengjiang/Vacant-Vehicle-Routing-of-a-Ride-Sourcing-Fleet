import _bootstrap
"""Generate calibration trajectories only; run estimation separately afterward."""
from configs.shanghai import CONFIG, INPUT_DIR, GENERATED_TRAJECTORY_FILE
from fleet_routing.networks.shanghai import load_network
from fleet_routing.workflows import generate_trajectories

if __name__ == '__main__':
    network = load_network(INPUT_DIR)
    generate_trajectories(network, CONFIG, GENERATED_TRAJECTORY_FILE)
