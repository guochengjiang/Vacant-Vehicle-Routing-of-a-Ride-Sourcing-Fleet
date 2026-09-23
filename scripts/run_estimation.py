import _bootstrap
"""Estimate alpha from a saved trajectory file; never run new simulations."""
from configs.shanghai import CONFIG, INPUT_DIR, TRAJECTORY_FILE, CALIBRATION_FILE
from fleet_routing.networks.shanghai import load_network
from fleet_routing.workflows import estimate_alpha

if __name__ == '__main__':
    network = load_network(INPUT_DIR)
    estimate_alpha(network, CONFIG, TRAJECTORY_FILE, CALIBRATION_FILE)
