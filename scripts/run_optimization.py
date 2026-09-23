import _bootstrap
"""Edit config.py, then right-click this file in PyCharm and choose Run."""
from configs.shanghai import CONFIG, INPUT_DIR, CALIBRATION_FILE, OPTIMIZATION_FILE
from fleet_routing.networks.shanghai import load_network
from fleet_routing.workflows import optimize_policy

if __name__ == '__main__':
    network = load_network(INPUT_DIR)
    optimize_policy(network, CONFIG, OPTIMIZATION_FILE, CALIBRATION_FILE)
