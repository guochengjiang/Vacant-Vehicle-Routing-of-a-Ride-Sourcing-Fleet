import _bootstrap
"""Run repeated simulations of a saved optimized policy or a baseline."""
from configs.shanghai import CONFIG, INPUT_DIR, OUTPUT_DIR, SIMULATION_POLICY, OPTIMIZATION_FILE
from fleet_routing.networks.shanghai import load_network
from fleet_routing.workflows import simulate_policy

if __name__ == '__main__':
    network = load_network(INPUT_DIR)
    simulate_policy(network, CONFIG, SIMULATION_POLICY, OUTPUT_DIR, OPTIMIZATION_FILE)
