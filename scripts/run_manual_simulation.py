"""Build a policy from manually entered theta and save repeated simulations."""
import _bootstrap
import numpy as np
from configs.shanghai import (CONFIG, INPUT_DIR, OUTPUT_DIR, MANUAL_THETA,
                              MANUAL_NAME, FEATURE_REFERENCE_FILE)
from fleet_routing.networks.shanghai import load_network
from fleet_routing.model.features import normalize_features
from fleet_routing.model.policy import extract_state_action_pairs, compute_softmax_policy_by_action_features
from fleet_routing.results_io import load_result, save_result
from fleet_routing.simulation import run_repeated_simulation

if __name__ == '__main__':
    network = load_network(INPUT_DIR)
    theta = np.asarray(MANUAL_THETA, dtype=float)
    if FEATURE_REFERENCE_FILE is None:
        features, _, _ = normalize_features(network.features)
    else:
        features = np.asarray(load_result(FEATURE_REFERENCE_FILE)['normalized_features'])
    if (features.ndim != 2 or features.shape[1] != network.n_links
            or theta.shape != (features.shape[0],)
            or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(features))):
        raise ValueError('Theta and features must be finite and match the current network')
    row, col = extract_state_action_pairs(network.outgoing_links)
    pi = compute_softmax_policy_by_action_features(features, theta, network.n_links, row, col)
    summary = run_repeated_simulation(
        pi, MANUAL_NAME, n_runs=CONFIG.n_runs, M=CONFIG.M,
        sim_hours=CONFIG.sim_hours, warmup_hours=CONFIG.warmup_hours,
        tick_duration=CONFIG.tick_duration, lambda_input=CONFIG.scaled_demand(network),
        network=network, beta=CONFIG.beta, output_dir=OUTPUT_DIR,
        seed=CONFIG.seed, save_trajectories=CONFIG.save_trajectories)
    save_result({'theta': theta, 'normalized_features': features, 'policy': pi,
                 'summary': summary}, OUTPUT_DIR / f'{MANUAL_NAME}_policy_M{CONFIG.M}.pkl')
