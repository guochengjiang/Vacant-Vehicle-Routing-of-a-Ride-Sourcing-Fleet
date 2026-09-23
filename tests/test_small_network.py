"""Topology, units, policy support and independent finite-difference checks."""
from dataclasses import replace
import numpy as np
from fleet_routing.networks.small import build_small_network
from fleet_routing.config import ExperimentConfig
from fleet_routing.diagnostics import make_optimizer
from fleet_routing.model.policy import extract_state_action_pairs, compute_softmax_policy_by_action_features
from fleet_routing.model.features import normalize_features


def test_small_network():
    network, meta = build_small_network()
    assert network.n_links == 76
    tail, head = meta['links'].T
    for i, actions in network.outgoing_links.items():
        assert actions and np.all(tail[actions] == head[i])
    np.testing.assert_allclose(network.tau, meta['lengths'] / 40)
    np.testing.assert_allclose(network.omega, (network.Q * network.c).sum(axis=1))
    np.testing.assert_allclose(network.Q.sum(axis=1), 1)
    f, _, _ = normalize_features(network.features)
    row, col = extract_state_action_pairs(network.outgoing_links)
    pi = compute_softmax_policy_by_action_features(f, np.array([.3]), network.n_links, row, col).toarray()
    np.testing.assert_allclose(pi.sum(axis=1), 1)
    assert np.all(pi[network.outgoing_mask() == 0] == 0)
    # Use a tiny grid and fresh optimizers to avoid cache/warm-start artifacts.
    network, _ = build_small_network(2, 2)
    cfg = ExperimentConfig(M=10, alpha=.8, demand_reference_fleet=12000,
                           tol_m=1e-11, tol_mu=1e-11, fixed_point_maxiter=100000)
    theta = np.array([.3])
    analytic = make_optimizer(network, cfg).grad(theta)
    h = 1e-3
    numeric = (make_optimizer(network, cfg).objective(theta + h)
               - make_optimizer(network, cfg).objective(theta - h)) / (2 * h)
    np.testing.assert_allclose(analytic, [numeric], rtol=2e-3, atol=2e-4)
