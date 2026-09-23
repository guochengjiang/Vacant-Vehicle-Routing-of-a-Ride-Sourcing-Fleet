"""Small synthetic regression checks; needs no Shanghai Input files.
Run: python verify_refactor.py
Checks extracted notebook computations against the refactored implementations.
"""
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import json
import re
import numpy as np
from fleet_routing.networks.shanghai import NetworkData
from fleet_routing.config import ExperimentConfig
from fleet_routing.model.features import normalize_features
from fleet_routing.model.policy import extract_state_action_pairs, compute_softmax_policy_by_action_features
from fleet_routing.model.fixed_point import compute_joint_fixed_point_final
from fleet_routing.optimization.optimizer import GradientOptimizer
from fleet_routing.optimization.sensitivities import solve_sensitivities_direct
from fleet_routing.simulation import run_simulation_real
from fleet_routing.estimation import estimate_alpha_from_trajectories
from fleet_routing.workflows import generate_trajectories, estimate_alpha, optimize_policy, simulate_policy
from fleet_routing.results_io import save_result
from unittest.mock import patch


def original_namespace():
    text = (Path(__file__).resolve().parents[1] / 'archive' / 'notebook_cells.txt').read_text()
    parts = re.split(r'===== CELL (\d+) \(\w+\) =====\n', text)
    cells = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
    ns = {}
    for i in (0, 5, 3, 4, 1, 9, 10, 11):
        exec(compile(cells[i], f'notebook_cell_{i}', 'exec'), ns)
    return ns


def test_notebook_parity_and_workflows():
    ns = original_namespace()
    network = NetworkData(
        c=np.full((3, 3), 0.025), v=np.full((3, 3), 8.),
        Q=np.array([[.2, .3, .5], [.4, .2, .4], [.3, .4, .3]]),
        lambda_vec=np.array([10., 20., 15.]), tau=np.array([.01, .015, .02]),
        omega=np.full(3, .025), features=np.array([[1., 2., 4.]]),
        init_dist=np.full(3, 1/3), outgoing_links={0:[1,2], 1:[0,2], 2:[0,1]})
    features, _, _ = normalize_features(network.features)
    row, col = extract_state_action_pairs(network.outgoing_links)
    theta = np.array([.3])
    pi = compute_softmax_policy_by_action_features(features, theta, 3, row, col)
    pi_old = ns['compute_softmax_policy_by_action_features'](features, theta, 3, row, col)
    np.testing.assert_array_equal(pi.toarray(), pi_old.toarray())
    kw = dict(outgoing_dict=network.outgoing_links, action_features=features,
              lambda_vec=network.lambda_vec, tau=network.tau, omega=network.omega,
              M=5, alpha=.8, Q=network.Q, v=network.v, c=network.c, beta=30,
              init_mu=network.init_dist, init_m=np.full(3,.2),
              outgoing_mask=network.outgoing_mask(), method='msa', max_iter=10000,
              tol_m=1e-10, tol_mu=1e-10)
    new = GradientOptimizer(**kw, solve_sensitivities=solve_sensitivities_direct,
                            solve_fixed_point=compute_joint_fixed_point_final)
    old = ns['GradientOptimizer'](**kw, solve_sensitivities=ns['solve_sensitivities_direct'],
                                  solve_fixed_point=ns['compute_joint_fixed_point_final'])
    with redirect_stdout(io.StringIO()):
        np.testing.assert_allclose(new.objective(theta), old.objective(theta), rtol=1e-12)
        np.testing.assert_allclose(new.grad(theta), old.grad(theta), rtol=1e-12)
        np.testing.assert_array_equal(new.mu_last, old.mu_last)
        np.testing.assert_array_equal(new.m_last, old.m_last)
        # Both entry orders must work; list theta is accepted after refactoring.
        new.clear_cache()
        np.testing.assert_allclose(new.grad([.3]), old.grad(theta), rtol=1e-7)
    for key in ('tau','c','v','Q','lambda_vec','init_dist'):
        ns[key] = getattr(network,key)
    ns['n_links'], ns['beta'] = 3, 30
    sim_kw = dict(M=5, sim_hours=.2, warmup_hours=.05, verbose=False)
    np.random.seed(42)
    before = ns['run_simulation_real'](pi, **sim_kw)
    np.random.seed(42)
    after = run_simulation_real(pi, **sim_kw, network=network)
    assert before[:3] == after[:3]
    for key in before[3]:
        np.testing.assert_array_equal(before[3][key], after[3][key])
    with redirect_stdout(io.StringIO()):
        a = ns['estimate_alpha_from_trajectories'](before[2], network.lambda_vec, network.tau, before[3]['z_avg'])
        b = estimate_alpha_from_trajectories(after[2], network.lambda_vec, network.tau, after[3]['z_avg'])
        np.testing.assert_array_equal(a, b)
    with TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
        d = Path(d)
        cfg = ExperimentConfig(M=5, demand_reference_fleet=12000, alpha=.8,
                               sim_hours=.2, warmup_hours=.05, n_runs=2, maxiter=20,
                               fixed_point_maxiter=10000, gtol=1e-3, save_trajectories=False)
        generated = generate_trajectories(network, cfg, d/'trajectories.pkl')
        with patch('fleet_routing.workflows.run_simulation_real', side_effect=AssertionError('Estimation must not simulate')):
            calibration = estimate_alpha(network, cfg, d/'trajectories.pkl', d/'calibration.pkl')
            save_result({k: generated[k] for k in ('trajectories', 'z_avg_combined')}, d/'legacy.pkl')
            legacy = estimate_alpha(network, cfg, d/'legacy.pkl', d/'legacy_calibration.pkl')
            np.testing.assert_equal(calibration['alpha'], legacy['alpha'])
            try:
                estimate_alpha(network, replace(cfg, demand_reference_fleet=6000),
                               d/'trajectories.pkl', d/'bad.pkl')
            except ValueError as exc:
                assert 'demand' in str(exc)
            else:
                raise AssertionError('Demand mismatch must be rejected')
        cfg = replace(cfg, alpha=None)
        saved = optimize_policy(network, cfg, d/'optimization.pkl', d/'calibration.pkl')
        summary = simulate_policy(network, cfg, 'optimized', d, d/'optimization.pkl')
        assert np.isfinite(saved['optimal_R']) and np.isfinite(summary['avg_R_mean'])
        assert summary['n_runs'] == 2
        summary_paths = list(d.glob('*.summary.json'))
        assert len(summary_paths) == 1
        readable = json.loads(summary_paths[0].read_text())
        np.testing.assert_allclose(readable['avg_R_mean'], summary['avg_R_mean'])
        assert readable['n_runs'] == 2
    print('PASS: policy, fixed point, objective, gradient, seeded simulation, alpha estimation.')
    print('PASS: trajectory generation -> estimation -> optimization -> simulation; legacy input and demand checks.')
    print('Shanghai network validation requires the original Input data; not performed here.')


if __name__ == '__main__':
    test_notebook_parity_and_workflows()
