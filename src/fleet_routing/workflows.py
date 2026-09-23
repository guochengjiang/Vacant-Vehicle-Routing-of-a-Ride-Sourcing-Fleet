"""Explicit orchestration for calibration, optimization and simulation."""
from dataclasses import asdict
from pathlib import Path
import time
import warnings
import numpy as np
from fleet_routing.model.features import normalize_features
from fleet_routing.model.policy import (extract_state_action_pairs, compute_softmax_policy_by_action_features,
                    make_uniform_policy, make_hotspot_policy)
from fleet_routing.model.fixed_point import compute_joint_fixed_point_final
from fleet_routing.optimization.sensitivities import solve_sensitivities_direct
from fleet_routing.optimization.optimizer import GradientOptimizer
from fleet_routing.simulation import run_simulation_real, run_repeated_simulation
from fleet_routing.estimation import estimate_alpha_from_trajectories
from fleet_routing.results_io import save_result, load_result


def generate_trajectories(network, cfg, output_file):
    """Generate and save calibration trajectories; do not estimate alpha."""
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    lam = cfg.scaled_demand(network)
    policies = {
        'random_walk': make_uniform_policy(network.outgoing_links, network.n_links),
        'hotspot_low': make_hotspot_policy(network.outgoing_links, network.n_links, network.lambda_vec, 0.5),
        'hotspot_high': make_hotspot_policy(network.outgoing_links, network.n_links, network.lambda_vec, 2.0),
    }
    trajectories, distributions = [], []
    for name, pi in policies.items():
        print(f'Calibration policy: {name}')
        _, _, traj, stats = run_simulation_real(
            pi, cfg.M, cfg.sim_hours, cfg.warmup_hours, cfg.tick_duration,
            lambda_input=lam, network=network, beta=cfg.beta)
        trajectories.extend(traj)
        distributions.append(stats['z_avg'])
    z_avg = np.mean(distributions, axis=0)
    result = dict(trajectories=trajectories, z_avg_combined=z_avg,
                  lambda_sim=lam, tau=network.tau, Parameters=asdict(cfg),
                  policies=list(policies))
    save_result(result, output_file)
    return result


def estimate_alpha(network, cfg, trajectory_file, output_file):
    """Load saved trajectories and estimate alpha without running simulation.

    Legacy files must contain trajectories and z_avg_combined. When demand
    metadata is absent, the configured demand is used with an explicit notice.
    """
    trajectory_file, output_file = Path(trajectory_file), Path(output_file)
    if trajectory_file.resolve() == output_file.resolve():
        raise ValueError('Trajectory input and calibration output must be different files')
    saved = load_result(trajectory_file)
    for key in ('trajectories', 'z_avg_combined'):
        if key not in saved:
            raise ValueError(f'Trajectory file is missing required field: {key}')
    trajectories = saved['trajectories']
    if len(trajectories) == 0:
        raise ValueError('Trajectory file contains no transitions')
    z_avg = np.asarray(saved['z_avg_combined'], dtype=float)
    if z_avg.shape != (network.n_links,) or not np.all(np.isfinite(z_avg)) or np.any(z_avg < 0):
        raise ValueError('z_avg_combined must be a finite nonnegative vector of length n_links')
    lam = cfg.scaled_demand(network)
    if 'lambda_sim' in saved:
        old_lam = np.asarray(saved['lambda_sim'])
        if old_lam.shape != lam.shape or not np.allclose(old_lam, lam):
            raise ValueError('Saved trajectory demand differs from config.py demand')
    else:
        print('Legacy file has no lambda_sim metadata. Using configured demand; '
              'verify that it matches the original experiment.')
    if 'tau' in saved:
        old_tau = np.asarray(saved['tau'])
        if old_tau.shape != network.tau.shape or not np.allclose(old_tau, network.tau):
            raise ValueError('Saved trajectory travel times differ from current network')
    print(f'Loading trajectories: {trajectory_file}')
    print(f'Estimating alpha from {len(trajectories)} saved transitions (no simulation).')
    alpha, ll2 = estimate_alpha_from_trajectories(trajectories, lam, network.tau, z_avg)
    params = asdict(cfg)
    params['alpha'] = float(alpha)
    result = dict(alpha=float(alpha), ll2=float(ll2), lambda_sim=lam,
                  Parameters=params, trajectory_source=str(trajectory_file.resolve()),
                  trajectory_parameters=saved.get('Parameters'),
                  z_avg_combined=z_avg, n_transitions=len(trajectories))
    save_result(result, output_file)
    return result


def optimize_policy(network, cfg, output_file, calibration_file=None,
                    theta_init=None, features=None):
    if cfg.heuristic not in (None, 'CF'):
        raise ValueError("heuristic must be None or 'CF'")
    lam = cfg.scaled_demand(network)
    alpha = cfg.alpha
    if alpha is None:
        if calibration_file is None:
            raise ValueError('Set alpha in config.py or provide a calibration file')
        calibration = load_result(calibration_file)
        if not np.allclose(calibration['lambda_sim'], lam):
            raise ValueError('Calibration demand differs from current demand; select a matching file')
        alpha = calibration['alpha']
    raw = network.features if features is None else features
    normalized, minima, ranges = normalize_features(raw)
    optimizer = GradientOptimizer(
        outgoing_dict=network.outgoing_links, action_features=normalized,
        lambda_vec=lam, tau=network.tau, omega=network.omega, M=cfg.M,
        alpha=alpha, Q=network.Q, v=network.v, c=network.c, beta=cfg.beta,
        init_mu=network.init_dist, init_m=np.full(network.n_links, 0.2, dtype=np.float32),
        outgoing_mask=network.outgoing_mask(),
        solve_sensitivities=solve_sensitivities_direct,
        solve_fixed_point=compute_joint_fixed_point_final,
        tol_m=cfg.tol_m, tol_mu=cfg.tol_mu, max_iter=cfg.fixed_point_maxiter,
        method=cfg.fixed_point_method, restart_interval=cfg.restart_interval,
        momentum=0.8, lr=0.5, phase1_tol=1e-4, heuristic=cfg.heuristic)
    if cfg.heuristic == 'CF':
        warnings.warn('CF retains the notebook sensitivity equations. Its gradient has not been '
                      're-derived for fixed matching probabilities; validate before interpreting the optimum.')
    theta_init = np.zeros(len(normalized)) if theta_init is None else np.asarray(theta_init, dtype=float)
    start = time.perf_counter()
    result = optimizer.optimize(theta_init, maxiter=cfg.maxiter, gtol=cfg.gtol)
    duration = time.perf_counter() - start
    row, col = extract_state_action_pairs(network.outgoing_links)
    pi = compute_softmax_policy_by_action_features(normalized, result.x, network.n_links, row, col)
    params = asdict(cfg)
    params['alpha'] = float(alpha)
    saved = dict(
        theta_hist=optimizer.theta_hist,
        # Notebook used ranges without epsilon here; keep that legacy field.
        theta_hist_original=[np.divide(t, ranges, out=np.full_like(t, np.nan), where=ranges != 0)
                             for t in optimizer.theta_hist],
        theta_raw_equivalent=result.x / (ranges + 1e-8),
        R_hist=optimizer.R_hist, theta_hist_BFGS=optimizer.iter_theta_hist,
        R_hist_BFGS=optimizer.iter_R_hist, m_hist=optimizer.m_hist,
        mu_hist=optimizer.mu_hist, state_hist=optimizer.state_hist,
        grad_hist=optimizer.grad_hist, grad_norm_hist=optimizer.grad_norm_hist,
        optimal_theta=result.x, optimal_R=-result.fun, Duration=duration,
        nfev=result.nfev, njev=result.njev, Full_result=result,
        Parameters=params, feature_minima=minima, feature_ranges=ranges,
        normalized_features=normalized, policy=pi, lambda_sim=lam)
    save_result(saved, output_file)
    if not result.success:
        warnings.warn(f'Optimizer did not report success: {result.message}')
    return saved


def simulate_policy(network, cfg, policy_name, output_dir, optimization_file=None):
    lam = cfg.scaled_demand(network)
    if policy_name == 'optimized':
        saved = load_result(optimization_file)
        params = saved['Parameters']
        if params['M'] != cfg.M or params['beta'] != cfg.beta or not np.allclose(saved['lambda_sim'], lam):
            raise ValueError('Optimization and simulation M/beta/demand differ; select the correct result')
        pi = saved['policy']
        label = 'CF' if params.get('heuristic') == 'CF' else 'optimal'
    elif policy_name == 'random_walk':
        pi = make_uniform_policy(network.outgoing_links, network.n_links)
        label = policy_name
    elif policy_name in ('hotspot_low', 'hotspot_high'):
        pi = make_hotspot_policy(network.outgoing_links, network.n_links, network.lambda_vec,
                                 0.5 if policy_name == 'hotspot_low' else 2.0)
        label = policy_name
    else:
        raise ValueError(f'Unknown policy: {policy_name}')
    name = f'{label}_demand{cfg.demand_reference_fleet}_{cfg.sim_hours:g}h'
    return run_repeated_simulation(
        pi, name, n_runs=cfg.n_runs, M=cfg.M, sim_hours=cfg.sim_hours,
        warmup_hours=cfg.warmup_hours, lambda_input=lam,
        save_trajectories=cfg.save_trajectories, network=network, beta=cfg.beta,
        output_dir=output_dir, seed=cfg.seed, tick_duration=cfg.tick_duration)
