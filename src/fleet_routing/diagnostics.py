"""Construct independent optimizer instances for numerical diagnostics."""
import numpy as np
from fleet_routing.model.features import normalize_features
from fleet_routing.model.fixed_point import compute_joint_fixed_point_final
from fleet_routing.optimization.optimizer import GradientOptimizer
from fleet_routing.optimization.sensitivities import solve_sensitivities_direct


def make_optimizer(network, cfg):
    if cfg.alpha is None:
        raise ValueError('Diagnostics require an explicit alpha')
    features, _, _ = normalize_features(network.features)
    return GradientOptimizer(
        outgoing_dict=network.outgoing_links, action_features=features,
        lambda_vec=cfg.scaled_demand(network), tau=network.tau, omega=network.omega,
        M=cfg.M, alpha=cfg.alpha, Q=network.Q, v=network.v, c=network.c,
        beta=cfg.beta, init_mu=network.init_dist.copy(), init_m=np.full(network.n_links, .2),
        outgoing_mask=network.outgoing_mask(), solve_sensitivities=solve_sensitivities_direct,
        solve_fixed_point=compute_joint_fixed_point_final, method=cfg.fixed_point_method,
        max_iter=cfg.fixed_point_maxiter, tol_m=cfg.tol_m, tol_mu=cfg.tol_mu,
        heuristic=cfg.heuristic)
