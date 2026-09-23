"""Original small-network experiment construction with unnormalized demand features."""
import numpy as np
from fleet_routing.networks.legacy_small import build_sioux_falls_like, compute_expected_time
from fleet_routing.optimization.optimizer import GradientOptimizer
from fleet_routing.model.fixed_point import compute_joint_fixed_point_final
from fleet_routing.optimization.sensitivities import solve_sensitivities_direct


def _quiet_fixed_point(*args, **kwargs):
    """Use the original small-network logging level without changing equations."""
    kwargs['verbose'] = False
    return compute_joint_fixed_point_final(*args, **kwargs)


def make_small_network_optimizer(cfg):
    net = build_sioux_falls_like(seed=cfg.NETWORK_SEED, seed2=cfg.DEMAND_SEED)
    print("n_links:", net["n_links"])
    print("c shape:", net["c"].shape)

    # compute expected time
    omega = compute_expected_time(net["dest"], net["c_time"])

    # outgoing_mask
    n = len(net["lambda_vec"])
    outgoing_mask = np.zeros((n, n), dtype=np.float64)
    for s, actions in net["outgoing_dict"].items():
        outgoing_mask[s, actions] = 1.0


    M = cfg.M
    alpha = cfg.ALPHA
    n_links = len(net["lambda_vec"])
    init_m = np.full(n_links, 0.2, dtype=np.float64)
    init_mu = np.full(n_links, 1 / n_links, dtype=np.float64)

    features = [net["lambda_vec"]]
    Q = net["dest"].copy()

    initial_charge = 35
    unit_fare = 35
    v = initial_charge + unit_fare * net["c"]
    v = np.round(v, 2)
    beta = 0.6 * 40


    optimizer = GradientOptimizer(
        outgoing_dict=net["outgoing_dict"],
        action_features=features,
        lambda_vec=net["lambda_vec"],
        tau=net["tau"],
        omega=omega,
        M=M,
        alpha=alpha,
        Q=Q,
        v=v,
        c=net["c"],
        beta=beta,
        init_mu=init_mu,
        init_m=init_m,
        outgoing_mask=outgoing_mask,
        solve_sensitivities=solve_sensitivities_direct,
        solve_fixed_point=_quiet_fixed_point,
        tol_m=cfg.TOL_M,
        tol_mu=cfg.TOL_MU,
        max_iter=cfg.FIXED_POINT_MAXITER,
        method='msa',
        restart_interval=10000,
        momentum=0.8,
        lr=0.5,
        phase1_tol=1e-4
    )

    return optimizer, net
