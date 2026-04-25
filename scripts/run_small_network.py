"""
Small network test
"""
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from src.sioux_fall_build.build_siouxfall import build_sioux_falls_like, compute_expected_time
from src.optimization.optimizer import GradientOptimizer
from src.model.fixed_point import compute_joint_fixed_point_final, build_P_matrix
from src.optimization.sensitivity import solve_sensitivities_direct



def main():

    net = build_sioux_falls_like(seed=2025, seed2=2025)
    print("n_links:", net["n_links"])
    print("c shape:", net["c"].shape)

    # compute expected time
    omega = compute_expected_time(net["dest"], net["c_time"])

    # outgoing_mask
    n = len(net["lambda_vec"])
    outgoing_mask = np.zeros((n, n), dtype=np.float64)
    for s, actions in net["outgoing_dict"].items():
        outgoing_mask[s, actions] = 1.0


    M = 1000
    alpha = 0.8
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
        solve_fixed_point=compute_joint_fixed_point_final,
        tol_m=1e-10,
        tol_mu=1e-10,
        max_iter= 200000,
        method='msa',
        restart_interval=10000,
        momentum=0.8,
        lr=0.5,
        phase1_tol=1e-4
    )

    # run
    theta_init = np.array([0.0])
    result = optimizer.optimize(theta_init, maxiter=100)

    # check grad
    optimizer.check_gradient(np.array([0.0]))

    print(f"\nOptimal theta: {result.x}")
    print(f"Optimal R: {-result.fun:.6f}")
    print(f"\n------------------------------------------\n")


    theta_init = np.array([0.0])
    result_num = optimizer.optimize_numerical(theta_init, 500, 1e-2)

    print(f"\nOptimal theta: {result.x}")
    print(f"Optimal R: {-result_num.fun:.6f}")
    print(f"\n------------------------------------------\n")

    theta_init = np.array([0.0])
    result_nelder = optimizer.optimize_nelder_mead(theta_init, maxiter=500)

    print(f"\nOptimal theta: {result.x}")
    print(f"Optimal R: {-result_nelder.fun:.6f}")




if __name__ == '__main__':
    main()