"""Run original small-network BFGS, gradient check and Nelder-Mead; no simulation."""
import _bootstrap
import numpy as np
from configs import small_network as cfg
from fleet_routing.small_experiment import make_small_network_optimizer
from fleet_routing.results_io import save_result


def report_and_save(name, result, optimizer):
    print(f'\n{name} result')
    print(f'Optimal theta: {result.x}')
    print(f'Optimal R: {-result.fun:.6f}')
    print(f'Success: {result.success}; status: {result.status}')
    print(f'Message: {result.message}')
    save_result(dict(result=result, theta=result.x, R=-result.fun,
                     theta_history=optimizer.iter_theta_hist.copy(),
                     feature_convention='raw lambda_vec; no normalization',
                     network_seed=cfg.NETWORK_SEED, demand_seed=cfg.DEMAND_SEED,
                     M=cfg.M, alpha=cfg.ALPHA), cfg.OUTPUT_DIR / f'legacy_{name}.pkl')


def main():
    optimizer, _ = make_small_network_optimizer(cfg)
    theta_init = np.asarray(cfg.THETA_INIT, dtype=float)
    print('\n=== Analytic BFGS ===')
    result = optimizer.optimize(theta_init.copy(), maxiter=cfg.BFGS_MAXITER, gtol=cfg.BFGS_GTOL)
    report_and_save('BFGS', result, optimizer)
    if cfg.CHECK_GRADIENT:
        optimizer.check_gradient(theta_init.copy())
    if cfg.RUN_NUMERICAL_BFGS:
        print('\n=== Numerical BFGS ===')
        result_num = optimizer.optimize_numerical(theta_init.copy(), cfg.NUMERICAL_MAXITER,
                                                  cfg.NUMERICAL_GTOL)
        report_and_save('numerical_BFGS', result_num, optimizer)
    print('\n=== Nelder-Mead ===')
    result_nelder = optimizer.optimize_nelder_mead(theta_init.copy(), maxiter=cfg.NELDER_MEAD_MAXITER)
    report_and_save('Nelder_Mead', result_nelder, optimizer)


if __name__ == '__main__':
    main()
