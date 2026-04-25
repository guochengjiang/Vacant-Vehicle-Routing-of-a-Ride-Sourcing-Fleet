import numpy as np
from scipy.optimize import minimize


from ..model.reward import compute_kappa, compute_R
from ..model.policy import extract_state_action_pairs, compute_softmax_policy_by_action_features
from scipy.sparse import coo_matrix, csr_matrix, issparse, diags
from .checkpoint import save_checkpoint, load_checkpoint
from scipy.optimize import minimize
import gc
import psutil


class GradientOptimizer:

    def __init__(self,
                 outgoing_dict,
                 action_features,
                 lambda_vec, tau, omega, M, alpha,
                 Q, v, c, beta,
                 init_mu,
                 init_m,
                 outgoing_mask,
                 solve_sensitivities,
                 solve_fixed_point,
                 # fixed point hyperparameters
                 fixed_m_states=None,
                 tol_m=1e-6,
                 tol_mu=1e-6,
                 max_iter=100000,
                 method='hybrid',
                 restart_interval=30,
                 momentum=0.9,
                 lr=0.5,
                 phase1_tol=1e-4):

        self.outgoing_dict = outgoing_dict
        self.action_features = action_features
        self.lambda_vec = lambda_vec
        self.tau = tau
        self.omega = omega
        self.M = M
        self.alpha = alpha
        self.Q = Q
        self.v = v
        self.c = c
        self.beta = beta
        self.outgoing_mask = outgoing_mask
        self.solver_grad = solve_sensitivities
        self.solver_fixed_point = solve_fixed_point

        self.fixed_m_states = fixed_m_states
        self.tol_m = tol_m
        self.tol_mu = tol_mu
        self.max_iter = max_iter
        self.method = method
        self.restart_interval = restart_interval
        self.momentum = momentum
        self.lr = lr
        self.phase1_tol = phase1_tol

        self.n = len(lambda_vec)
        self.row_idx, self.col_idx = extract_state_action_pairs(outgoing_dict)

        # Warm start
        self.mu_last = init_mu.copy()
        self.m_last = init_m.copy()
        self.init_mu = init_mu.copy()
        self.init_m = init_m.copy()

        self.current_state = {}

        # history
        self.theta_hist = []
        self.R_hist = []
        self.m_hist = []
        self.mu_hist = []
        self.state_hist = []
        self.grad_hist = []
        self.grad_norm_hist = []

        self.iter_theta_hist = []
        self.iter_R_hist = []
        self.iter_grad_norm_hist = []
        self.iter_theta_hist_bfgs_num = []
        self.iter_theta_hist_nelder = []

        # cache
        self._cache_theta = None
        self._cache_obj = None
        self._cache_grad = None

    def reset_warm_start(self):
        self.mu_last = self.init_mu.copy()
        self.m_last = self.init_m.copy()

    def clear_cache(self):
        self._cache_theta = None
        self._cache_obj = None
        self._cache_grad = None

    def clear_history(self):
        self.theta_hist = []
        self.R_hist = []
        self.m_hist = []
        self.mu_hist = []
        self.state_hist = []
        self.grad_hist = []
        self.grad_norm_hist = []
        self.iter_theta_hist = []
        self.iter_R_hist = []
        self.iter_grad_norm_hist = []

        self.iter_theta_hist_bfgs_num = []
        self.iter_theta_hist_nelder = []

    def _compute_fixed_point(self, theta_vec):
        pi = compute_softmax_policy_by_action_features(
            self.action_features, theta_vec, self.n, self.row_idx, self.col_idx
        )
        pi_dense = pi.toarray() if issparse(pi) else np.asarray(pi)

        mu, m = self.solver_fixed_point(
            self.lambda_vec, self.tau, self.omega, self.M, self.alpha,
            self.Q, self.outgoing_dict, pi,
            self.mu_last, self.m_last, self.row_idx, self.col_idx,
            self.v, self.c, self.beta,
            fixed_m_states=self.fixed_m_states,
            tol_m=self.tol_m,
            tol_mu=self.tol_mu,
            max_iter=self.max_iter,
            method=self.method,
            restart_interval=self.restart_interval,
            momentum=self.momentum,
            lr=self.lr,
            phase1_tol=self.phase1_tol,
            verbose=False,
            heuristic=None
        )

        # update warm start
        self.mu_last = mu.copy()
        self.m_last = m.copy()

        return mu, m, pi_dense

    def _build_feature_matrix(self, k):
        x_r = np.zeros((self.n, self.n), dtype=np.float64)
        for s, actions in self.outgoing_dict.items():
            for a in actions:
                x_r[s, a] = self.action_features[k][a]
        return x_r

    def _compute(self, theta):
        if (self._cache_theta is not None and
                np.array_equal(theta, self._cache_theta)):
            return

        print(f"[Step {len(self.theta_hist) + 1}] theta = {theta}, computing...")

        mu, m, pi_dense = self._compute_fixed_point(theta)

        kappa = compute_kappa(self.v, self.c, self.Q, self.beta)
        R_pi = compute_R(mu, self.tau, self.omega, m, kappa, self.beta)

        N = np.sum(mu * (-self.beta * self.tau + kappa * m))
        D = np.sum(mu * (self.tau + self.omega * m))

        dN_dm = mu * kappa
        dN_dmu = -self.beta * self.tau + kappa * m
        dD_dm = mu * self.omega
        dD_dmu = self.tau + self.omega * m


        K = len(theta)
        grad_R = np.zeros(K)

        for k in range(K):
            x_r_k = self._build_feature_matrix(k)

            dmu, dm, info = self.solver_grad(
                self.lambda_vec, self.tau, self.omega, self.M, self.alpha,
                self.Q, pi_dense, mu, m,
                x_r_k, self.outgoing_mask, self.outgoing_dict
            )

            dN_dtheta = np.dot(dN_dm, dm) + np.dot(dN_dmu, dmu)
            dD_dtheta = np.dot(dD_dm, dm) + np.dot(dD_dmu, dmu)
            grad_R[k] = (dN_dtheta * D - N * dD_dtheta) / (D ** 2)


        grad_norm = np.linalg.norm(-grad_R)

        self.current_state = {
            'mu': mu, 'm': m,
            # 'pi': pi_dense,
            'Q': self.Q,
            'kappa': kappa, 'N': N, 'D': D,
            'dN_dm': dN_dm, 'dN_dmu': dN_dmu,
            'dD_dm': dD_dm, 'dD_dmu': dD_dmu,
            'R': R_pi,
            'grad': -grad_R,
            'grad_norm': grad_norm
        }

        self.theta_hist.append(theta.copy())
        self.R_hist.append(R_pi)
        self.m_hist.append(m.copy())
        self.mu_hist.append(mu.copy())
        self.grad_hist.append(-grad_R.copy())
        self.grad_norm_hist.append(grad_norm)
        self.state_hist.append({
            'mu': mu.copy(),
            'm': m.copy(),
            # 'pi': pi_dense.copy(),
            'kappa': kappa.copy(),
            'N': N,
            'D': D,
            'dN_dm': dN_dm.copy(),
            'dN_dmu': dN_dmu.copy(),
            'dD_dm': dD_dm.copy(),
            'dD_dmu': dD_dmu.copy(),
            'R': R_pi,
            'grad': -grad_R.copy(),
            'grad_norm': grad_norm
        })

        save_checkpoint(self)

        print(f"[Step {len(self.theta_hist)}] theta = {theta}, R = {R_pi:.6f}, ||grad|| = {grad_norm:.2e}")

        self._cache_theta = theta.copy()
        self._cache_obj = -R_pi
        self._cache_grad = -grad_R

        del pi_dense

        gc.collect()
        mem_gb = psutil.Process().memory_info().rss / 1e9
        print(f"  Memory: {mem_gb:.2f} GB\n")

    def _compute_objective(self, theta):
        if (self._cache_theta is not None and
                np.array_equal(theta, self._cache_theta)):
            return

        print(f"[Compute f] theta = {theta}")

        # fixed point
        mu, m, pi_dense = self._compute_fixed_point(theta)

        # compute R
        kappa = compute_kappa(self.v, self.c, self.Q, self.beta)
        R_pi = compute_R(mu, self.tau, self.omega, m, kappa, self.beta)

        # cache
        self._cache_theta = theta.copy()
        self._cache_obj = -R_pi
        self._cache_mu = mu
        self._cache_m = m
        self._cache_pi = pi_dense
        self._cache_kappa = kappa
        self._cache_grad = None

    def _compute_gradient(self, theta):
        if (self._cache_theta is not None and
                np.array_equal(theta, self._cache_theta) and
                self._cache_grad is not None):
            return

        # confirm f already computed
        if not np.array_equal(theta, self._cache_theta):
            self._compute_objective(theta)

        print(f"[Compute grad] theta = {theta}")

        mu = self._cache_mu
        m = self._cache_m
        pi_dense = self._cache_pi
        kappa = self._cache_kappa

        N = np.sum(mu * (-self.beta * self.tau + kappa * m))
        D = np.sum(mu * (self.tau + self.omega * m))

        dN_dm = mu * kappa
        dN_dmu = -self.beta * self.tau + kappa * m
        dD_dm = mu * self.omega
        dD_dmu = self.tau + self.omega * m


        K = len(theta) if hasattr(theta, '__len__') else 1
        grad_R = np.zeros(K)

        for k in range(K):
            x_r_k = self._build_feature_matrix(k)

            dmu, dm, info = self.solver_grad(
                self.lambda_vec, self.tau, self.omega, self.M, self.alpha,
                self.Q, pi_dense, mu, m,
                x_r_k, self.outgoing_mask, self.outgoing_dict
            )

            dN_dtheta = np.dot(dN_dm, dm) + np.dot(dN_dmu, dmu)
            dD_dtheta = np.dot(dD_dm, dm) + np.dot(dD_dmu, dmu)
            grad_R[k] = (dN_dtheta * D - N * dD_dtheta) / (D ** 2)

        self._cache_grad = -grad_R
        grad_norm = np.linalg.norm(-grad_R)


        self.theta_hist.append(theta.copy())
        self.R_hist.append(-self._cache_obj)
        self.m_hist.append(m.copy())
        self.mu_hist.append(mu.copy())
        self.grad_hist.append(-grad_R.copy())  # 新增
        self.grad_norm_hist.append(grad_norm)  # 新增
        self.state_hist.append({
            'mu': mu.copy(),
            'm': m.copy(),
            # 'pi': pi_dense.copy(),
            'kappa': kappa.copy(),
            'N': N,
            'D': D,
            'dN_dm': dN_dm.copy(),
            'dN_dmu': dN_dmu.copy(),
            'dD_dm': dD_dm.copy(),
            'dD_dmu': dD_dmu.copy(),
            'R': -self._cache_obj,
            'grad': -grad_R.copy(),
            'grad_norm': grad_norm
        })

        self._cache_pi = None
        del pi_dense

        gc.collect()

        mem_gb = psutil.Process().memory_info().rss / 1e9
        print(f"  Memory: {mem_gb:.2f} GB. ||grad|| = {grad_norm:.2e}")

    def objective(self, theta):

        self._compute_objective(theta)
        return self._cache_obj

    def grad(self, theta):

        self._compute_gradient(theta)
        return self._cache_grad



    def optimize(self, theta_init, method='BFGS', maxiter=100, bounds=None, gtol=1e-5):

        def callback(xk):
            self.iter_theta_hist.append(xk.copy())
            self.iter_R_hist.append(-self._cache_obj)
            self.iter_grad_norm_hist.append(np.linalg.norm(self._cache_grad))

            print(f"\n[BFGS Iter {len(self.iter_theta_hist)}] "
                  f"theta = {xk}, "
                  f"R = {-self._cache_obj:.6f}, "
                  f"||grad|| = {np.linalg.norm(self._cache_grad):.2e}")
            print(f"\n----------------------------------------------------\n")

        self.clear_history()
        self.clear_cache()
        self.reset_warm_start()

        if bounds is not None and method == 'BFGS':
            method = 'L-BFGS-B'

        result = minimize(
            fun=self.objective,
            x0=theta_init,
            method=method,
            jac=self.grad,
            callback=callback,
            bounds=bounds,
            options={'disp': True, 'maxiter': maxiter, 'gtol':gtol}
        )

        return result



    def optimize_numerical(self, theta_init, maxiter=50, gtol=1e-5, epsilon=1e-6):


        def callback(xk):
            self.iter_theta_hist_bfgs_num.append(xk.copy())

            print(f"\n[BFGS Iter {len(self.iter_theta_hist_bfgs_num)}] "
                  f"theta = {xk}, "
                  f"R = {-self._cache_obj:.6f}, ")
            print(f"\n----------------------------------------------------\n")

        self.clear_history()
        self.clear_cache()
        self.reset_warm_start()

        result = minimize(
            fun=self.objective,
            x0=theta_init,
            method='BFGS',
            callback=callback,
            options={
                'disp': True,
                'maxiter': maxiter,
                'gtol': gtol,
                'eps': epsilon
            }
        )
        return result


    def optimize_nelder_mead(self, theta_init, maxiter=500, xatol=1e-6, fatol=1e-6):

        def callback(xk):
            """每次迭代结束后调用"""
            self.iter_theta_hist_nelder.append(xk.copy())
            print(f"\n[Nelder-Mead Iter {len(self.iter_theta_hist_nelder)}] "
                  f"theta = {xk}, "
                  f"R = {-self._cache_obj:.6f}")
            print(f"----------------------------------------------------\n")

        self.clear_history()
        self.clear_cache()
        self.reset_warm_start()

        result = minimize(
            fun=self.objective,
            x0=theta_init,
            method='Nelder-Mead',
            callback=callback,
            options={
                'disp': True,
                'maxiter': maxiter,
                'xatol': xatol,
                'fatol': fatol
            }
        )
        return result

    def numerical_gradient(self, theta, epsilon=1e-6):
        grad = np.zeros_like(theta)

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon

            theta_minus = theta.copy()
            theta_minus[i] -= epsilon

            self.reset_warm_start()
            self.clear_cache()
            f_plus = self.objective(theta_plus)

            self.reset_warm_start()
            self.clear_cache()
            f_minus = self.objective(theta_minus)

            grad[i] = (f_plus - f_minus) / (2 * epsilon)

        return grad

    def check_gradient(self, theta, epsilon=1e-6):
        self.reset_warm_start()
        self.clear_cache()
        analytical = self.grad(theta)

        numerical = self.numerical_gradient(theta, epsilon)

        diff = np.abs(analytical - numerical)
        rel_err = diff / (np.abs(numerical) + 1e-8)

        print(f"Gradient Comparison:")
        print(f"  Numerical: {analytical}")
        print(f"  Analytical: {numerical}")
        print(f"  Difference: {diff}")
        print(f"  Relative Error: {rel_err}")

        return analytical, numerical

