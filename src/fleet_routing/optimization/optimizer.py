"""BFGS optimizer and diagnostics (active implementation from cell 1)."""
import gc
import numpy as np
import psutil
from scipy.sparse import issparse
from fleet_routing.model.policy import (extract_state_action_pairs, compute_softmax_policy_by_action_features,
                    compute_kappa, compute_R)

class GradientOptimizer:
    """
    Combined objective and gradient computation
    """
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
                 # Fixed-point solver parameters
                 fixed_m_states=None,
                 tol_m=1e-6,
                 tol_mu=1e-6,
                 max_iter=100000,
                 method='hybrid',
                 restart_interval=30,
                 momentum=0.9,
                 lr=0.5,
                 phase1_tol=1e-4,
                 heuristic=None):
        
        # Store parameters
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
        self.solver_fixed_point = solve_fixed_point
        self.solver_grad = solve_sensitivities
        self.heuristic = heuristic
        
        # Fixed-point parameters
        self.fixed_m_states = fixed_m_states
        self.tol_m = tol_m
        self.tol_mu = tol_mu
        self.max_iter = max_iter
        self.method = method
        self.restart_interval = restart_interval
        self.momentum = momentum
        self.lr = lr
        self.phase1_tol = phase1_tol

        
        # Network information
        self.n = len(lambda_vec)
        self.row_idx, self.col_idx = extract_state_action_pairs(outgoing_dict)
        
        # Warm start (instance variables)
        self.mu_last = init_mu.copy()
        self.m_last = init_m.copy()
        self.init_mu = init_mu.copy()
        self.init_m = init_m.copy()
        
        # Current state (instance variables)
        self.current_state = {}
        
        # History
        self.theta_hist = []
        self.R_hist = []
        self.m_hist = []
        self.mu_hist = []
        self.state_hist = []  # Full state history
        self.grad_hist = []           # Added
        self.grad_norm_hist = []      # Added
        
        # History of accepted optimizer iterations
        self.iter_theta_hist = []
        self.iter_R_hist = []
        self.iter_grad_norm_hist = []
        
        # Cache
        self._cache_theta = None
        self._cache_obj = None
        self._cache_grad = None
    
    def reset_warm_start(self):
        """Reset the warm start to its initial state"""
        self.mu_last = self.init_mu.copy()
        self.m_last = self.init_m.copy()
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache_theta = None
        self._cache_obj = None
        self._cache_grad = None
    
    def clear_history(self):
        """Clear the history"""
        self.theta_hist = []
        self.R_hist = []
        self.m_hist = []
        self.mu_hist = []
        self.state_hist = []  # Added
        self.grad_hist = []           # Added
        self.grad_norm_hist = []      # Added
        self.iter_theta_hist = []
        self.iter_R_hist = []
        self.iter_grad_norm_hist = []
    
    
    def _compute_fixed_point(self, theta_vec):
        """Compute the fixed point"""
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
            verbose=True,
            heuristic=self.heuristic
        )
        
        # Update the warm start
        self.mu_last = mu.copy()
        self.m_last = m.copy()
        
        return mu, m, pi_dense
    
    def _build_feature_matrix(self, k):
        """Build the feature matrix for parameter k"""
        x_r = np.zeros((self.n, self.n), dtype=np.float64)
        for s, actions in self.outgoing_dict.items():
            for a in actions:
                x_r[s, a] = self.action_features[k][a]
        return x_r
    
        
    def _compute_objective(self, theta):
        """Compute only the objective"""
        if (self._cache_theta is not None and 
            np.array_equal(theta, self._cache_theta)):
            return

        print(f"[Compute f] theta = {theta}")

        # Compute the fixed point
        mu, m, pi_dense = self._compute_fixed_point(theta)

        # Compute R
        kappa = compute_kappa(self.v, self.c, self.Q, self.beta)
        R_pi = compute_R(mu, self.tau, self.omega, m, kappa, self.beta)

        # Cache objective-related values only
        self._cache_theta = theta.copy()
        self._cache_obj = -R_pi
        self._cache_mu = mu
        self._cache_m = m
        self._cache_pi = pi_dense
        self._cache_kappa = kappa
        self._cache_grad = None  # Mark the gradient as not yet evaluated
        
    def _compute_gradient(self, theta):
        """Compute the gradient, assuming f has already been evaluated"""
        if (self._cache_theta is not None and 
            np.array_equal(theta, self._cache_theta) and
            self._cache_grad is not None):
            return

        # Ensure f has been evaluated
        if not np.array_equal(theta, self._cache_theta):
            self._compute_objective(theta)

        print(f"[Compute grad] theta = {theta}")

        # Retrieve cached values
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

        # Compute the gradient
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
        
        # Record history
        self.theta_hist.append(theta.copy())
        self.R_hist.append(-self._cache_obj)
        self.m_hist.append(m.copy())
        self.mu_hist.append(mu.copy())
        self.grad_hist.append(-grad_R.copy())       # Added
        self.grad_norm_hist.append(grad_norm)       # Added
        self.state_hist.append({
            'mu': mu.copy(),
            'm': m.copy(),
            #'pi': pi_dense.copy(),
            'kappa': kappa.copy(),
            'N': N,
            'D': D,
            'dN_dm': dN_dm.copy(),
            'dN_dmu': dN_dmu.copy(),
            'dD_dm': dD_dm.copy(),
            'dD_dmu': dD_dmu.copy(),
            'R': -self._cache_obj,
            'grad': -grad_R.copy(),       # Added
            'grad_norm':  grad_norm       # Added
        })
        
        self._cache_pi = None  # Release the reference
        del pi_dense
        
        gc.collect()

        mem_gb = psutil.Process().memory_info().rss / 1e9
        print(f"  Memory: {mem_gb:.2f} GB. ||grad|| = {grad_norm:.2e}")
        
        
    def objective(self, theta):
        """Return the objective value"""
        theta = np.asarray(theta, dtype=float)
        self._compute_objective(theta)
        return self._cache_obj


    def grad(self, theta):
        """Return the gradient"""
        theta = np.asarray(theta, dtype=float)
        self._compute_gradient(theta)
        return self._cache_grad
    
    
    def optimize(self, theta_init, method='BFGS', maxiter=100, bounds=None, gtol=1e-5):
        """Run optimization"""
        from scipy.optimize import minimize
        
        def callback(xk):
            """Called after each BFGS iteration"""
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
        self.reset_warm_start()  # ← Reset to ensure reproducibility
        
        if bounds is not None and method == 'BFGS':
            method = 'L-BFGS-B'
        
        result = minimize(
            fun=self.objective,
            x0=theta_init,
            method=method,
            jac=self.grad,
            callback=callback,
            bounds=bounds,
            options={'disp': True, 'maxiter': maxiter, 'gtol': gtol}
        )
        
        return result
    

    
    def optimize_numerical(self, theta_init, maxiter=50, gtol=1e-5, epsilon=1e-6):
        """Optimize using numerical gradients"""
        from scipy.optimize import minimize

        self.clear_history()
        self.clear_cache()
        self.reset_warm_start()

        def callback(xk):
            self.iter_theta_hist.append(xk.copy())
            print(f'\n[Numerical BFGS Iter {len(self.iter_theta_hist)}] theta = {xk}')

        result = minimize(
            fun=self.objective,
            x0=theta_init,
            callback=callback,
            method='BFGS',
            options={
                'disp': True, 
                'maxiter': maxiter, 
                'gtol': gtol,
                'eps': epsilon  # Finite-difference step size
            }
        )
        return result
    
    
    def optimize_nelder_mead(self, theta_init, maxiter=500, xatol=1e-6, fatol=1e-6):
        """Optimize with Nelder-Mead (derivative-free)"""
        from scipy.optimize import minimize

        self.clear_history()
        self.clear_cache()
        self.reset_warm_start()

        def callback(xk):
            self.iter_theta_hist.append(xk.copy())
            print(f'\n[Nelder-Mead Iter {len(self.iter_theta_hist)}] theta = {xk}')

        result = minimize(
            fun=self.objective,
            x0=theta_init,
            callback=callback,
            method='Nelder-Mead',
            options={
                'disp': True,
                'maxiter': maxiter,
                'xatol': xatol,  # Convergence tolerance for x
                'fatol': fatol   # Convergence tolerance for f(x)
            }
        )
        return result
    
    def numerical_gradient(self, theta, epsilon=1e-6):
        """Compute numerical gradients, resetting the warm start each time"""
        grad = np.zeros_like(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            
            # Reset and compute f_plus
            self.reset_warm_start()
            self.clear_cache()
            f_plus = self.objective(theta_plus)
            
            # Reset and compute f_minus
            self.reset_warm_start()
            self.clear_cache()
            f_minus = self.objective(theta_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return grad
    
    def check_gradient(self, theta, epsilon=1e-6):
        """Compare analytical and numerical gradients"""
        self.reset_warm_start()
        self.clear_cache()
        analytical = self.grad(theta)
        
        numerical = self.numerical_gradient(theta, epsilon)
        
        diff = np.abs(analytical - numerical)
        rel_err = diff / (np.abs(numerical) + 1e-8)
        
        print(f"Gradient comparison:")
        print(f"  Analytical: {analytical}")
        print(f"  Numerical: {numerical}")
        print(f"  Difference: {diff}")
        print(f"  Relative error: {rel_err}")
        
        return analytical, numerical
