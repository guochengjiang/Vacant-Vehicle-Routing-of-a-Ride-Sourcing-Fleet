import numpy as np


def compute_kappa(v, c, q, beta):
    return np.sum((v - beta * c) * q, axis=1)

def compute_R(mu, tau, omega, m, kappa, beta):
    num = np.sum(mu * (-beta * tau + kappa * m))
    denom = np.sum(mu * (tau + omega * m))
    return num / denom if denom > 0 else -np.inf