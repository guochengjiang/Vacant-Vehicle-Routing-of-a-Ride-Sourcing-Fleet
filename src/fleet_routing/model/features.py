"""Feature normalization and optional direct-profit experiment."""
import numpy as np


def normalize_features(features):
    features = np.atleast_2d(np.asarray(features, dtype=float))
    minima = features.min(axis=1)
    ranges = np.ptp(features, axis=1)
    normalized = (features - minima[:, None]) / (ranges[:, None] + 1e-8)
    return normalized, minima, ranges


def direct_profit_features(network, m, beta=30):
    """Cell 73 formula; m must be one explicitly selected matching vector."""
    m = np.asarray(m)
    if m.shape != (network.n_links,):
        raise ValueError("Select one m vector, not the entire m_hist")
    fare = m * np.sum(network.Q * network.v, axis=1)
    cost = beta * (network.tau + m * network.omega)
    return np.array([fare - cost])
