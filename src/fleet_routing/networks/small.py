"""Synthetic bidirectional grid, not the standard Sioux Falls benchmark.

Unlike the legacy generator, successors leave the current link's head and all
travel times are hours. This intentionally changes legacy small-network results.
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path
from fleet_routing.networks.shanghai import NetworkData


def build_small_network(width=6, height=4, seed=2025, speed=40.0):
    """Return NetworkData and topology metadata; length units match speed/hour."""
    if width < 2 or height < 2 or speed <= 0:
        raise ValueError('Grid dimensions must be >= 2 and speed must be positive')
    rng = np.random.default_rng(seed)
    coords = np.array([(x, y) for y in range(height) for x in range(width)], float)
    coords += rng.normal(0, .05, coords.shape)
    links = []
    for y in range(height):
        for x in range(width):
            u = y * width + x
            for v in ([u + 1] if x + 1 < width else []) + ([u + width] if y + 1 < height else []):
                links.extend([(u, v), (v, u)])
    links = np.asarray(links)
    tail, head = links.T
    lengths = np.linalg.norm(coords[tail] - coords[head], axis=1)
    tau = lengths / speed
    adjacency = np.full((len(coords), len(coords)), np.inf)
    np.fill_diagonal(adjacency, 0)
    adjacency[tail, head] = tau
    distance = shortest_path(adjacency, directed=True)
    c = distance[head[:, None], tail[None, :]] + tau[None, :]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    attraction = np.exp(-np.linalg.norm(coords[head] - center, axis=1))
    lam = 25 + 95 * attraction + rng.uniform(0, 5, len(links))
    Q = rng.uniform(.5, 1.5, c.shape) * (1 + 3 * attraction)[None, :]
    Q /= Q.sum(axis=1, keepdims=True)
    outgoing = {i: np.flatnonzero(tail == head[i]).tolist() for i in range(len(links))}
    network = NetworkData(c=c, v=35 + 35 * speed * c, Q=Q, lambda_vec=lam,
                          tau=tau, omega=np.sum(Q * c, axis=1),
                          features=lam[None, :], init_dist=np.full(len(links), 1 / len(links)),
                          outgoing_links=outgoing)
    return network, {'links': links, 'coordinates': coords, 'lengths': lengths}
