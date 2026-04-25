import numpy as np
import heapq
from collections import defaultdict

def allpairs_shortest_time_on_nodes(n_nodes, adjacency):
    INF = 1e18
    dist = np.full((n_nodes, n_nodes), INF, dtype=float)
    for src in range(n_nodes):
        d = np.full(n_nodes, INF, dtype=float)
        d[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            du, u = heapq.heappop(pq)
            if du != d[u]:
                continue
            for v, w in adjacency[u]:
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist[src, :] = d
    return dist


def build_dense_link_to_link_c(directed_links, time_min_per_link, n_nodes, include_linkj_travel=True):
    adjacency = [[] for _ in range(n_nodes)]
    for (u, v), w in zip(directed_links, time_min_per_link):
        adjacency[u].append((v, w))

    node_dist = allpairs_shortest_time_on_nodes(n_nodes, adjacency)

    L = len(directed_links)
    tail_of = np.array([u for (u, v) in directed_links])
    head_of = np.array([v for (u, v) in directed_links])
    c = np.zeros((L, L), dtype=float)

    add_j = time_min_per_link if include_linkj_travel else 0.0
    for i in range(L):
        vi = head_of[i]

        c[i, :] = node_dist[vi, tail_of] + add_j

    return c


def build_sioux_falls_like(seed=42, seed2=42,
                           grid_w=6, grid_h=4,
                           speed_mph=40.0,  # 与 c/40 一致
                           initial_charge=35.0,
                           unit_fare=35.0,
                           alpha=0.8,
                           M=10000,
                           b=0.2):
    rng = np.random.default_rng(seed)

    def nid(r, c):
        return r * grid_w + c

    coords = {}
    for r in range(grid_h):
        for c in range(grid_w):
            coords[nid(r, c)] = (c + 0.05 * rng.normal(), r + 0.05 * rng.normal())

    undirected_edges = []

    for r in range(grid_h):
        for c in range(grid_w - 1):
            undirected_edges.append((nid(r, c), nid(r, c + 1)))

    for r in range(grid_h - 1):
        for c in range(grid_w):
            undirected_edges.append((nid(r, c), nid(r + 1, c)))

    directed_links = []  # (u -> v)
    for u, v in undirected_edges:
        directed_links.append((u, v))
        directed_links.append((v, u))
    L = len(directed_links)  # 76

    def euclid(a, b):
        ax, ay = coords[a]
        bx, by = coords[b]
        return np.hypot(ax - bx, ay - by)

    link_length = np.array([euclid(u, v) for (u, v) in directed_links])  # (L,)

    link_length *= (1.0 + 0.05 * rng.normal(size=L))
    link_length = np.maximum(link_length, 1e-3)

    time_min_per_link = 60.0 * link_length / speed_mph

    head_of = np.array([v for (_, v) in directed_links])
    tail_of = np.array([u for (u, _) in directed_links])

    node_to_out_links = defaultdict(list)
    for j, (u, v) in enumerate(directed_links):
        node_to_out_links[v].append(j)

    outgoing_dict = {}
    for i, (u, v) in enumerate(directed_links):
        outgoing = node_to_out_links[v]

        outgoing_dict[i] = outgoing.copy()

    outgoing_links = outgoing_dict

    n_nodes = grid_w * grid_h
    c = build_dense_link_to_link_c(
        directed_links=directed_links,
        time_min_per_link=time_min_per_link,
        n_nodes=n_nodes,
        include_linkj_travel=True
    )
    c_time = c / 40.0

    cbd_rows = range(max(0, grid_h // 2 - 1), min(grid_h, grid_h // 2 + 1))
    cbd_cols = range(max(0, grid_w // 2 - 1), min(grid_w, grid_w // 2 + 1))
    cbd_nodes = {nid(r, c) for r in cbd_rows for c in cbd_cols}

    is_cbd_link = np.array([(u in cbd_nodes) or (v in cbd_nodes) for (u, v) in directed_links])

    tau = np.where(is_cbd_link, 0.05, 0.20)

    dest = np.zeros((L, L), dtype=float)

    cbd_sink_mask = np.array([head in cbd_nodes for (_, head) in directed_links])
    non_cbd_sink_mask = ~cbd_sink_mask

    for i in range(L):
        if is_cbd_link[i]:
            w_cbd = 0.55 + 0.10 * rng.random()  # 0.55~0.65
        else:
            w_cbd = 0.75 + 0.10 * rng.random()  # 0.75~0.85（
        w_non = 1.0 - w_cbd

        cbd_candidates = np.where(cbd_sink_mask)[0]
        non_candidates = np.where(non_cbd_sink_mask)[0]

        if len(cbd_candidates) > 0:
            x = rng.random(len(cbd_candidates))
            x = x / x.sum()
            dest[i, cbd_candidates] = w_cbd * x
        if len(non_candidates) > 0:
            y = rng.random(len(non_candidates))
            y = y / y.sum()
            dest[i, non_candidates] = w_non * y

    dest = dest / dest.sum(axis=1, keepdims=True)

    omega = compute_expected_time(dest, c_time)


    rng = np.random.RandomState(seed2)

    lambda_vec = np.where(is_cbd_link, 120.0, 25.0)
    lambda_vec = lambda_vec + rng.uniform(-5, 5, size=lambda_vec.shape)

    v = initial_charge + unit_fare * c
    v = np.round(v, 2)

    mu0 = np.ones(L) / L
    features = [lambda_vec.copy()]
    init_m = np.full(L, 0.2, dtype=np.float32)
    init_mu = np.full(L, 1.0 / L, dtype=np.float32)

    params = dict(
        alpha=alpha,
        M=M,
        b=b,
        initial_charge=initial_charge,
        unit_fare=unit_fare,
        speed_mph=speed_mph
    )

    link_index = np.arange(L)
    link_tail = tail_of
    link_head = head_of

    return {

        "c": c,
        "c_time": c_time,
        "tau": tau,
        "outgoing_dict": outgoing_dict,
        "outgoing_links": outgoing_links,
        "v": v,
        "dest": dest,
        "omega": omega,
        "lambda_vec": lambda_vec,
        "features": features,
        "mu0": mu0,
        "init_m": init_m,
        "init_mu": init_mu,
        "M": M,
        "b": b,
        "alpha": alpha,
        "n_links": L,
        "directed_links": directed_links,  # list of (u->v)
        "link_tail": link_tail,
        "link_head": link_head,
        "coords": coords,
        "params": params,
    }

def compute_expected_time(Q, c):
    #return np.array(Q.multiply(c).sum(axis=1)).flatten()
    return np.sum(Q * c, axis=1)