import random
import math
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt

#đọc dataframe ra danh sách kề adj
def adj_list_from_df(df):
    adj = defaultdict(list)
    for _, row in df.iterrows():
        u = row.v_from
        v = row.v_to
        w = float(row.weight)
        adj[u].append((v, w))
        if v not in adj:
            adj[v] = adj[v]
    return dict(adj)

#build edges từ adj
def edges_from_adj(adj):
    edges = {}
    for u, neigh in adj.items():
        for v, w in neigh:
            edges[(u, v)] = float(w)
    return edges

# tìm path đơn giản với bfs
def find_simple_paths(adj, source, sink, max_paths=50, max_depth=20):
    paths = []
    stack = [(source, [source])]
    while stack and len(paths) < max_paths:
        node, path = stack.pop()
        if len(path) > max_depth:
            continue
        if node == sink:
            paths.append(list(path))
            continue
        for v, _ in adj.get(node, []):
            if v in path:
                continue
            stack.append((v, path + [v]))
    return paths

# build incidence: for each path which edges it uses
def build_path_edge_map(paths):
    path_edges = []
    for p in paths:
        edges = []
        for i in range(len(p)-1):
            edges.append((p[i], p[i+1]))
        path_edges.append(edges)
    return path_edges

#hàm sửa flow
def repair_flows_by_edge_scaling(x, path_edges, edge_capacity, eps=1e-9, max_iter=100):
    x = [max(0.0, val) for val in x]
    n_paths = len(x)
    for it in range(max_iter):
        # compute loads
        edge_load = defaultdict(float)
        for k in range(n_paths):
            for e in path_edges[k]:
                edge_load[e] += x[k]
        violated = False
        for e, load in edge_load.items():
            cap = edge_capacity.get(e, 0.0)
            if load > cap + eps and load > 0:
                violated = True
                scale = cap / load if load > 0 else 0.0
                # scale down all paths that use e
                for k in range(n_paths):
                    if e in path_edges[k]:
                        x[k] *= scale
        if not violated:
            break
    # final cleanup: ensure no tiny negatives etc
    x = [max(0.0, float(val)) for val in x]
    return x

# compute fitness (total flow)
def total_flow(x):
    return sum(x)

#các hàm khởi tạo
def initialize_bats(n_bats, n_paths, edge_capacity, path_edges):
    bats = []
    for _ in range(n_bats):
        # random initial flows small fraction of capacities
        x = [random.random() * 0.5 for _ in range(n_paths)]
        x = repair_flows_by_edge_scaling(x, path_edges, edge_capacity)
        bats.append(x)
    return bats

def initialize_loudness(n_bats,A_min,A_max):
    loudness=[random.uniform(A_min, A_max) for _ in range(n_bats)]
    return loudness
def initialize_frequency(n_bats,f_min,f_max):
    frequency=[random.uniform(f_min, f_max) for _ in range(n_bats)]
    return frequency
def initialize_r(n_bats,r_min,r_max):
    r=[random.uniform(r_min, r_max) for _ in range(n_bats)]
    r_0=r.copy()
    return r,r_0
def initialize_velocities(n_bats, n_paths, scale=0.1):
    return [[(random.random() - 0.5) * scale for _ in range(n_paths)] for _ in range(n_bats)]

#hàm tìm fitness tốt nhất
def find_best_fitness(n_bats,fitnesses,bats):
    best_idx = max(range(n_bats), key=lambda i: fitnesses[i])
    best_x = bats[best_idx].copy()
    best_fit = fitnesses[best_idx]
    return best_idx,best_x,best_fit

#hàm bat maxflow
def bat_max_flow(dataframe, source='s', sink='t',
                max_iterations=500, n_bats=30,
                f_min=0, f_max=2, A_min=0.5, A_max=2.0,
                r_min=0.0, r_max=1.0, alpha=0.9, gamma=0.9,
                max_paths=50, verbose=True,drawing=False):
    adj=adj_list_from_df(dataframe)
    # prep
    edges = edges_from_adj(adj)
    paths = find_simple_paths(adj, source, sink, max_paths=max_paths)
    if not paths:
        raise ValueError("not found any path")
    path_edges = build_path_edge_map(paths)
    n_paths = len(paths)

    # map edge capacities
    edge_capacity = dict(edges)

    # initialize population
    bats = initialize_bats(n_bats, n_paths, edge_capacity, path_edges)
    velocities = initialize_velocities(n_bats, n_paths, scale=0.5)
    frequencies = initialize_frequency(n_bats,f_min,f_max)
    loudness = initialize_loudness(n_bats,A_min,A_max)
    r,r0 = initialize_r(n_bats,r_min,r_max)

    # fitnesses tất cả dơi
    fitnesses = [total_flow(repair_flows_by_edge_scaling(b, path_edges, edge_capacity)) for b in bats]
    # best
    best_idx,best_x,best_fit=find_best_fitness(n_bats,fitnesses,bats)

    history = [best_fit]
    #vẽ biểu đồ
    if drawing:
        plt.ion()
        fig, ax = plt.subplots()

    for t in range(1, max_iterations+1):
        for i in range(n_bats):
            # update frequency
            frequencies[i] = f_min + (f_max - f_min) * random.random()
            # velocity update (like PSO influenced by best)
            velocities[i] = [velocities[i][k] + (bats[i][k] - best_x[k]) * frequencies[i]
                             for k in range(n_paths)]
            # propose new solution
            new_x = [bats[i][k] + velocities[i][k] for k in range(n_paths)]

            # local search: with probability > r[i], perform a local walk around best
            if random.random() > r[i]:
                avg_L = sum(loudness) / len(loudness)
                # perturb best to create local candidate
                local = best_x.copy()
                for k in range(n_paths):
                    step = (random.random() - 0.5) * avg_L * (edge_capacity.get(path_edges[k][0], 1.0) if path_edges[k] else 1.0)
                    local[k] = max(0.0, local[k] + step)
                new_x = local

            # repair candidate to satisfy capacity constraints
            new_x = repair_flows_by_edge_scaling(new_x, path_edges, edge_capacity)
            new_fit = total_flow(new_x)

            # Accept with condition (like Bat)
            if (random.random() < loudness[i]) and (new_fit > fitnesses[i]):
                bats[i] = new_x
                velocities[i] = velocities[i]
                fitnesses[i] = new_fit
                loudness[i] *= alpha
                r[i] = r0[i] * (1 - math.exp(-gamma * t))

            # update global best
            if new_fit > best_fit:
                best_fit = new_fit
                best_x = new_x.copy()

        history.append(best_fit)
        #vẽ biểu đồ liên tục
        if drawing:
            ax.clear()
            ax.plot(history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.set_title("Bat algo for 8 queens")
            plt.pause(0.01) 
        if verbose and (t % max(1, max_iterations//10) == 0):
            print(f"Iter {t}/{max_iterations} — best flow = {best_fit:.4f} — paths={n_paths}")
            
    if drawing:
        plt.ioff()
        plt.show()
    # Prepare result: convert best_x into edge flows and path listing
    # compute per-edge load
    edge_load = defaultdict(float)
    for k, flow in enumerate(best_x):
        for e in path_edges[k]:
            edge_load[e] += flow

    # build readable best_flow per path and per edge
    best_paths_with_flow = [(paths[k], round(best_x[k], 6)) for k in range(n_paths) if best_x[k] > 1e-9]
    best_edge_load = {e: round(edge_load[e], 6) for e in edge_capacity.keys()}

    return best_fit,best_paths_with_flow,best_edge_load,history

