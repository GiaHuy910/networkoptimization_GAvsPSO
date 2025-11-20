import pandas as pd
from collections import deque

def adj_list_from_file(df):
    adj_list = {}
    for _, row in df.iterrows():
        u = str(row["v_from"])
        v = str(row["v_to"])
        cap = int(row["weight"])

        if u not in adj_list:
            adj_list[u] = []
        if v not in adj_list:
            adj_list[v] = []  # đảm bảo node xuất hiện

        adj_list[u].append((v, cap))
    return adj_list

def bfs(residual,source,sink,parent):
    visited = set()
    queue = deque([source])
    visited.add(source)

    while queue:
        u = queue.popleft()

        for v in residual[u]:
            if v not in visited and residual[u][v] > 0:  # còn dư
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False

def edmond_karp(adj_list,source,sink):
    residual = {}
    history=[]

    #cạnh rỗng
    for u in adj_list:
        if u not in residual:
            residual[u] = {}
        for v, cap in adj_list[u]:
            if v not in residual:
                residual[v] = {}
    #thêm cap thuận và ngược 0
    for u in adj_list:
        for v, cap in adj_list[u]:
            residual[u][v] = cap           # cạnh thuận
            if u not in residual[v]:       # cạnh ngược
                residual[v][u] = 0.0
    parent = {}
    max_flow = 0.0

    # Lặp BFS để tìm đường tăng
    while bfs(residual, source, sink, parent):
        # Tìm bottleneck
        v = sink
        bottleneck = float("inf")

        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, residual[u][v])
            v = u

        # Cập nhật residual graph
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= bottleneck
            residual[v][u] += bottleneck
            v = u

        max_flow += bottleneck
        history.append(max_flow)

    return max_flow, residual,history

def print_flows(adj, residual):
    print("Last Flow (flow / capacity):")
    for u in adj:
        for v, cap in adj[u]:
            used = residual[v][u]  # luồng đã đẩy qua = residual reverse
            print(f"{u} -> {v} : {used} / {cap}")

def run_edmond_karp_algo(dataframe,source,sink,verbose=False):
    adj = adj_list_from_file(dataframe)
    maxflow, residual,history = edmond_karp(adj, source, sink)
    
    if verbose:
        print(f"MAX FLOW = {maxflow}")
        print_flows(adj, residual)
    return maxflow,residual,history
