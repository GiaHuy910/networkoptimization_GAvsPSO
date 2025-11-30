import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import pandas as pd
import matplotlib.pyplot as plt

from Analyse import run_bat,plot_history, plot_flow_network

def build_edges_and_capacity_from_adj_list(adj_list):
    # Tạo danh sách rỗng để lưu các cạnh và trọng số (capacity) tương ứng
    edges = []
    capacities = []

    # Duyệt qua từng đỉnh u và danh sách kề của nó
    # adj_list[u] = [(v1, w1), (v2, w2), ...]
    for u, neighs in adj_list.items():
        # Duyệt qua từng cặp (v, w) trong danh sách kề của u
        # v: đỉnh kề
        # w: trọng số / capacity của cạnh (u → v)
        for v, w in neighs:
            # Thêm cặp cạnh (u, v) vào danh sách edges
            edges.append((u, v))

            # Thêm trọng số tương ứng vào danh sách capacities
            capacities.append(w)

    # Trả về hai danh sách:
    # edges     → danh sách các cạnh dạng (u, v)
    # capacities → danh sách trọng số tương ứng với từng cạnh
    return edges, capacities

def run_analyse_bat_maxflow(csv_path, source='s', sink='t', show_plot=True):
    
    df = pd.read_csv(csv_path)
    print(df)
    best_val, best_paths, best_edges, history = run_bat(df, source=source, sink=sink)

    print("\nFinal Results:")
    print("Best flow:", best_val)
    print("Paths:", best_paths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("BAT Algorithm Max Flow", fontsize=18)

    # Vẽ hội tụ
    plot_history(history, embed=True, ax=ax1,
                 color='green', linewidth=2,
                 title="Convergence of BAT")

    # Vẽ flow network
    adj_list = {}
    for _, row in df.iterrows():
        u, v, w = row['v_from'], row['v_to'], row['weight']
        adj_list.setdefault(u, []).append((v, w))
        if v not in adj_list:
            adj_list[v] = []
    edges, capacities = build_edges_and_capacity_from_adj_list(adj_list)
    plot_flow_network(edges, capacities, embed=True, ax=ax2)
    
    if show_plot:
        plt.show()

    return best_val, best_paths, best_edges, history
