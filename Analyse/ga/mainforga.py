import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import pandas as pd
import matplotlib.pyplot as plt

from Analyse import run_ga,plot_history, plot_flow_network

def build_edges_and_capacity_from_adj_list(adj_list):
    
    edges = []
    capacities = []
    for u, neighs in adj_list.items():
        for v, w in neighs:
            edges.append((u, v))
            capacities.append(w)
    return edges, capacities

def run_analysis_ga_maxflow_(csv_path, source='s', sink='t', show_plot=True):
    
    df = pd.read_csv(csv_path)
    print(df)
    all_paths_with_flows, best_flow_value,generation_history,best_fitness_history=run_ga(df, source, sink)
    print("\nFinal Results:")
    print("Best flow:", best_flow_value)
    print("Paths:")

    for path_info in all_paths_with_flows:
        print(f"Flow: {path_info['flow']:.4f} | Route: {path_info['path_string']}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GA Algorithm Max Flow", fontsize=18)

    # Vẽ hội tụ
    plot_history(best_fitness_history, embed=True, ax=ax1,
                 color='blue', linewidth=2,
                 title="Convergence of GA")

    # Vẽ flow network
    adj_list = {}
    for _, row in df.iterrows():
        u, v, w = row['v_from'], row['v_to'], row['weight']
        adj_list.setdefault(u, []).append((v, w))
        if v not in adj_list:
            adj_list[v] = []
    edges, capacities = build_edges_and_capacity_from_adj_list(adj_list)
    plot_flow_network(edges, capacities, embed=True, ax=ax2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show_plot:
        plt.show()
    return  all_paths_with_flows, best_flow_value,generation_history,best_fitness_history

#run_analysis_ga_maxflow_(r'Graph/Graphs_edges.csv')
