import pandas as pd
from Algorithm.bat_algo.Bat_algo_max_flow import bat_max_flow
import matplotlib.pyplot as plt
path = r'Graph/Graphs_edges.csv'
df = pd.read_csv(path)

best_fit,best_paths_with_flow,best_edge_load,paths,history = bat_max_flow(
    df, source='s', sink='t',
    max_iterations=200, n_bats=5,
    f_min=0.0, f_max=2.0, A_min=1.0, A_max=2.0,
    r_min=0.0, r_max=1.0,
    alpha=0.9, gamma=0.9,
    max_paths=100, verbose=False,drawing=True)
