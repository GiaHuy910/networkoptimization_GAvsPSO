import matplotlib.pyplot as plt
import networkx as nx

def plot_history(history, embed=False, ax=None, color='blue', linewidth=2, title=None):
    if len(history) == 0:
        raise ValueError("plot_history(): history rỗng, không thể vẽ")

    if embed and ax is None:
        raise ValueError("plot_history(): embed=True nhưng không truyền ax")

    history = list(history)
    target_ax = ax if embed else plt.gca()
    target_ax.plot(history, color=color, linewidth=linewidth, label="Best fitness")
    target_ax.set_xlabel("Iteration")
    target_ax.set_ylabel("Fitness")
    if title:
        target_ax.set_title(title)
    target_ax.grid(True)
    target_ax.legend()
    if not embed:
        plt.show()


def plot_flow_network(edges, capacities, embed=False, ax=None):
    G = nx.DiGraph()
    for (u,v), c in zip(edges, capacities):
        G.add_edge(u, v, weight=c)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    if embed:
        if ax is None:
            raise ValueError("plot_flow_network(): embed=True nhưng không truyền ax")
        nx.draw(G, pos, with_labels=True, node_size=700,
                node_color='lightblue', arrowsize=20, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    else:
        plt.figure(figsize=(8,5))
        nx.draw(G, pos, with_labels=True, node_size=700,
                node_color='lightblue', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
