from matplotlib import pyplot as plt
import networkx as nx



def draw_tour(title: str, adj_matrix: list[list[list[int]]], tour: tuple[int]):
    """Draw Graph with Optional Highlighted Tour

    Args:
        title (str): Title of the Graph
        adj_matrix (List[List[List[int]]]): List of multiple adjacency matrix
        tour (Tuple[int], optional): A tour (optimal path) to highlight. Defaults to None.
    """
    location_name = [str(i) for i in range(len(adj_matrix[0]))]
    G = nx.Graph()
    G.add_nodes_from(location_name)
    for i in range(len(location_name)):
        for j in range(i + 1, len(location_name)):
            edge_label = ""
            for k, label in enumerate(["d"]):
                edge_label += f"{label} = {adj_matrix[k][i][j]}\n"
            G.add_edge(
                location_name[i], location_name[j],
                weight=adj_matrix[k][i][j], label=edge_label)

    fig = plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, k=0.5, seed=128)

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_size=3000,
            node_color='skyblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.3)

    # Highlight edges in the tour
    
    tour_edges = [*zip(tour, tour[1:] + [tour[0]])]
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges,
                               edge_color='purple', width=2.5)

    plt.title(title)
    plt.show()
    return G
