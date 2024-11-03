from random import randint
from typing import List

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(title: str, location_name: List[str], adj_matrix: List[List[List[int]]], edge_labels: List[str]):
    """Draw Graph

    Args:
        title (str): Title of the Graph
        location_name (List[str]): Location/Vertice/Node Names
        adj_matrix (List[List[List[int]]]): List of multiple adjacency matrix
        edge_labels (List[str]): Labels for each adjacency matrix, must be same length with adj_matrix
    """

    G = nx.Graph()
    G.add_nodes_from(location_name)
    for i in range(len(location_name)):
        for j in range(i+1, len(location_name)):
            edge_label = ""
            for k, label in enumerate(edge_labels):
                edge_label += f"{label} = {adj_matrix[k][i][j]}\n"
            G.add_edge(
                location_name[i], location_name[j],
                weight=adj_matrix[k][i][j], label=edge_label)

    fig = plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, k=0.5, seed=128)
    nx.draw(G, pos, with_labels=True, node_size=3000,
            node_color='skyblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.3)
    plt.title(title)
    plt.show()


def generate_random_symmetric_matrix(size=8, min=0, max=9):
    # Initialize an empty matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    # Fill the upper triangle (excluding the diagonal) and reflect it to the lower triangle
    for i in range(size):
        for j in range(i+1, size):
            value = randint(min, max)  # Random numbers between 0 and 9
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix
