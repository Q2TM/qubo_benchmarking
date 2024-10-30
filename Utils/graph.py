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


def calculate_cost(
    assign: List[int],
    distance_matrix: List[List[int]],
    interaction_matrix: List[List[int]],
):
    """Calculate cost when assign facility i on site assign[i]

    Args:
        assign (List[int]): assign facility i on site assign[i]
        distance_matrix (List[List[int]]): Adjacency matrix of distance
        interaction_matrix (List[List[int]]): Adjacency matrix of interaction

    Returns:
        int: Total cost
    """
    size = len(assign)

    total_cost = 0
    for i in range(size):
        for j in range(size):
            total_cost += distance_matrix[i][j] * \
                interaction_matrix[assign[i]][assign[j]]

    return total_cost


def assign_facilities(
    assign: List[int],
    location_name: List[str],
    facility_name: List[str],
    distance_matrix: List[List[int]],
    interaction_matrix: List[List[int]],
):
    size = len(assign)

    # Create Three empty matrix with size of size*size
    # distance_matrix = [[0 for _ in range(size)] for _ in range(size)]
    mapped_interaction_matrix = [[0 for _ in range(size)] for _ in range(size)]
    cost_matrix = [[0 for _ in range(size)] for _ in range(size)]

    node_labels = []
    for i in range(size):
        print(f"{location_name[i]} is assigned to {facility_name[assign[i]]}")
        node_labels.append(
            f"{facility_name[assign[i]]}\nOn {location_name[i]}")

    total_cost = calculate_cost(assign, distance_matrix, interaction_matrix)

    print(f"Total Cost: {total_cost}")

    draw_graph(
        f"Assignment Result (Total Cost: {total_cost})", node_labels, [
            distance_matrix, mapped_interaction_matrix, cost_matrix], ["Distance", "Interaction", "Cost"])


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
