import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_graph(title: str, location_name: list[str], adj_matrix: list[list[list[int]]], edge_labels: list[str]):
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


def generate_random_symmetric_matrix(size: int = 8, min: int = 0, max: int = 9, seed: int = None):
    """Generate a random distance matrix.

    Args: 
        size (int): Number of nodes
        min (int): Minimum distance
        max (int): Maximum distance
        seed (int): Random seed

    Returns:
        list[list[int]]: Distance matrix
    """

    if seed:
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        rng = np.random.default_rng()

    triangle = np.triu(rng.integers(min, max, size=(size, size)), k=1)
    sym = triangle + triangle.T

    np.fill_diagonal(sym, 0)

    return sym
    # # Initialize an empty matrix
    # matrix = [[0 for _ in range(size)] for _ in range(size)]

    # # Fill the upper triangle (excluding the diagonal) and reflect it to the lower triangle
    # for i in range(size):
    #     for j in range(i+1, size):
    #         value = random.randint(min, max)  # Random numbers between 0 and 9
    #         matrix[i][j] = value
    #         matrix[j][i] = value

    # return matrix


def make_sparse(matrix: list[list[int]], dense_ratio: float):
    """Make matrix sparse by randomly changing element to zero

    Args:
        matrix (list[list[int]]): Input Matrix
        dense_ratio (float): Target Dense Ratio (Including Diagonal)

    Returns:
        list[list[int]]: Output Matrix
    """

    n = len(matrix)

    u_ratio = dense_ratio * n ** 2 / (n ** 2 - n)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() > u_ratio:
                matrix[i][j] = 0
                matrix[j][i] = 0

    return matrix


def make_sparse_with_seed(matrix: list[list[int]], dense_ratio: float, seed: str):
    random.seed(seed)
    return make_sparse(matrix, dense_ratio)


def print_matrix(matrix: list[list[int]]):
    for row in matrix:
        print(row)


def create_with_seed(seed: str, size: int, min: int, max: int):
    # Set seed
    random.seed(seed)
    return generate_random_symmetric_matrix(size, min, max)


def create_sparse_with_seed(seed: str, size: int, min: int, max: int, dense_ratio: float):
    random.seed(seed)
    m = generate_random_symmetric_matrix(size, min, max)
    return make_sparse(m, dense_ratio)


def create_qap_input(size: int, min: int, max: int, dense_ratio: float, extra_seed="none"):
    base_seed = f"size={size};min={min};max={
        max};dense_ratio={dense_ratio};extra_seed={extra_seed}"

    distance_matrix = create_sparse_with_seed(
        f"problem=qap_distance;{base_seed}", size, min, max, dense_ratio)

    interaction_matrix = create_sparse_with_seed(
        f"problem=qap_interaction;{base_seed}", size, min, max, dense_ratio)

    return distance_matrix, interaction_matrix
