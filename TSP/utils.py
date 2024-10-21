import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx

from itertools import permutations
from qiskit_optimization import QuadraticProgram
from dotenv import load_dotenv


load_dotenv()

QISKIT_TOKEN = os.getenv("QISKIT_TOKEN")


def randomize_cities(num_nodes: int):
    """Randomize the cities and return the pairwise distance matrix"""
    positions = np.random.rand(num_nodes, 2)

    fig, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], color='blue')

    for i, (x, y) in enumerate(positions):
        ax.text(x, y, f'N{i}', fontsize=12, ha='right')

    plt.title("Randomized cities")
    plt.grid(True)
    plt.show()

    # pairwise distance using Euclidean distance
    dist = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    return (dist * 10).astype(int)


def brute_force_tsp(weights: np.ndarray) -> tuple:
    """Brute force solution for the TSP"""
    size = weights.shape[0]

    min_cost = np.inf
    min_path = None

    for path in permutations(range(size)):
        cost = 0
        for i in range(size):
            cost += weights[path[i], path[(i+1) % size]]

        if cost < min_cost:
            min_cost = cost
            min_path = path

    return min_path, min_cost


def draw_graph(weights: np.ndarray) -> None:
    """Draw the graph of the TSP"""
    n = weights.shape[0]

    G = nx.Graph()
    G.add_nodes_from(np.arange(n))
    colors = ['r' for node in G.nodes()]
    pos = nx.spring_layout(G)

    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=weights[i, j])

    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600,
                     alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


def draw_tour(weights: np.ndarray, path: list) -> None:
    """Draw the tour of the TSP"""
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(weights.shape[0]))
    pos = nx.spring_layout(G)
    colors = ['r' for node in G.nodes()]
    n = len(path)
    for i in range(n):
        j = (i + 1) % n
        G.add_edge(path[i], path[j], weight=weights[path[i], path[j]])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, font_color="b", edge_labels=edge_labels)

def reorder(x: list, size: int):
    """Reorder the binary list to the order of the cities"""
    y = np.zeros(size)
    for i, v in enumerate(x):
        if v == 1:
            y[int(i) % size] = i // size
    return y


def make_tsp_qp(weights: np.ndarray) -> QuadraticProgram:
    """Create the Quadratic Program for the TSP"""
    qp = QuadraticProgram()

    size = weights.shape[0]

    cities = {}  # dictionary to store the city and order
    for i in range(size):
        for p in range(size):
            cities[f'x_{i}_{p}'] = qp.binary_var(f'x_{i}_{p}')

    # Objective function
    quadratic_matrix = {}
    for i in range(size):
        for j in range(size):
            if i != j:
                for p in range(size):
                    quadratic_matrix[(f'x_{i}_{p}', f'x_{j}_{
                                      (p+1) % size}')] = weights[i, j]

    qp.minimize(quadratic=quadratic_matrix)

    # Constraint 1: each city is visited exactly once
    for i in range(size):
        qp.linear_constraint(
            linear={f'x_{i}_{p}': 1 for p in range(size)}, sense='==', rhs=1)

    # Constraint 2: each city is left exactly once
    for p in range(size):
        qp.linear_constraint(
            linear={f'x_{i}_{p}': 1 for i in range(size)}, sense='==', rhs=1)

    return qp


