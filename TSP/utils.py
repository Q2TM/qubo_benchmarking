import matplotlib.pyplot as plt
import numpy as np

from qiskit_optimization import QuadraticProgram
from dotenv import load_dotenv
import os

load_dotenv()

QISKIT_TOKEN = os.getenv("QISKIT_TOKEN")

def randomize_cities(num_nodes: int):
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
    return dist


def make_tsp(weights: np.ndarray) -> QuadraticProgram:
    qp = QuadraticProgram()

    size = weights.shape[0]

    cities = {} # dictionary to store the city and order
    for i in range(size):
        for p in range(size):
            cities[f'x_{i}_{p}'] = qp.binary_var(f'x_{i}_{p}')

    # Objective function
    quadratic_matrix = {}
    for i in range(size):
        for j in range(size):
            if i != j:
                for p in range(size):
                    quadratic_matrix[(f'x_{i}_{p}', f'x_{j}_{(p+1)%size}')] = weights[i, j] 

    qp.minimize(quadratic=quadratic_matrix)
                    
    # Constraint 1: each city is visited exactly once
    for i in range(size):
        qp.linear_constraint(linear={f'x_{i}_{p}': 1 for p in range(size)}, sense='==', rhs=1)

    # Constraint 2: each city is left exactly once
    for p in range(size):
        qp.linear_constraint(linear={f'x_{i}_{p}': 1 for i in range(size)}, sense='==', rhs=1)

    return qp