import numpy as np
import random


def dense_distance_matrix(n, max_distance=100):
    """Create a dense distance matrix where all nodes are connected."""
    matrix = np.random.randint(1, max_distance, size=(n, n))
    np.fill_diagonal(matrix, 0)  # Set diagonal to zero, as distance to itself is zero
    return matrix

def sparse_distance_matrix(n, max_distance=100, min_edges=3):
    """Create a sparse distance matrix with at least one possible tour."""
    matrix = np.zeros((n, n))  # Initialize with "inf" (no direct path)
    
    # Create a minimal spanning tree for connectivity
    for i in range(1, n):
        distance = random.randint(1, max_distance)
        matrix[i-1, i] = distance
        matrix[i, i-1] = distance
    
    # Add a few more random edges to ensure a feasible TSP path
    while np.isinf(matrix).sum() > n * (n - 1) - min_edges:
        i, j = random.sample(range(n), 2)
        if np.isinf(matrix[i, j]):
            distance = random.randint(1, max_distance)
            matrix[i, j] = distance
            matrix[j, i] = distance
    
    np.fill_diagonal(matrix, 0)
    return matrix

# # Generate matrices for sizes 4 to 10
# for n in range(4, 11):
#     print(f"n = {n}")
#     print("Dense Distance Matrix:")
#     print(dense_distance_matrix(n))
#     print("\nSparse Distance Matrix:")
#     print(sparse_distance_matrix(n))
#     print("\n" + "-"*50 + "\n")


