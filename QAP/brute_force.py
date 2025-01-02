import time
from QAP.graph import calculate_cost


def solve_bf(interaction_matrix, distance_matrix, timeout_sec=100):
    from itertools import permutations

    n = len(interaction_matrix)
    m = len(distance_matrix)
    assert n == m

    best_cost = float('inf')
    best_permutation = ()

    time_start = time.time()

    for perm in permutations(range(n)):
        cost = calculate_cost(perm, interaction_matrix, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_permutation = perm

        if time.time() - time_start > timeout_sec:
            break

    return best_cost, best_permutation, time.time() - time_start
