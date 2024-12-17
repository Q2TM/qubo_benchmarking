import numpy as np
import sys
import time
import threading
import _thread as thread

from itertools import permutations
from amplify import VariableGenerator, ConstraintList, one_hot, Poly
from qiskit_optimization import QuadraticProgram


def create_amplify_qubo_model(matrix: list[list[int]], weight: int = 1) -> dict:
    size = len(matrix)
    
    gen = VariableGenerator()
    q = gen.array(type="Binary", shape=(size, size), name="q")
    obj = Poly()
    for i in range(size):
        for j in range(size):
            if i != j:
                for p in range(size):
                    obj += q[i, p] * q[j, (p+1) % size] * matrix[j][i]

    c = ConstraintList()
    for i in range(size):
        c += one_hot(q[i])
        c += one_hot(q[:, i].sum())

    c *= weight
    model = obj + c
    return {"model": model, "obj": obj, "c": c, "q": q}


def create_qiskit_qubo_model(matrix: np.ndarray) -> QuadraticProgram:
    """
    Create a Quadratic Program formulation for the Traveling Salesman Problem (TSP).

    The function takes a matrix of pairwise city distances and encodes the TSP
    problem as a Quadratic Program (QP). The QP formulation aims to minimize
    the total distance traveled by visiting each city exactly once and returning
    to the starting city.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D square matrix where `matrix[i, j]` represents the distance between
        city `i` and city `j`. The matrix must have shape `(n, n)`, where `n` is
        the number of cities.

    Returns
    -------
    qp : QuadraticProgram
        A Quadratic Program representing the TSP problem. The objective is to minimize
        the total distance of the tour, and the constraints ensure that each city is
        visited exactly once and left exactly once.
    """
    qp = QuadraticProgram()

    size = len(matrix)

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
                    quadratic_matrix[(f'x_{i}_{p}', f'x_{j}_{(p+1) % size}')] = matrix[j][j]

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

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt

def bruteforce(matrix: list[list[int]], timeout: float=-1) -> tuple:
    """
    Brute force solution for the Traveling Salesman Problem (TSP).

    This function evaluates all possible routes between cities and selects the one
    with the shortest total distance using a brute force method. The brute force method
    checks every possible permutation of city visits.

    Parameters
    ----------
    matrix : list[list[int]]
        A 2D matrix where `weights[j][i]` represents the distance between city `i`
        and city `j`. The matrix should have shape `(n, n)`.

    Returns
    -------
    min_path : tuple
        The path (as a tuple of city indices) that results in the shortest tour.
    min_cost : float
        The total distance of the optimal tour.

    Examples
    --------
    >>> weights = [[0, 10, 15, 20],
    ...            [10, 0, 35, 25],
    ...            [15, 35, 0, 30],
    ...            [20, 25, 30, 0]])
    >>> min_path, min_cost = brute_force_tsp(weights)
    >>> print(min_path, min_cost)
    (0, 1, 3, 2) 80

    Notes
    -----
    This method may be computationally expensive for large numbers of cities
    due to the factorial growth of possible routes.
    """
    if timeout > 0:
        timer = threading.Timer(timeout, quit_function, args=["bruteforce"])
        timer.start()
    
    try:
        start_time = time.time()              
        size = len(matrix)

        min_cost = np.inf
        min_path = None

        for path in permutations(range(size)):
            cost = 0
            for i in range(size):
                cost += matrix[path[(i+1) % size]][path[i]]

            if cost < min_cost:
                min_cost = cost
                min_path = path
       
    finally:
        if timeout > 0:
            timer.cancel()
        return min_cost, min_path, time.time() - start_time


def reorder(x: list, size: int):
    """
    Reorder the binary list to match the order of the cities.

    Given a binary list representing city visits in the TSP problem, this function 
    reorders the list to match the order in which cities are visited based on a 
    specific format.

    Parameters
    ----------
    x : list
        A binary list of length `size * size` representing the possible visitations
        of cities. A value of `1` at index `i` indicates that the city `(i % size)` 
        is visited at step `(i // size)`.
    size : int
        The number of cities involved in the TSP problem.

    Returns
    -------
    y : list
        A reordered array of city indices, indicating the order in which the cities 
        are visited.

    Examples
    --------
    >>> x = [0, 1, 0, 1, 0, 1, 1, 0, 0]
    >>> size = 3
    >>> reorder(x, size)
    [1, 2, 0]
    """
    y = np.zeros(size, dtype=int)
    for i, v in enumerate(x):
        if v == 1:
            y[int(i) % size] = int(i // size)
    return y.tolist()