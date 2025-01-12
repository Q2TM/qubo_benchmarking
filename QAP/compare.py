import time

from dataclasses import dataclass
from amplify import solve
import numpy as np

from QAP.model import create_qap_qp_model
from QAP.graph import calculate_cost

from Utils.graph import create_qap_input
from Utils.solvers import CreateFixstarsClient, CreateGurobiClient, CreateDWaveClient


@dataclass
class SolverResult:
    name: str
    objective: float | None
    execution_time: float | None
    total_time: float | None


@dataclass
class SolverError:
    name: str
    error: Exception | None


@dataclass
class CompareResult:
    """Result of QAP Run"""

    # Key 1
    nodes: int
    max_edge_weight: int
    avg_edge_weight: float

    distance_matrix: list[list[int]]
    interaction_matrix: list[list[int]]

    # Key 2
    qp_weight: int

    time_model_formulation: float

    solvers_results: list[SolverResult]


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


clientG10s = CreateGurobiClient(timeout_sec=10)
clientG100s = CreateGurobiClient(timeout_sec=100)
clientFixstars1s = CreateFixstarsClient(timeout=1000)
clientFixstars10s = CreateFixstarsClient(timeout=10000)
clientDWave41 = CreateDWaveClient("Advantage_system4.1")
clientDWave64 = CreateDWaveClient("Advantage_system6.4")
clientDWaveV2 = CreateDWaveClient("Advantage2_prototype2.6")

solvers = [
    ("Gurobi 10s", clientG10s),
    ("Gurobi 100s", clientG100s),
    ("Fixstars 1s", clientFixstars1s),
    ("Fixstars 10s", clientFixstars10s),
    ("D-Wave AS4.1", clientDWave41),
    ("D-Wave AS6.4", clientDWave64),
    ("D-Wave V2p2.6", clientDWaveV2),
]


def run_compare_solvers(
        nodes: int,
        max_edge_weight: int,
        constraint_weight: int,
        extra_seed: str = "none"
):
    create_model_start = time.time()

    distance_matrix, interaction_matrix = create_qap_input(
        nodes, 1, max_edge_weight, 1, extra_seed=extra_seed)

    avg_edge_weight = float(
        (np.average(distance_matrix) + np.average(interaction_matrix)) / 2)

    qp_weight = constraint_weight

    qp_model = create_qap_qp_model(
        distance_matrix, interaction_matrix, qp_weight)
    create_model_end = time.time()

    time_model_formulation = create_model_end - create_model_start

    global solvers

    # Brute Force
    best_cost_bf, _, bf_time = solve_bf(
        interaction_matrix, distance_matrix)

    results = []
    errors = []

    for solver in solvers:
        solver_objective = None
        solver_execution_time = None
        solver_error = None
        solver_total_time = None

        solver_total_time_start = time.time()

        try:
            result = solve(qp_model["model"], solver[1])
            solver_objective = result.best.objective
            solver_execution_time = result.execution_time.total_seconds()
            solver_total_time = time.time() - solver_total_time_start
        except Exception as err:
            solver_error = err

        results.append(SolverResult(
            name=solver[0],
            objective=solver_objective,
            execution_time=solver_execution_time,
            total_time=solver_total_time,
        ))

        errors.append(SolverError(
            name=solver[0],
            error=solver_error
        ))

    return CompareResult(
        nodes=nodes,
        max_edge_weight=max_edge_weight,
        avg_edge_weight=avg_edge_weight,
        distance_matrix=distance_matrix,
        interaction_matrix=interaction_matrix,
        qp_weight=qp_weight,
        time_model_formulation=time_model_formulation,
        solvers_results=results,
    ), errors
