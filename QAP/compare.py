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
clientFixstars = CreateFixstarsClient(timeout=10000)
clientDWave41 = CreateDWaveClient("Advantage_system4.1")
clientDWave64 = CreateDWaveClient("Advantage_system6.4")
clientDWaveV2 = CreateDWaveClient("Advantage2_prototype2.6")


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

    global clientG10s, clientG1000s, clientFixstars, clientDWave41, clientDWave64, clientDWaveV2

    # Brute Force
    best_cost_bf, _, bf_time = solve_bf(
        interaction_matrix, distance_matrix)

    # Gurobi 10s timeout
    gurobi10s_objective = None
    gurobi10s_execution_time = None
    gurobi10s_error = None
    real_gurobi10s_start = time.time()
    real_gurobi10s_time = None
    try:
        resultG = solve(qp_model["model"], clientG10s)
        gurobi10s_objective = resultG.best.objective
        gurobi10s_execution_time = resultG.execution_time.total_seconds()
        real_gurobi10s_time = time.time() - real_gurobi10s_start
    except Exception as err:
        gurobi10s_error = err

    # Gurobi 100s timeout
    gurobi100s_objective = None
    gurobi100s_execution_time = None
    gurobi100s_error = None
    real_gurobi100s_start = time.time()
    real_gurobi100s_time = None
    try:
        resultG = solve(qp_model["model"], clientG100s)
        gurobi100s_objective = resultG.best.objective
        gurobi100s_execution_time = resultG.execution_time.total_seconds()
        real_gurobi100s_time = time.time() - real_gurobi100s_start
    except Exception as err:
        gurobi100s_error = err

    # Fixstars
    fixstars_objective = None
    fixstars_execution_time = None
    fixstars_error = None
    real_fixstars_start = time.time()
    real_fixstars_time = None
    try:
        resultFS = solve(qp_model["model"], clientFixstars)
        fixstars_objective = resultFS.best.objective
        fixstars_execution_time = resultFS.solutions[0].time.total_seconds()
        real_fixstars_time = time.time() - real_fixstars_start
    except Exception as err:
        fixstars_error = err

    # D-Wave Advantage_system4.1
    dwave41_objective = None
    dwave41_execution_time = None
    dwave41_error = None
    real_dwave41_start = time.time()
    real_dwave41_time = None
    try:
        resultDWave = solve(qp_model["model"], clientDWave41)
        dwave41_objective = resultDWave.best.objective
        dwave41_execution_time = resultDWave.execution_time.total_seconds()
        real_dwave41_time = time.time() - real_dwave41_start
    except Exception as err:
        dwave41_error = err

    # D-Wave Advantage_system6.4
    dwave64_objective = None
    dwave64_execution_time = None
    dwave64_error = None
    real_dwave64_start = time.time()
    real_dwave64_time = None
    try:
        resultDWave = solve(qp_model["model"], clientDWave64)
        dwave64_objective = resultDWave.best.objective
        dwave64_execution_time = resultDWave.execution_time.total_seconds()
        real_dwave64_time = time.time() - real_dwave64_start
    except Exception as err:
        dwave64_error = err

    # D-Wave Advantage2_prototype2.6
    dwaveV2_objective = None
    dwaveV2_execution_time = None
    dwaveV2_error = None
    real_dwaveV2_start = time.time()
    real_dwaveV2_time = None
    try:
        resultDWave = solve(qp_model["model"], clientDWaveV2)
        dwaveV2_objective = resultDWave.best.objective
        dwaveV2_execution_time = resultDWave.execution_time.total_seconds()
        real_dwaveV2_time = time.time() - real_dwaveV2_start
    except Exception as err:
        dwaveV2_error = err

    return CompareResult(
        nodes=nodes,
        max_edge_weight=max_edge_weight,
        avg_edge_weight=avg_edge_weight,
        distance_matrix=distance_matrix,
        interaction_matrix=interaction_matrix,
        qp_weight=qp_weight,
        time_model_formulation=time_model_formulation,
        solvers_results=[
            SolverResult(
                name="Brute Force", objective=best_cost_bf,
                execution_time=bf_time, total_time=bf_time),
            SolverResult(
                name="Gurobi 10s", objective=gurobi10s_objective,
                execution_time=gurobi10s_execution_time, total_time=real_gurobi10s_time),
            SolverResult(
                name="Gurobi 100s", objective=gurobi100s_objective,
                execution_time=gurobi100s_execution_time, total_time=real_gurobi100s_time),
            SolverResult(
                name="Fixstars", objective=fixstars_objective,
                execution_time=fixstars_execution_time, total_time=real_fixstars_time),
            SolverResult(
                name="D-Wave AS4.1", objective=dwave41_objective,
                execution_time=dwave41_execution_time, total_time=real_dwave41_time),
            SolverResult(
                name="D-Wave AS6.4", objective=dwave64_objective,
                execution_time=dwave64_execution_time, total_time=real_dwave64_time),
            SolverResult(
                name="D-Wave V2p2.6", objective=dwave64_objective,
                execution_time=dwaveV2_execution_time, total_time=real_dwaveV2_time),
        ],
    ), [
        SolverError(name="Gurobi 10s", error=gurobi10s_error),
        SolverError(name="Gurobi 100s", error=gurobi100s_error),
        SolverError(name="Fixstars", error=fixstars_error),
        SolverError(name="D-Wave AS4.1", error=dwave41_error),
        SolverError(name="D-Wave AS6.4", error=dwave64_error),
        SolverError(name="D-Wave V2p2.6", error=dwaveV2_error),
    ]
