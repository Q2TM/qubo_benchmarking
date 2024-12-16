from dataclasses import dataclass
from amplify import solve
from model import create_qap_qp_model
from graph import calculate_cost
import numpy as np

import time
import sys
import os

# Add the project root directory to sys.path
if True:
    sys.path.append(os.path.abspath('..'))

# Prevent autopep8
if True:
    from Utils.graph import create_qap_input
    from Utils.solvers import CreateFixstarsClient, CreateGurobiClient, CreateDWaveClient


@dataclass
class SolverResult:
    name: str
    objective: float | None
    execution_time: float | None


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
clientG1000s = CreateGurobiClient(timeout_sec=1000)
clientFixstars = CreateFixstarsClient()
clientDWave = CreateDWaveClient()


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

    global clientG10s, clientG1000s, clientFixstars, clientDWave

    # Brute Force
    best_cost_bf, _, bf_time = solve_bf(
        interaction_matrix, distance_matrix)

    # Gurobi 10s timeout
    gurobi10s_objective = None
    gurobi10s_execution_time = None
    gurobi10s_error = None
    try:
        resultG = solve(qp_model["model"], clientG10s)
        gurobi10s_objective = resultG.best.objective
        gurobi10s_execution_time = resultG.execution_time.total_seconds()
    except Exception as err:
        gurobi10s_error = err

    # Gurobi 1000s timeout
    gurobi1000s_objective = None
    gurobi1000s_execution_time = None
    gurobi1000s_error = None
    try:
        resultG = solve(qp_model["model"], clientG10s)
        gurobi1000s_objective = resultG.best.objective
        gurobi1000s_execution_time = resultG.execution_time.total_seconds()
    except Exception as err:
        gurobi1000s_error = err

    # Fixstars
    fixstars_objective = None
    fixstars_execution_time = None
    fixstars_error = None
    try:
        resultFS = solve(qp_model["model"], clientFixstars)
        fixstars_objective = resultFS.best.objective
        fixstars_execution_time = resultFS.solutions[0].time.total_seconds()
    except Exception as err:
        fixstars_error = err

    # DWave
    dwave_objective = None
    dwave_execution_time = None
    dwave_error = None
    try:
        resultDWave = solve(qp_model["model"], clientDWave)
        dwave_objective = resultDWave.best.objective
        dwave_execution_time = resultDWave.execution_time.total_seconds()
    except Exception as err:
        dwave_error = err

    return CompareResult(
        nodes=nodes,
        max_edge_weight=max_edge_weight,
        avg_edge_weight=avg_edge_weight,
        distance_matrix=distance_matrix,
        interaction_matrix=interaction_matrix,
        qp_weight=qp_weight,
        time_model_formulation=time_model_formulation,
        solvers_results=[
            SolverResult(name="Brute Force", objective=best_cost_bf,
                         execution_time=bf_time),
            SolverResult(name="Gurobi 10s", objective=gurobi10s_objective,
                         execution_time=gurobi10s_execution_time),
            SolverResult(name="Gurobi 1000s", objective=gurobi1000s_objective,
                         execution_time=gurobi1000s_execution_time),
            SolverResult(name="Fixstars", objective=fixstars_objective,
                         execution_time=fixstars_execution_time),
            SolverResult(name="D-Wave", objective=dwave_objective,
                         execution_time=dwave_execution_time)
        ],
    ), [
        SolverError(name="Gurobi 10s", error=gurobi10s_error),
        SolverError(name="Gurobi 1000s", error=gurobi1000s_error),
        SolverError(name="Fixstars", error=fixstars_error),
        SolverError(name="D-Wave", error=dwave_error)
    ]
