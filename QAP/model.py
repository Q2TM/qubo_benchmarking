from amplify import VariableGenerator, equal_to, one_hot, Poly, ConstraintList
import time
from Utils.graph import create_qap_input
import numpy as np


def create_qap_qp_model(distance_matrix: list[list[int]], interaction_matrix: list[list[int]], weight=1):
    assert (len(distance_matrix) == len(interaction_matrix))
    size = len(distance_matrix)

    gen = VariableGenerator()
    # Facility i on Site j
    q = gen.array(type="Binary", shape=(size, size), name="q")

    obj = Poly()

    for i1 in range(size):
        for j1 in range(size):
            for i2 in range(i1 + 1, size):
                for j2 in range(size):
                    if j1 != j2:
                        obj += q[i1, j1] * q[i2, j2] * \
                            interaction_matrix[i1][i2] * \
                            distance_matrix[j1][j2]

    c = ConstraintList()

    for i in range(size):
        c += one_hot(q[i])
        c += one_hot(q[:, i].sum())

    c *= weight

    model = obj + c

    return {"model": model, "obj": obj, "c": c, "q": q}


def solution_to_map(result, variables, size=4):
    mapped_solution = [-1] * size

    for i in range(size):
        for j in range(size):
            if result.best.values[variables[i, j]] == 1:
                # print(f"{facility_name[i]} is assigned to {location_name[j]}")
                mapped_solution[i] = j

    return mapped_solution


def create_model_from_seed(
        nodes: int,
        max_edge_weight: int,
        constraint_weight: int,
        extra_seed: str = "none"):

    create_model_start = time.time()

    distance_matrix, interaction_matrix = create_qap_input(
        nodes, 1, max_edge_weight, 1, extra_seed=extra_seed)

    # avg_edge_weight = float(
    #     (np.average(distance_matrix) + np.average(interaction_matrix)) / 2)

    qp_weight = constraint_weight

    qp_model = create_qap_qp_model(
        distance_matrix, interaction_matrix, qp_weight)
    create_model_end = time.time()

    time_model_formulation = create_model_end - create_model_start

    return qp_model, time_model_formulation, distance_matrix, interaction_matrix
