from typing import List
from amplify import VariableGenerator, equal_to, one_hot, Poly, ConstraintList


def create_qap_qp_model(distance_matrix: List[List[int]], interaction_matrix: List[List[int]], weight=1):
    assert (len(distance_matrix) == len(interaction_matrix))
    size = len(distance_matrix)

    gen = VariableGenerator()
    # Facility i on Site j
    q = gen.array(type="Binary", shape=(size, size), name="q")

    obj = Poly()

    for i1 in range(size):
        for j1 in range(size):
            for i2 in range(size):
                for j2 in range(size):
                    if i1 != i2 and j1 != j2:
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
                # print(f"{location_name[j]} is assigned to {faculty_name[i]}")
                mapped_solution[j] = i

    return mapped_solution
