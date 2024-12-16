import sys
import os

from Utils.graph import draw_graph


def calculate_cost(
    assign: list[int] | tuple[int, ...],
    distance_matrix: list[list[int]],
    interaction_matrix: list[list[int]],
):
    """Calculate cost when assign facility i on site assign[i]

    Args:
        assign (List[int]): assign facility i on site assign[i]
        distance_matrix (List[List[int]]): Adjacency matrix of distance
        interaction_matrix (List[List[int]]): Adjacency matrix of interaction

    Returns:
        int: Total cost
    """
    size = len(assign)

    total_cost = 0
    for i in range(size):
        for j in range(i + 1, size):
            total_cost += interaction_matrix[i][j] * \
                distance_matrix[assign[i]][assign[j]]

    return total_cost


def assign_facilities(
    assign: list[int] | tuple[int, ...],
    location_name: list[str],
    facility_name: list[str],
    distance_matrix: list[list[int]],
    interaction_matrix: list[list[int]],
):
    size = len(assign)

    mapped_distance_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            mapped_distance_matrix[i][j] = distance_matrix[assign[i]][assign[j]]

    cost_matrix = [[mapped_distance_matrix[i][j] * interaction_matrix[i][j]
                    for j in range(size)] for i in range(size)]

    node_labels = []
    for i in range(size):
        print(f"{facility_name[i]} is assigned to {location_name[assign[i]]}")
        node_labels.append(
            f"{facility_name[i]}\nOn {
                location_name[assign[i]]}")

    total_cost = calculate_cost(assign, distance_matrix, interaction_matrix)

    print(f"Total Cost: {total_cost}")

    draw_graph(
        f"Assignment Result (Total Cost: {total_cost})", node_labels, [
            mapped_distance_matrix, interaction_matrix, cost_matrix], ["Distance", "Interaction", "Cost"])
