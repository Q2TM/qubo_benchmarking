import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath('..'))

# Prevent autopep8
if True:
    from Utils.graph import draw_graph


def calculate_cost(
    assign: list[int],
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
        for j in range(size):
            total_cost += distance_matrix[i][j] * \
                interaction_matrix[assign[i]][assign[j]]

    return total_cost


def assign_facilities(
    assign: list[int],
    location_name: list[str],
    facility_name: list[str],
    distance_matrix: list[list[int]],
    interaction_matrix: list[list[int]],
):
    size = len(assign)

    # Create Three empty matrix with size of size*size
    # distance_matrix = [[0 for _ in range(size)] for _ in range(size)]
    mapped_interaction_matrix = [[0 for _ in range(size)] for _ in range(size)]
    cost_matrix = [[0 for _ in range(size)] for _ in range(size)]

    node_labels = []
    for i in range(size):
        print(f"{location_name[i]} is assigned to {facility_name[assign[i]]}")
        node_labels.append(
            f"{facility_name[assign[i]]}\nOn {
                location_name[i]}")

    total_cost = calculate_cost(assign, distance_matrix, interaction_matrix)

    print(f"Total Cost: {total_cost}")

    draw_graph(
        f"Assignment Result (Total Cost: {total_cost})", node_labels, [
            distance_matrix, mapped_interaction_matrix, cost_matrix], ["Distance", "Interaction", "Cost"])
