import time

from amplify import solve
from typing import Any
from dataclasses import dataclass, asdict, make_dataclass


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


def compare_result(nodes: int, time_model_formulation: float, solvers_results: list[SolverResult], **extras) -> Any:
    """
    Dynamically creates a dataclass and returns an instance with the provided arguments.

    This function avoids the complexity of nested dataclasses by flattening all the attributes 
    into a single dynamically created dataclass. It is particularly useful when the input may 
    include nested structures or additional attributes that are not known beforehand.

    Arguments are in order of CompareResult dataclass.

    Parameters
    ----------
    nodes : int
        The number of nodes.
    time_model_formulation : float
        Time taken to formulate the QP model.
    solvers_results : list
        A list of results from different solvers.
    **extras : dict
        Additional parameters to include as fields in the resulting dataclass.

    Returns
    -------
    CompareResult : dataclass instance
        A dynamically created dataclass instance with all provided attributes as fields.

    Notes
    -----
    - This function avoids nested dataclasses for simplicity and flattens the structure into a single-level dataclass.
    - `make_dataclass` from `dataclasses` is used to dynamically define the `CompareResult` class.
    - The type of each attribute is inferred dynamically using `type(value)`.
    - This approach ensures flexibility while keeping the implementation manageable.

    Example
    -------
    >>> from dataclasses import asdict
    >>> solver_result = SolverResult("Qiskit", 100, 0.1, 0.2)
    >>> result = compare_result(
    ...     nodes=10,
    ...     time_model_formulation=0.023,
    ...     solvers_results=[solver_result],
    ...     qp_weight=5,
    ...     distance_matrix=[[0, 1], [1, 0]],
    ...     custom_field="Custom Value"
    ... )
    >>> print(result.nodes)
    10
    >>> print(result.distance_matrix)
    [[0, 1], [1, 0]]
    >>> print(result.custom_field)
    'Custom Value'
    >>> print(asdict(result))
    {'nodes': 10, 'qp_weight': 5, 'distance_matrix': [[0, 1], [1, 0]], 'custom_field': 'Custom Value', 'time_model_formulation': 0.023, 'solvers_results': [{'name': 'Qiskit', 'objective': 100, 'execution_time': 0.1, 'total_time': 0.2}]}
    """

    result_dict = {"nodes": nodes,
                   **extras,
                   "time_model_formulation": time_model_formulation,
                   "solvers_results": solvers_results}
    return make_dataclass("CompareResult", zip(result_dict, map(type, result_dict.values())))(**result_dict)


def run_compare_solvers(
        nodes: int,
        qp_model: dict,
        time_model_formulation: float,
        bruteforce: tuple[int, float] = None,
        amplifys: list[tuple] = [],
        qiskits: list[tuple] = [],
        **kwargs
):
    """
    Benchmarking different solvers solving a Quadratic Program (QP) model.
    
    Args:
        nodes (int): Number of nodes
        qp_model (dict): Quadratic program model. {"model": model, ...}
        time_model_formulation (float): Time taken to formulate the QP model
        bruteforce (tuple[int, float], optional): (bruteforce cost, bruteforce time taken). Defaults to None.
        amplifys (list[tuple], optional): List of Amplify solvers. Defaults to [].
        qiskits (list[tuple], optional): List of Qiskit solvers. Defaults to [].
    """

    solvers_results = []
    errors = []

    if bruteforce:
        # Brute Force
        solvers_results.append(
            SolverResult(
                name="Brute Force", objective=bruteforce[0],
                execution_time=bruteforce[1], total_time=bruteforce[1]))

    for name, client in amplifys:
        objective = None
        execution_time = None
        error = None
        real_start = time.time()
        real_time = None
        try:
            result = solve(qp_model["model"], client)
            objective = result.best.objective
            execution_time = result.solutions[0].time.total_seconds()
            real_time = time.time() - real_start
        except Exception as err:
            error = err

        solvers_results.append(SolverResult(
            name=name, objective=objective,
            execution_time=execution_time, total_time=real_time),)
        errors.append(SolverError(name="Gurobi 10s", error=error))

    return compare_result(
        nodes=nodes,
        time_model_formulation=time_model_formulation,
        solvers_results=solvers_results,
        **kwargs), errors


if __name__ == "__main__":
    print(asdict(compare_result(nodes=5,
                                time_model_formulation=0.5,
                                solvers_results=[SolverResult("Qiskit", 100, 0.1, 0.2),
                                                 SolverResult("Amplify", 200, 0.2, 0.3)],
                                qp_weight=5,
                                add_info={"extra": SolverError(
                                    "Qiskit", None)},
                                max_edge_weight=10,
                                avg_edge_weight=5,
                                distance_matrix=[
                                    [1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                interaction_matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
