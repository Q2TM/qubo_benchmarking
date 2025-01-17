import json
import time
from dataclasses import asdict
from typing import Any, Literal
from tqdm import tqdm

from TSP.model import Tsp
from TSP.core import bruteforce
from Utils.solvers import CreateGurobiClient, CreateFixstarsClient, CreateDWaveClient
from Utils.measure import run_compare_solvers


# Set up clients
clientG10s = CreateGurobiClient(
    library_path=r"D:\miniconda3\envs\cwq\gurobi110.dll", timeout_sec=10)
clientG100s = CreateGurobiClient(
    library_path=r"D:\miniconda3\envs\cwq\gurobi110.dll", timeout_sec=100)
clientFixstars = CreateFixstarsClient(timeout=1000)
clientFixstars10s = CreateFixstarsClient(timeout=10_000)
clientDWave41 = CreateDWaveClient("Advantage_system4.1")
clientDWave64 = CreateDWaveClient("Advantage_system6.4")
clientDWaveV2 = CreateDWaveClient("Advantage2_prototype2.6")

solvers = [ ("Fixstars 1s", clientFixstars),
           ("Fixstars 10s", clientFixstars10s),
    ("DWave 4.1", clientDWave41),
    ("DWave 6.4", clientDWave64),
   ]
# solvers = []
# Set up problems
nodes = [11, 12]
repeat = 10


def run(nodes: int, solver: Literal['Qiskit', 'Amplify'], penalty: int = 1_000_000) -> tuple[Any, list]:
    """
    Run TSP benchmark for a given node and solver

    Args:
        nodes (int): Number of nodes
        solver (str): Solver name
        penalty (int, optional): Penalty. Defaults to 1_000_000.

    Returns:
        tuple[CompareResult, list]: Result and errors
    """
    # Create TSP model
    tsp = Tsp(nodes, solver=solver, seed=123, draw=False)

    # Problem formulation
    create_model_start = time.time()
    qubo = tsp.qubo(penalty=penalty)
    time_model_formulation = time.time() - create_model_start

    # bruteforce
    cost, _, time_taken = bruteforce(tsp.matrix, timeout=100)
    bruteforce_result = (cost, time_taken)

    # Run with different solvers
    return run_compare_solvers(nodes, qubo, time_model_formulation, bruteforce=bruteforce_result, amplifys=solvers,
                              max_edge_weight=tsp.max_edge_weight,
                               avg_edge_weight=tsp.avg_edge_weight,
                                distance_matrix=tsp.matrix,
                                constraint_weight=penalty,
                                )

# Run benchmark
results = []
for node in tqdm(nodes, desc="Nodes"):
    for it in tqdm(range(repeat), desc="Repeat"):
        result, errors = run(node, "Amplify", 1_000_000)
        results.append(result)

for err in errors:
    if err.error is not None:
        print(f"Error at node={node} it={it}")
        print(err)

# Convert dataclass to dict
serializable= list(map(lambda x: asdict(x), results))

# Convert result into json
with open("TSP/result.json", "w") as f:
    json.dump(serializable, f)
