import json
from dataclasses import asdict
from tqdm import tqdm

from Utils.measure import run_compare_solvers
from Utils.solvers import CreateFixstarsClient, CreateGurobiClient, CreateDWaveClient

from QAP.brute_force import solve_bf
from QAP.model import create_model_from_seed

solvers = [
    ("Gurobi 10s", CreateGurobiClient(timeout_sec=10)),
    # ("Gurobi 100s", CreateGurobiClient(timeout_sec=100)),
    # ("Fixstars 1s", CreateFixstarsClient(timeout=1000)),
    ("Fixstars 10s", CreateFixstarsClient(timeout=10000)),
    ("D-Wave AS4.1", CreateDWaveClient("Advantage_system4.1")),
    ("D-Wave AS6.4", CreateDWaveClient("Advantage_system6.4")),
    # ("D-Wave V2p2.6", CreateDWaveClient("Advantage2_prototype2.6")),
]

# solvers = [
#     ("Gurobi 10s", CreateGurobiClient(timeout_sec=10)),
#     ("Gurobi 100s", CreateGurobiClient(timeout_sec=100)),
#     ("Gurobi 1000s", CreateGurobiClient(timeout_sec=1000)),
#     ("Gurobi 10000s", CreateGurobiClient(timeout_sec=10000)),
# ]


nodes_set = [4, 5, 6, 7, 8, 9, 10, 11, 12]
repeat = 10

results = []

n = len(nodes_set) * repeat

for nodes in tqdm(nodes_set, desc="Nodes"):
    for it in tqdm(range(repeat), desc="Repeat"):
        qp_model, time_model_formulation, distance_matrix, interaction_matrix = create_model_from_seed(
            nodes=nodes, max_edge_weight=9, constraint_weight=1_000_000)

        bf_cost, bf_perm, bf_time = solve_bf(
            interaction_matrix, distance_matrix, timeout_sec=100)

        result, errors = run_compare_solvers(
            nodes=nodes,
            qp_model=qp_model,
            time_model_formulation=time_model_formulation,
            bruteforce=(bf_cost, bf_time),
            amplifys=solvers)
        results.append(result)

        for err in errors:
            if err.error is not None:
                print(f"Error at nodes={nodes} it={it}")
                print(err)

# Convert dataclass to dict
serializable = list(map(lambda x: asdict(x), results))

# Convert result into json
with open("QAP/result_v6.json", "w") as f:
    json.dump(serializable, f)
