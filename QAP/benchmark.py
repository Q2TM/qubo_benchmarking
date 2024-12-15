import json
from compare import run_compare_solvers
from dataclasses import asdict
from tqdm import tqdm

nodes = [4, 5, 6, 7, 8, 10, 12, 15, 20, 69]
constraint_weights = [1_000, 1_000_000, 1_000_000_000]
repeat = 3

result = []

n = len(nodes) * len(constraint_weights) * repeat

for node in tqdm(nodes, desc="Nodes"):
    for con_weight in tqdm(constraint_weights, desc="Weights"):
        for it in tqdm(range(repeat), desc="Repeat"):
            result.append(run_compare_solvers(
                node, 9, con_weight, extra_seed=f"{it}")[0])

# Convert dataclass to dict
serializable = list(map(lambda x: asdict(x), result))

# Convert result into json
with open("result_v3.json", "w") as f:
    json.dump(serializable, f)
