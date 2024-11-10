import json
from compare import run_compare_solvers
from dataclasses import asdict
from tqdm import tqdm

nodes = [4, 5, 6, 7, 8, 10, 12, 15, 20, 69]
weights = [9, 99, 99999]
repeat = 3

result = []

n = len(nodes) * len(weights) * repeat

for i in tqdm(nodes, desc="Nodes"):
    for j in tqdm(weights, desc="Weights"):
        for it in tqdm(range(repeat), desc="Repeat"):
            result.append(run_compare_solvers(i, j))

# Convert dataclass to dict
serializable = list(map(lambda x: asdict(x), result))

# Convert result into json
with open("result.json", "w") as f:
    json.dump(serializable, f)
