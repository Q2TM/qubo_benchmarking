import json
from dataclasses import asdict
from tqdm import tqdm

from QAP.compare import run_compare_solvers

# TODO Matrix for V4
nodes = [69]
repeat = 1

results = []

n = len(nodes) * repeat

for node in tqdm(nodes, desc="Nodes"):
    for it in tqdm(range(repeat), desc="Repeat"):
        result, errors = run_compare_solvers(
            node, 9, 1_000_000, extra_seed=f"{it}")
        results.append(result)

        for err in errors:
            if err.error is not None:
                print(f"Error at node={node} it={it}")
                print(err)

# Convert dataclass to dict
serializable = list(map(lambda x: asdict(x), results))

# Convert result into json
with open("QAP/result_v4.json", "w") as f:
    json.dump(serializable, f)
