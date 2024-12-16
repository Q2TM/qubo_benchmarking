# Quadratic Assignment Problem

dealing with assigning a set of facilities to a set of locations, considering the pairwise distances and flows between them.

**goal**: Minimizing the total cost or distance, taking into account both the distances and the flow.

There are various types of algorithms for different problem structures, such as:

1. Precise algorithms
2. Approximation algorithms
3. Metaheuristics like genetic algorithms and simulated annealing
4. Specialized algorithms

**Example**
Given four facilities (F1, F2, F3, F4) and four Locations (L1, L2, L3, L4). We have a cost matrix that represents the pairwise distances or costs between facilities. Additionally, we have a flow matrix that represents the interaction or flow between locations. Find the assignment that minimizes the total cost based on the interactions between facilities and locations. Each facility must be assigned to exactly one location, and each location can only accommodate one facility.

https://www.geeksforgeeks.org/quadratic-assignment-problem-qap/

## Files

- `old` - Folder of outdated files
  - `old/problem.ipynb` - @Qwenty228's attempt to solve problem with Qiskit
  - `old/solvers.ipynb` - Previous version of current `report.ipynb` written by @leomotors
- `report.ipynb` and `report.pdf` - @leomotors solving problem with Fixstars, Gurobi and D-Wave

- `benchmark.py` - Script to run benchmark (Result used by `report.ipynb`)
- `compare.py` - Function for running three solvers and output Dataclasses of Result
- `graph.py` - Utility functions for graph calculaton
- `model.py` - Utility functions for model creation

- `result_v3.json` - Latest benchmark result (See `report.pdf`)

## Details

See `report.pdf`

## Running Benchmark

Use this command from root directory

```
poetry run python3 QAP/benchmark.py > QAP/tmp_stdout.txt
```

Otherwise Gurobi License Information and Error Log will flush on your screen and you cannot track the progress.

Note: TQDM output as stderr
