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

1. `problem.ipynb` - @Qwenty228 solving problem with Qiskit
2. `solvers.ipynb` - @leomotors solving problem with Fixstars, Gurobi and D-Wave

## Details

See `report.pdf`
