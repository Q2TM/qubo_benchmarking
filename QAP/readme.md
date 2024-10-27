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

## Problem

Note: We are assigning 4 facilities on 4 locations, in total there will be
16 decision variables.

I found that any Annealing solvers : Qiskit (Local Simulation), Fixstars (Cloud GPU),
D-Wave (QPU) cannot solve this problem at 4 facilities with big numbers.

## Complexity

Where $N$ = Number of Facilities = Number of Locations

- Variables = $N^2$ = $\mathrm{O}(N^2)$
- Objective Terms = $\frac{N^4 - 2N^3 + n^2}{2}$ = $\mathrm{O}(N^4)$
- Constraints = $2N$ = $\mathrm{O}(N)$

## Update 16 Oct 2024

The problem likely occured by small penalty weight (λ), this can be modified by `c *= weight` when finalizing constraints

The effects on `weight`

Note: Gurobi is used as control for best solution

|                                                                             | Fixstars            | DWave               |
| --------------------------------------------------------------------------- | ------------------- | ------------------- |
| Too small (Or default = 1)                                                  | Result not feasible | Result not feasible |
| Little too small <td colspan="2">Either will give not optimal solution</td> |
| Good Value                                                                  | Best Solution       | Best Solution       |
| Too large                                                                   | Best Solution       | Not best Solution   |
