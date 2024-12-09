import json


class RunData:
    """
    "nodes": 4,
    "max_edge_weight": 9,
    "avg_edge_weight": 4.5,
    "distance_matrix": [
      [0, 4, 9, 7],
      [4, 0, 3, 9],
      [9, 3, 0, 4],
      [7, 9, 4, 0]
    ],
    "qp_weight": 1000000,
    "time_model_formulation": 0.0005192756652832031,
    "gurobi_objective": 364.0,
    "gurobi_execution_time": 0.007966,
    "fixstars_objective": 364.0,
    "fixstars_execution_time": 0.985476,
    "dwave_objective": 364.0,
    "dwave_execution_time": 0.190519"""

    def __init__(self, filename="result.json"):
        self.filename = filename
        self.results = []
        
    def add(self, node: int, distance_matrix: list[list[int]], qp_weight: int, model_formulation: float, gurobi_objective: float, gurobi_execution_time: float, fixstars_objective: float, fixstars_execution_time: float, dwave_objective: float, dwave_execution_time: float):
        data = {}
        data["nodes"] = node
        data["max_edge_weight"] = max([max(x) for x in distance_matrix])
        data["avg_edge_weight"] = sum([sum(x) for x in distance_matrix]) / (node * (node - 1)) # diag is 0
        data["distance_matrix"] = distance_matrix
        data["qp_weight"] = qp_weight
        data["time_model_formulation"] = model_formulation
        data["gurobi_objective"] = gurobi_objective
        data["gurobi_execution_time"] = gurobi_execution_time
        data["fixstars_objective"] = fixstars_objective
        data["fixstars_execution_time"] = fixstars_execution_time
        data["dwave_objective"] = dwave_objective
        data["dwave_execution_time"] = dwave_execution_time
        self.results.append(data)
      
    def save(self):
        with open(self.filename, mode="w") as f:
            json.dump(self.results, f)

    def loaddata(self):
        with open(self.filename) as f:
            data = json.load(f)
        self.results = data

