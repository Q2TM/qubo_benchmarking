import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

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
        
    def barplot(self):      
        # Convert data to a pandas DataFrame
        if not self.results:
            self.loaddata()
            if not self.results:
                print("No data to plot")
                return

        df = pd.DataFrame(self.results)
        # Check for non-null objectives and execution times
        success_conditions = (
            (df["gurobi_objective"].notnull()) & (df["gurobi_execution_time"].notnull()),
            (df["fixstars_objective"].notnull()) & (df["fixstars_execution_time"].notnull()),
            (df["dwave_objective"].notnull()) & (df["dwave_execution_time"].notnull())
        )

        # Count successes for each solver by nodes
        success_counts = {
            "nodes": df["nodes"].unique(),
            "gurobi_success": [((df["nodes"] == node) & success_conditions[0]).sum() for node in df["nodes"].unique()],
            "fixstars_success": [((df["nodes"] == node) & success_conditions[1]).sum() for node in df["nodes"].unique()],
            "dwave_success": [((df["nodes"] == node) & success_conditions[2]).sum() for node in df["nodes"].unique()],
        }

        # Convert success_counts to DataFrame for plotting
        success_df = pd.DataFrame(success_counts)

        # Plotting
        bar_width = 0.25
        index = np.arange(len(success_df["nodes"]))

        plt.figure(figsize=(10, 6))
        plt.bar(index, success_df["gurobi_success"], bar_width, label="Gurobi Success", color="skyblue")
        plt.bar(index + bar_width, success_df["fixstars_success"], bar_width, label="Fixstars Success", color="salmon")
        plt.bar(index + 2 * bar_width, success_df["dwave_success"], bar_width, label="D-Wave Success", color="lightgreen")

        # Adding labels and title
        plt.xlabel("Nodes")
        plt.ylabel("Number of Successful Runs")
        plt.title("Count of Successful Runs by Solver and Node Count")
        plt.xticks(index + bar_width, success_df["nodes"])
        plt.legend(title="Solver")

        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.show()

