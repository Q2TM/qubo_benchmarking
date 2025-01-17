import sys
from datetime import timedelta
import time
import re
import numpy as np
from numpy.typing import NDArray
from amplify import VariableGenerator, solve
from dataclasses import dataclass, asdict, make_dataclass

import Max3SAT
sys.path.append('../') # To use Utils
from Utils.solvers import CreateFixstarsClient, CreateGurobiClient, CreateDWaveClient

class solver:
    def __init__(self, problem: Max3SAT):
        self.problem = problem
        self.model, self.time_model_formulation = self.QUBO_matrix_to_model(self.problem.QUBO_matrix)

    @staticmethod
    def QUBO_matrix_to_model(QUBO_matrix: np.ndarray):
        """
        Converts a QUBO matrix into a specific model format using VariableGenerator.
        """
        create_model_start = time.time()

        size = QUBO_matrix.shape[0] # [n+m]
        
        gen = VariableGenerator()
        model = gen.matrix("Binary", size)
        q = model.quadratic

        for i in range(size):
            for j in range(size):
                q[i][j] = QUBO_matrix[i][j]

        create_model_end = time.time()
        time_model_formulation = create_model_end - create_model_start

        print("Finished model formulation.")
        print("Model formulation time:", time_model_formulation)

        return model, time_model_formulation

    def bruteForce(self) -> tuple[NDArray[np.int_], int]:
        num_x = self.problem.QUBO_matrix.shape[0]
        min_energy = np.inf
        result_x = np.zeros(num_x, dtype=int)

        for i in range(2 ** num_x):
            x = np.array([int(bit) for bit in bin(i)[2:].zfill(num_x)])  # Convert to binary
            energy = x @ self.problem.QUBO_matrix @ x  # Efficient dot product
            if energy < min_energy:
                min_energy = energy
                result_x = x
            if ((min_energy + self.problem.K) * -1 == self.problem.num_clauses):  # Optimal solution found
                break
        
        return result_x, min_energy
    
    def solve_with_client(self, client_type: str) -> tuple[int, int, timedelta, timedelta]:
        """
        Solves the Max3SAT problem using different solvers (FixStar, Gurobi, or D-Wave) 
        based on the provided client type and returns the solution, energy, and the number 
        of satisfied clauses.

        Args:
            client_type (str): The type of solver client ('FixStars{10s / 100s}'', 'Gurobi{10s / 100s}', or 'DWave{4.1 / 6.4 / V2}').

        Returns:
            tuple: A tuple containing:
                - int: The number of clauses satisfied (calculated based on g(x, w) = -(K + z @ F(φ) @z).
                - int: The number of clauses satisfied (verified by the problem.verify() method).
                - timedelta: Execution time of the solver.
                - timedelta: Total time taken to solve the problem.
        """
        
        # Client initialization and solving based on the client type
        try:
            if client_type == "FixStars1s":
                client = CreateFixstarsClient(timeout=1000)
            elif client_type == "FixStars10s":
                client = CreateFixstarsClient(timeout=10000)
            elif client_type == "FixStars100s":
                client = CreateFixstarsClient(timeout=100000)
            elif client_type == "Gurobi10s":
                client = CreateGurobiClient(timeout_sec=10, library_path="D:\\My programs\\Gurobi\\win64\\bin\\gurobi110.dll")
            elif client_type == "Gurobi100s":
                client = CreateGurobiClient(timeout_sec=100, library_path="D:\\My programs\\Gurobi\\win64\\bin\\gurobi110.dll")
            elif client_type == "Gurobi150s":
                client = CreateGurobiClient(timeout_sec=150, library_path="D:\\My programs\\Gurobi\\win64\\bin\\gurobi110.dll")
            elif client_type == "DWave4.1":
                client = CreateDWaveClient("Advantage_system4.1")
            elif client_type == "DWave6.4":
                client = CreateDWaveClient("Advantage_system6.4")
            elif client_type == "DWaveV2":
                client = CreateDWaveClient("Advantage2_prototype2.6")
            else:
                raise ValueError("Unsupported client type.")
            
            real_start = time.time()

            result = solve(self.model, client)

            literal_result = np.array(list(result.best.values.values()))
            best_objective = result.best.objective
            execution_time = result.solutions[0].time.total_seconds()

            total_time = time.time() - real_start
            # Calculate the number of clauses satisfied using the formula:
            # g(x, w) = -(K + z^T F(φ) z)
            max_clauses_satisfied_Obj = (self.problem.K + best_objective) * -1
            max_clauses_satisfied_Verified = self.problem.verify(literal_result)[1] # Number of satisfied clauses

            # print(f"{client_type} x result:", literal_result)
            # print(f"{client_type} min energy:", best_objective)
            # print(f"Max clauses satisfied with {client_type}: ", max_clauses_satisfied)
            # print(f"{client_type} Execution time:", execution_time)
            print(f"{client_type} Successfully solve, Total time:", total_time)

            return max_clauses_satisfied_Obj, max_clauses_satisfied_Verified, execution_time, total_time

        except Exception as e:
            # Handle any unexpected errors gracefully and log them
            print(f"Error during solving with {client_type}: {str(e)}")
            # Return None or appropriate default values if an error occurs
            return str(e), None, None, None