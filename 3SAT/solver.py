import sys
import numpy as np
from numpy.typing import NDArray
from amplify import VariableGenerator, solve

import Max3SAT
sys.path.append('../') # To use Utils
from Utils.solvers import GetGurobiClient, GetFixstarClient, GetDWaveClient

class solver:
    def __init__(self, problem: Max3SAT):
        self.problem = problem
        self.model = self.QUBO_matrix_to_model(self.problem.QUBO_matrix)

    @staticmethod
    def QUBO_matrix_to_model(QUBO_matrix: np.ndarray):
        """
        Converts a QUBO matrix into a specific model format using VariableGenerator.
        """
        size = QUBO_matrix.shape[0] # [n+m]
        
        gen = VariableGenerator()
        model = gen.matrix("Binary", size)
        q = model.quadratic

        for i in range(size):
            for j in range(size):
                q[i][j] = QUBO_matrix[i][j]

        return model

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
    
    def solve_with_client(self, client_type: str) -> tuple[NDArray[np.int_], int, int]:
        """
        Solves the Max3SAT problem using different solvers (FixStar, Gurobi, or D-Wave) 
        based on the provided client type and returns the solution, energy, and the number 
        of satisfied clauses.

        Args:
            client_type (str): The type of solver client ('FixStar', 'Gurobi', or 'D-Wave').

        Returns:
            tuple: A tuple containing:
                - NDArray[np.int_]: Solution vector (literal assignments).
                - int: Objective value (energy) of the best solution.
                - int: The number of clauses satisfied (calculated based on g(x, w) = -(K + z @ F(φ) @z).
        """
        
        # Client initialization and solving based on the client type
        try:
            if client_type == "FixStar":
                client = GetFixstarClient()
            elif client_type == "Gurobi":
                client = GetGurobiClient()
            elif client_type == "D-Wave":
                client = GetDWaveClient()
            else:
                raise ValueError("Unsupported client type. Choose FixStar', 'Gurobi', or 'D-Wave'.")
            
            result = solve(self.model, client)

            literal_result = np.array(list(result.best.values.values()))
            best_objective = result.best.objective

            # Calculate the number of clauses satisfied using the formula:
            # g(x, w) = -(K + z^T F(φ) z)
            max_clauses_satisfied = (self.problem.K + best_objective) * -1

            print(f"{client_type} x result:", literal_result)
            print(f"{client_type} min energy:", best_objective)
            print(f"Max clauses satisfied with {client_type}: ", max_clauses_satisfied)

            return literal_result, best_objective, max_clauses_satisfied

        except Exception as e:
            # Handle any unexpected errors gracefully and log them
            print(f"Error during solving with {client_type}: {str(e)}")
            # Return None or appropriate default values if an error occurs
            return np.array([]), None, None