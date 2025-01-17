import re
import numpy as np
import time
from numpy.typing import NDArray
import sympy

class Max3SAT:
    """
    A class to convert a Max 3-SAT problem into a QUBO representation.

    Attributes:
        num_literals (int): The number of literals in the problem.
        num_clauses (int): The number of clauses in the problem.
        clauses (NDArray[np.int_]): The clauses of the Max 3-SAT problem.
        QUBO_matrix (NDArray[np.int_]): The QUBO representation of the problem.
        K (int): The constant term in the QUBO representation.
    """

    def __init__(self, num_literals: int, clauses: NDArray[np.int_]):
        self.num_literals = num_literals
        self.num_clauses = clauses.shape[0]
        self.clauses = clauses
        self.QUBO_matrix, self.K, self.time_QUBO_formulation = self.problem_3SAT_to_QUBO(self.clauses, self.num_literals)
        self.nonzero_element_percentage = np.count_nonzero(self.QUBO_matrix) / (self.QUBO_matrix.shape[0] * self.QUBO_matrix.shape[1]) * 100

    def problem_3SAT_to_QUBO(self, clauses: NDArray[np.int_] = None, num_literals: int = None) -> tuple[NDArray[np.int_], int]:
        """
        Converts a Max 3-SAT problem to its QUBO representation.

        Args:
            clauses (NDArray[np.int_]): A 2D array where each row represents a clause. If None, uses self.clauses.
            num_literals (int): The number of literals in the problem. If None, uses self.num_literals.

        Returns:
            tuple[NDArray[np.int_], int]: A tuple containing the QUBO matrix and the constant term K and time took to formulate.
        """

        # Use instance attributes if arguments are not provided
        if clauses is None:
            clauses = self.clauses
        if num_literals is None:
            num_literals = self.num_literals

        time_QUBO_formulation = 0
        create_QUBO_start = time.time()

        num_x = num_literals
        x = sympy.symbols(f'x0:{num_x}')  # x0, x1, ... x_num_x-1
        num_m = clauses.shape[0]  # Number of clauses
        w = sympy.symbols(f'w0:{num_m}')  # w0, w1, ... w_num_m-1
        QUBO_matrix = np.zeros((num_x + num_m, num_x + num_m), dtype=int)
        sum_g = 0
        K = 0

        for i in range(num_m):
            x_array = clauses[i]
            if np.abs(x_array).max() > num_x:
                raise ValueError("Clause literals exceed the specified number of literals.")

            y_i1 = x[x_array[0] - 1] if x_array[0] > 0 else (1 - x[-x_array[0] - 1])
            y_i2 = x[x_array[1] - 1] if x_array[1] > 0 else (1 - x[-x_array[1] - 1])
            y_i3 = x[x_array[2] - 1] if x_array[2] > 0 else (1 - x[-x_array[2] - 1])
            sum_g += y_i1 + y_i2 + y_i3 + (w[i] * (y_i1 + y_i2 + y_i3)) - (y_i1 * y_i2) - (y_i1 * y_i3) - (y_i2 * y_i3) - 2 * w[i]

        sum_neg_g = sympy.simplify(-1 * sum_g)
        sum_neg_g_dict = {str(term): coefficient for term, coefficient in sum_neg_g.as_coefficients_dict().items()}

        for term, coefficient in sum_neg_g_dict.items():
            if re.match(r'^w\d+$', term):  # w[i]
                i = int(term[1:]) + num_x
                QUBO_matrix[i, i] = coefficient
            elif re.match(r'^x\d+$', term):  # x[i]
                i = int(term[1:])
                QUBO_matrix[i, i] = coefficient
            elif match := re.match(r'^w(\d+)\*x(\d+)$', term):  # w[i] * x[j]
                w_idx, x_idx = map(int, match.groups())
                QUBO_matrix[x_idx, w_idx + num_x] = coefficient
            elif match := re.match(r'^x(\d+)\*x(\d+)$', term):  # x[i] * x[j]
                x1, x2 = map(int, match.groups())
                QUBO_matrix[x1, x2] = coefficient
            elif term == '1':  # Constant term
                K = coefficient

        create_QUBO_end = time.time()
        time_QUBO_formulation = create_QUBO_end - create_QUBO_start

        print(f"Finished QUBO formulation. [{self.num_literals}, {self.num_clauses}]")
        print("QUBO formulation time:", time_QUBO_formulation)

        return QUBO_matrix, K, time_QUBO_formulation
    
    def verify(self, result_values: NDArray[np.int_], clauses: NDArray[np.int_] = None) -> tuple[bool, int, list]:
        """
        Verifies if the solution satisfies the given clauses.

        Args:
            result_values (NDArray[np.int_]): A 1D array representing the assignment of literals. This is a required argument.
            clauses (NDArray[np.int_]): A 2D array where each row represents a clause. If None, uses self.clauses.\

        Returns:
            tuple[bool, int, list]: A tuple containing:
                - A boolean indicating whether all clauses are satisfied.
                - The number of satisfied clauses.
                - A list of unsatisfied clauses.
        """

        # Use instance's clauses if argument is not provided
        if clauses is None:
            clauses = self.clauses

        num_clauses_True = 0
        wrong_clause = []

        for clause in clauses:
            clause_Truth_value = False
            for literal in clause:
                if literal > 0 and result_values[literal - 1] == 1:
                    clause_Truth_value = True
                    break
                elif literal < 0 and result_values[abs(literal) - 1] == 0:
                    clause_Truth_value = True
                    break
            if clause_Truth_value:
                num_clauses_True += 1
            else:
                wrong_clause.append(clause)
        return (num_clauses_True == clauses.shape[0]), num_clauses_True, wrong_clause
    