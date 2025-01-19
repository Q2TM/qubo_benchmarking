"""Utility functions for reading cnf files."""
import numpy as np
from numpy.typing import NDArray
from pysat.formula import CNF
from pysat.solvers import Solver

def read_cnf_file(filename: str) -> tuple[NDArray[np.int_], int, int]:
    """Read a CNF file and return the clauses array, number of variables, and number of clauses."""
    clauses = []
    num_literals = 0
    num_clauses = 0

    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            # Skip comment lines
            if line.startswith('c'):
                continue
            
            # Read the problem line to get the number of variables
            if line.startswith('p cnf'):
                parts = line.split()
                num_literals = int(parts[2])
                num_clauses = int(parts[3])
                continue
            
            # Read clauses
            clause = [x for x in line.split() if x != "0"]  # Skip the trailing 0
            if (len(clause) == 3):
                clauses.append(clause)
    
    return np.array(clauses, dtype="int"), num_literals, num_clauses

def read_cnf_file_custom(problem_num: int, num_literals: int, num_clauses: int, file_literals: int, file_clauses: int) -> tuple[NDArray[np.int_], int, int]:
    """Read a CNF file and return the clauses array, number of variables, and number of clauses.
    This function is used to create a custom test sets that is not from SATLIB.
    Need to input the number of literals and clauses from the used file.
    File used needs to be larger than the specified number of literals and clauses.
    Ex. A problem of 100 literals and 430 clauses can be generated from a file with 200 literals and 860 clauses.
    There's no garanty that the specified number of literals and clauses can be created from the chosen file. ie. error"""
    filename = f'TestSets\\Satisfiable\\uf{file_literals}-{file_clauses}\\uf{file_literals}-{problem_num}.cnf'
    clauses = []

    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            # Skip comment lines
            if line.startswith('c'):
                continue
            
            # Read clauses
            clause = [x for x in line.split() if x != "0"]  # Skip the trailing 0
            if (len(clause) == 3 and all((int(element) <= num_literals and -num_literals <= int(element)) for element in clause)):
                clauses.append(clause)
            
            if (len(clauses) == num_clauses):
                return np.array(clauses, dtype="int"), num_literals, num_clauses
    
    return "error", len(clauses), np.array(clauses, dtype="int")

def generator_3SAT(num_literals: int, num_clauses: int) -> tuple[NDArray[np.int_], int, int]:
    """Generate a random satisfiable 3SAT problem with the given number of literals and clauses.
    use only for small problems."""
    satisfiable = False
    while not satisfiable:
        clauses = []
        clause_set = set()  # Use a set to ensure unique clauses
        while len(clauses) < num_clauses:
            clause = set()  # Use a set to ensure unique literals
            while len(clause) < 3:
                literal = np.random.randint(1, num_literals + 1)
                if np.random.rand() < 0.5:
                    literal = -literal
                clause.add(literal)
            clause = sorted(clause)
            clause_tuple = tuple(clause)  # Convert to a tuple to allow comparison in a set
            
            if clause_tuple not in clause_set:  # Only add unique clauses
                clause_set.add(clause_tuple)
                clauses.append(clause)
        satisfiable = is_satisfiable_pysat(np.array(clauses, dtype="int"))  # Check if satisfiable
    return np.array(clauses, dtype="int"), num_literals, num_clauses

def is_satisfiable_pysat(clauses: np.ndarray) -> bool:
    cnf = CNF()
    cnf.extend(clauses.tolist())  # Convert clauses to a format PySAT understands
    with Solver(bootstrap_with=cnf) as solver:
        return solver.solve()