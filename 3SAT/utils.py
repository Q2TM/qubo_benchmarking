"""Utility functions for reading cnf files."""
import numpy as np
from numpy.typing import NDArray

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
