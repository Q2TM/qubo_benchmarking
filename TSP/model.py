from typing import Literal
from qiskit_optimization.converters import QuadraticProgramToQubo
from TSP.core import create_amplify_qubo_model, create_qiskit_qubo_model, reorder
from TSP.graph import draw_tour
from Utils.graph import draw_graph, generate_random_symmetric_matrix



class Tsp_Qiskit:
    def __init__(self):
        self.final_distribution_bin = None
    
    def qubo(self, penalty: int = 1000):
        qp = create_qiskit_qubo_model(self.matrix)
        qp2qubo = QuadraticProgramToQubo(penalty=penalty)
        qubo = qp2qubo.convert(qp)
        qubitOp, offseet = qubo.to_ising()
        return {"model": qubitOp, "offset": offseet}
    
    def intepret(self, result, verbose: bool = False):
        pass
    
    def plot_distribution(self, n: int = 0):
        pass

class Tsp(Tsp_Qiskit):
    def __init__(self, n: int, min: int = 1, max: int = 9, solver: Literal['Qiskit', 'Amplify'] = 'Amplify', seed: int = None, draw: bool = False):
        if solver == 'Qiskit':
            super().__init__(self.matrix)
        self.n = n
        self.solver = solver
        self.matrix = generate_random_symmetric_matrix(n, min=min, max=max, seed=seed)
        
        if draw:
            labels = [str(i) for i in range(n)]
            self.graph = draw_graph("TSP cities", labels, [self.matrix], ["d"])
            
    @property
    def max_edge_weight(self):
        return max(map(max, self.matrix))

    @property
    def avg_edge_weight(self):
        return sum(map(sum, self.matrix)) / (self.n ** 2)
            
    def draw_tour(self, tour):
        draw_tour("TSP Tour", [self.matrix], list(map(str, tour)))
        
    def get_cost(self, tour):
        return sum([self.matrix[tour[i-1]][tour[i]] for i in range(self.n)])
                    
    def qubo(self, penalty = 1000):
        if self.solver == 'Qiskit':
            return super().qubo(penalty)
        else:
            return create_amplify_qubo_model(self.matrix, penalty)
    
    def interpret(self, result) -> tuple[tuple, float, float]:
        if self.solver == 'Qiskit':
            return super().intepret(result)
        else:
            objective = result.best.objective
            time = result.solutions[0].time.total_seconds()

            return reorder(list(result.best.values.values()), self.n), objective, time

    
if __name__ == "__main__":
    tsp = Tsp(8, draw=True)
    print(tsp.qubo())
    print(tsp.matrix)