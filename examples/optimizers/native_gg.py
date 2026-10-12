from iwopy.benchmarks import QuadraticProblem
from iwopy.optimizers import GG

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = GG(problem, step_max=0.5, step_min=1e-4, max_iterations=30)
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
