from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.optuna import Optimizer_optuna

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_optuna(
        problem,
        sampler="TPESampler",
        budget=24,
        batch_size=4,
        seed=42,
        vectorize=True,
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
