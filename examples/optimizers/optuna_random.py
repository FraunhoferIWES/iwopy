from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.optuna import Optimizer_optuna

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_optuna(
        problem,
        sampler="RandomSampler",  # Independent random-sampling baseline
        budget=100,  # Demo trials; production start: 1000+, subject to cost
        batch_size=8,  # Trials evaluated together per iwopy batch
        seed=42,  # Reproducible sampling
        vectorize=True,  # Evaluate each batch as an iwopy population
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
