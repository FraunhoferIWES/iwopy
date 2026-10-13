from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="optuna",  # Optimizer family
        optimizer="TPESampler",  # Tree-structured Parzen estimator
        problem=problem,
        n_startup_trials=20,  # Initial random trials before fitting TPE
        vectorize=True,  # Evaluate each batch as an iwopy population
        batch_size=1,  # Incorporate each trial before proposing the next
        seed=42,  # Reproducible sampling
        max_evaluations=100,  # Demo trials; production start: 1000+, subject to cost
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
