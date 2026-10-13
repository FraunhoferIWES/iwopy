from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.optuna import Optimizer_optuna

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_optuna(
        problem,
        sampler="TPESampler",  # Tree-structured Parzen estimator
        budget=100,  # Demo trials; production start: 1000+, subject to cost
        batch_size=1,  # Incorporate each trial before proposing the next
        sampler_pars={
            "n_startup_trials": 20,  # Initial random trials before fitting TPE
        },
        seed=42,  # Reproducible sampling
        vectorize=True,  # Evaluate each batch as an iwopy population
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
