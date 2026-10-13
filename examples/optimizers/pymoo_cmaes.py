from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pymoo",  # Optimizer family
        optimizer="CMAES",  # Covariance-matrix adaptation
        problem=problem,
        x0=problem.initial_values_float(),  # Problem-defined starting point
        sigma=0.25,  # Initial scale in normalized variable coordinates
        verb_log=0,  # Disable native file logging
        vectorize=True,  # Evaluate each generation as an iwopy population
        seed=42,  # Reproducible sampling
        n_max_gen=10,  # Demo generations; production start: 1000
        ftol=1e-6,  # Objective-space convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
