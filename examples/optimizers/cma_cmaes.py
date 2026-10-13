from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    variable_ranges = problem.max_values_float() - problem.min_values_float()
    solver = create_optimizer(
        family="cma",  # Optimizer family
        optimizer="CMAES",  # Algorithm to use
        problem=problem,
        sigma0=0.25 * min(variable_ranges),  # Initial search scale in variable units
        seed=42,  # Reproducible sampling
        max_iterations=50,  # Demo generations; production start: 1000
        tolx=1e-6,  # Variable convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
