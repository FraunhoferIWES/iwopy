from iwopy.benchmarks import QuadraticProblem
from iwopy.optimizers import GG

if __name__ == "__main__":
    problem = QuadraticProblem()
    variable_ranges = problem.max_values_float() - problem.min_values_float()
    solver = GG(
        problem,
        step_max=0.1 * variable_ranges,  # Initial steps: 10% of variable ranges
        step_min=1e-6 * variable_ranges,  # Minimum steps in variable units
        f_tol=1e-8,  # Objective improvement stopping tolerance
        max_iterations=100,  # Demo iterations; production start: 1000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
