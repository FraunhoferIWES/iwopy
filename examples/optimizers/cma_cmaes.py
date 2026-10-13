from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.cma import Optimizer_cma

if __name__ == "__main__":
    problem = QuadraticProblem()
    variable_ranges = problem.max_values_float() - problem.min_values_float()
    solver = Optimizer_cma(
        problem,
        sigma0=0.25 * min(variable_ranges),  # Initial search scale in variable units
        cma_pars={
            "maxiter": 50,  # Demo generations; production start: 1000
            "tolx": 1e-6,  # Variable convergence tolerance
            "seed": 42,  # Reproducible sampling
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
