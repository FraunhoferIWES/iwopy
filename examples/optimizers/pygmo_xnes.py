from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="xnes",  # Exponential natural evolution strategy
        problem=problem,
        sigma0=0.25,  # Initial search step size
        force_bounds=True,  # Keep sampled candidates within variable bounds
        seed=42,  # Reproducible evolution
        pop_size=20,  # Production: max(5, 4 + int(3 * log(n_vars)))
        max_iterations=20,  # Demo generations; production start: 1000
        ftol=1e-6,  # Population objective-spread stopping tolerance
        xtol=1e-6,  # Search-distribution stopping tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
