from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="differential_evolution",  # Algorithm to use
        problem=problem,
        vectorize=True,  # Evaluate each generation as an iwopy population
        popsize=5,  # Demo multiplier per free variable; production: 15
        seed=42,  # Reproducible initialization and variation
        maxiter=30,  # Demo generations; production start: 1000
        tol=1e-6,  # Relative population-energy convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
