from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pymoo",  # Optimizer family
        optimizer="GA",  # Genetic algorithm
        problem=problem,
        vectorize=True,  # Evaluate each generation as an iwopy population
        pop_size=20,  # Demo population; production start: 100
        seed=42,  # Reproducible sampling and variation
        n_max_gen=10,  # Demo generations; production start: 1000
        ftol=1e-6,  # Objective-space convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
