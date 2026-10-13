from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="gwo",  # Grey-wolf optimization
        problem=problem,
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo population; production start: 100
        max_iterations=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
