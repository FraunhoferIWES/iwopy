from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="nevergrad",  # Optimizer family
        optimizer="OnePlusOne",  # One-plus-one evolutionary search
        problem=problem,
        vectorize=True,  # Evaluate each batch as an iwopy population
        batch_size=1,  # Update the incumbent after each evaluated candidate
        seed=42,  # Reproducible sampling
        max_evaluations=100,  # Demo budget; production start: 10000+ evaluations
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
