from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="nevergrad",  # Optimizer family
        optimizer="TBPSA",  # Population-size adaptation for noisy objectives
        problem=problem,
        vectorize=True,  # Evaluate each batch as an iwopy population
        batch_size=8,  # Outstanding asks per batch; no threads are created
        seed=42,  # Reproducible sampling
        budget=100,  # Demo budget; production start: 10000+ evaluations
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
