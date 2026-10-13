from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.nevergrad import Optimizer_nevergrad

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_nevergrad(
        problem,
        algorithm="OnePlusOne",  # One-plus-one evolutionary search
        budget=100,  # Demo budget; production start: 10000+ evaluations
        batch_size=1,  # Update the incumbent after each evaluated candidate
        seed=42,  # Reproducible sampling
        vectorize=True,  # Evaluate each batch as an iwopy population
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
