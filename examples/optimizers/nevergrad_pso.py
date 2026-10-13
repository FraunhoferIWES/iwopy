from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.nevergrad import Optimizer_nevergrad

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_nevergrad(
        problem,
        algorithm="PSO",  # Particle-swarm optimization
        budget=100,  # Demo budget; production start: 10000+ evaluations
        batch_size=8,  # Outstanding asks per batch; no threads are created
        seed=42,  # Reproducible sampling
        vectorize=True,  # Evaluate each batch as an iwopy population
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
