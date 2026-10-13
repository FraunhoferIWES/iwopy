from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="sga",  # Simple genetic algorithm
        problem=problem,
        cr=0.9,  # Crossover probability
        m=0.02,  # Per-gene mutation probability
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo population; production start: 100
        gen=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
