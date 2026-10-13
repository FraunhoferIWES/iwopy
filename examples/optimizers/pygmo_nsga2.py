from iwopy.benchmarks import ParetoProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="nsga2",  # Non-dominated sorting genetic algorithm
        problem=problem,
        cr=0.95,  # Crossover probability
        m=1.0 / problem.n_vars_float,  # Per-gene mutation probability
        seed=42,  # Reproducible evolution
        pop_size=20,  # Production: 100+ individuals, in multiples of four
        max_iterations=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
