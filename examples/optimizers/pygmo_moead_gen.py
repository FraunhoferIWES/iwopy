from iwopy.benchmarks import ParetoProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="moead_gen",  # Generational multi-objective decomposition
        problem=problem,
        neighbours=5,  # Production: 20 neighbours with a larger population
        decomposition="tchebycheff",  # Scalar subproblem formulation
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo Pareto population; production start: 100+
        gen=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
