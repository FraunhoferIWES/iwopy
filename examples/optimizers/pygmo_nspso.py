from iwopy.benchmarks import ParetoProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="nspso",  # Non-dominated sorting particle swarm
        problem=problem,
        omega=0.6,  # Inertia coefficient
        diversity_mechanism="crowding distance",  # Pareto diversity measure
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo Pareto swarm; production start: 100+
        max_iterations=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
