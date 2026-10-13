from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="pso",  # Particle-swarm optimization
        problem=problem,
        omega=0.7298,  # Inertia or constriction coefficient
        eta1=2.05,  # Attraction toward personal best positions
        eta2=2.05,  # Attraction toward neighbourhood best positions
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo swarm; production start: 40
        max_iterations=10,  # Demo generations; production start: 1000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
