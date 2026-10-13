from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="de1220",  # Adapt differential-evolution strategies
        problem=problem,
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo population; production start: 100
        max_iterations=10,  # Demo generations; production start: 1000
        ftol=1e-6,  # Population objective-spread stopping tolerance
        xtol=1e-6,  # Population variable-spread stopping tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
