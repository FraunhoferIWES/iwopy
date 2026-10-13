from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="COBYLA",  # Derivative-free constrained optimization
        problem=problem,
        rhobeg=1.0,  # Initial trust-region radius in variable units
        max_evaluations=500,  # Demo evaluations; production start: 10000
        tol=1e-6,  # Final trust-region radius
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
