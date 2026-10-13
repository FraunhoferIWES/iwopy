from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="COBYQA",  # Derivative-free quadratic-model optimization
        problem=problem,
        initial_tr_radius=1.0,  # Initial radius in variable units
        final_tr_radius=1e-6,  # Final trust-region radius
        max_iterations=100,  # Demo iterations; production start: 1000
        max_evaluations=500,  # Demo evaluations; production start: 10000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
