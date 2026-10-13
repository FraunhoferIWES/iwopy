from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="compass_search",  # Derivative-free coordinate search
        problem=problem,
        start_range=0.1,  # Initial step as a fraction of variable ranges
        stop_range=1e-6,  # Minimum step as a fraction of variable ranges
        pop_size=1,  # One problem-defined starting point
        max_evaluations=200,  # Demo evaluations; production start: 10000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
