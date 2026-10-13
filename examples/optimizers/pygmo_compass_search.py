from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "compass_search",  # Derivative-free coordinate search
            "max_fevals": 200,  # Demo evaluations; production start: 10000
            "start_range": 0.1,  # Initial step as a fraction of variable ranges
            "stop_range": 1e-6,  # Minimum step as a fraction of variable ranges
        },
        setup_pars={
            "pop_size": 1,  # One problem-defined starting point
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
