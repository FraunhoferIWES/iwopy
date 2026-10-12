from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "gwo", "gen": 3, "seed": 42},
        setup_pars={"pop_size": 20, "seed": 42, "verbosity": 0},
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
