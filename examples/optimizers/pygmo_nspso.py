from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "nspso", "gen": 3, "seed": 42},
        setup_pars={"pop_size": 20, "seed": 42, "verbosity": 0},
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
