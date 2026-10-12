from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    import pygmo

    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "nlopt", "optimizer": "auglag", "maxeval": 100},
        setup_pars={"pop_size": 20, "seed": 42, "verbosity": 0},
    )
    solver.initialize(verbosity=0)
    local_optimizer = pygmo.nlopt("lbfgs")
    local_optimizer.maxeval = 100
    solver.algo.extract(pygmo.nlopt).local_optimizer = local_optimizer
    result = solver.solve(verbosity=0)
    print(result)
