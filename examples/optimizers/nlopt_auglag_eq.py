from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    import pygmo

    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "nlopt",  # Select the NLopt backend
            "optimizer": "auglag_eq",  # Equality-focused augmented Lagrangian
            "maxeval": 500,  # Demo evaluations; production start: 10000
            "xtol_rel": 1e-6,  # Relative variable convergence tolerance
            "ftol_rel": 1e-8,  # Relative objective convergence tolerance
        },
        setup_pars={
            "pop_size": 1,  # One problem-defined starting point
        },
    )
    solver.initialize(verbosity=0)
    local_optimizer = pygmo.nlopt("lbfgs")
    local_optimizer.maxeval = 500  # Demo subproblem budget; production: 10000
    local_optimizer.xtol_rel = 1e-6  # Relative variable convergence tolerance
    solver.algo.extract(pygmo.nlopt).local_optimizer = local_optimizer
    result = solver.solve(verbosity=0)
    print(result)
