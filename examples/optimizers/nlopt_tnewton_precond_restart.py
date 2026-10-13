from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "nlopt",  # Select the NLopt backend
            "optimizer": "tnewton_precond_restart",  # Preconditioning and restarts
            "maxeval": 500,  # Demo evaluations; production start: 10000
            "xtol_rel": 1e-6,  # Relative variable convergence tolerance
            "ftol_rel": 1e-8,  # Relative objective convergence tolerance
        },
        setup_pars={
            "pop_size": 1,  # One problem-defined starting point
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
