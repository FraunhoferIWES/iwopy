from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.cma import Optimizer_cma

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_cma(
        problem,
        sigma0=0.3,
        cma_pars={"seed": 42, "popsize": 8, "maxiter": 5},
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
