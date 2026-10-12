from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy_de

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy_de(
        problem,
        vectorize=True,
        scipy_pars={"seed": 42, "popsize": 5, "maxiter": 10},
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
