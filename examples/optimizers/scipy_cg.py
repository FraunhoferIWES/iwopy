from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": "CG", "options": {"maxiter": 150}},
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
