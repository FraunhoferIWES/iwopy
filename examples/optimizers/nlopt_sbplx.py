from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="nlopt",  # Optimizer family
        optimizer="sbplx",  # Subspace simplex search
        problem=problem,
        pop_size=1,  # One problem-defined starting point
        maxeval=500,  # Demo evaluations; production start: 10000
        xtol_rel=1e-6,  # Relative variable convergence tolerance
        ftol_rel=1e-8,  # Relative objective convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
