from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.deap import Optimizer_deap

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_deap(
        problem,
        algorithm="eaSimple",
        budget=24,
        batch_size=4,
        seed=42,
        vectorize=True,
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
