from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.deap import Optimizer_deap

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_deap(
        problem,
        algorithm="eaMuPlusLambda",  # Select survivors from parents and offspring
        budget=400,  # Demo evaluations; production start: 10000+
        batch_size=20,  # Demo population/offspring; production start: 100
        cxpb=0.5,  # Crossover probability; cxpb + mutpb must not exceed one
        mutpb=0.2,  # Mutation probability
        seed=42,  # Reproducible initialization and variation
        vectorize=True,  # Evaluate offspring as an iwopy population
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
