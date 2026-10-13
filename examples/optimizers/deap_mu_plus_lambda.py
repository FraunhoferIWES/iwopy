from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="deap",  # Optimizer family
        optimizer="eaMuPlusLambda",  # Select survivors from parents and offspring
        problem=problem,
        cxpb=0.5,  # Crossover probability; cxpb + mutpb must not exceed one
        mutpb=0.2,  # Mutation probability
        vectorize=True,  # Evaluate offspring as an iwopy population
        batch_size=20,  # Demo population/offspring; production start: 100
        seed=42,  # Reproducible initialization and variation
        budget=400,  # Demo evaluations; production start: 10000+
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
