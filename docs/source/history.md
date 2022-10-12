# History

## v0.0.11-alpha

This is the initial release of **iwopy** - ready for testing.

So far not many models have been transferred from the Fraunhofer IWES in-house predecessor *iwes-opt*, they will be added in the following versions. We are just getting started here!

Enjoy - we are awaiting comments and issues, thanks for testing.

**Full Changelog**: https://github.com/FraunhoferIWES/iwopy/commits/v0.0.11

## v0.0.12-alpha

- Benchmarks:
    - New benchmark `rosenbrock` added

- Tests:
    - Extended tests for pygmo and pymoo

- Fixes:
    - Bugs fixed with discretization
    
**Full Changelog**: https://github.com/FraunhoferIWES/iwopy/commits/v0.0.12

## v0.0.13-alpha

- Wrappers:
    - Introducing `SimpleObjective` and `SimpleConstraint`, for functions of simple numeric types like `f(x, y, ...)`
- Examples:
    - New example: `electrostatics`, solvable with IPOPT, GA, PSO
- Notebooks:
    - Work on `examples.ipynb`: Adding section on simple function minimization
    
**Full Changelog**: https://github.com/FraunhoferIWES/iwopy/commits/v0.0.13

## v0.0.14-alpha

- Notebooks:
    - Work on `examples.ipynb`: Adding electrostatics example
- Core:
    - `OptResults` can now be printed

**Full Changelog**: https://github.com/FraunhoferIWES/iwopy/commits/v0.0.14

## v0.0.15-alpha

- Utils:
    - `RegularDiscretizationGrid` now with interpolation parameter: None, nearest or linear
- Optimizers:
    - New optimizer: `GG`, Greedy Gradient. A pretty straight forward gradient based local optimizer that projects out (or reverses) directions of constraint violation. Mostly implemented for the purpose of testing gradients, but maybe worth a try also for problem solving.
- Examples:
    - `electrostatics`: New optional constraint `MinDist`, forcing charges to keep a minimal distance. Also adding a script that demonstrates how to solve this problem using `NLOPT` via `pygmo`, and another for solving the problem using the `GG` algorithm.

**Full Changelog**: https://github.com/FraunhoferIWES/iwopy/commits/v0.0.15
