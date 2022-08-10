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
