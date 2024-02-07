# History

## v0.0.11-alpha

This is the initial release of **iwopy** - ready for testing.

So far not many models have been transferred from the Fraunhofer IWES in-house predecessor *iwes-opt*, they will be added in the following versions. We are just getting started here!

Enjoy - we are awaiting comments and issues, thanks for testing.

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.0.11](https://github.com/FraunhoferIWES/iwopy/commits/v0.0.11)

## v0.0.12-alpha

- Benchmarks:
  - New benchmark `rosenbrock` added
- Tests:
  - Extended tests for pygmo and pymoo
- Fixes:
  - Bugs fixed with discretization

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.0.12](https://github.com/FraunhoferIWES/iwopy/commits/v0.0.12)

## v0.0.13-alpha

- Wrappers:
  - Introducing `SimpleObjective` and `SimpleConstraint`, for functions of simple numeric types like `f(x, y, ...)`
- Examples:
  - New example: `electrostatics`, solvable with IPOPT, GA, PSO
- Notebooks:
  - Work on `examples.ipynb`: Adding section on simple function minimization

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.0.13](https://github.com/FraunhoferIWES/iwopy/commits/v0.0.13)

## v0.0.14-alpha

- Notebooks:
  - Work on `examples.ipynb`: Adding electrostatics example
- Core:
  - `OptResults` can now be printed

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.0.14](https://github.com/FraunhoferIWES/iwopy/commits/v0.0.14)

## v0.0.15-alpha

- Utils:
  - `RegularDiscretizationGrid` now with interpolation parameter: None, nearest or linear
- Wrappers:
  - New wrapper: `LocalFD`, Local Finite Difference. Calculates function derivatives by applying local finite difference rules. Currently orders 1 and 2 are implemented, more could be added later.
- Optimizers:
  - New optimizer: `GG`, Greedy Gradient. A pretty straight forward gradient based local optimizer that projects out (or reverses) directions of constraint violation. Mostly implemented for the purpose of testing gradients, but maybe worth a try also for problem solving.
- Examples:
  - `electrostatics`: New optional constraint `MinDist`, forcing charges to keep a minimal distance. Also adding a script that demonstrates how to solve this problem using `NLOPT` via `pygmo`, and another for solving the problem using the `GG` algorithm.

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.0.15](https://github.com/FraunhoferIWES/iwopy/commits/v0.0.15)

## v0.1.0-alpha

- Bug fixes
- Updated documentation

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.0](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.0)

## v0.1.1-alpha

- Tests:
  - pygmo: Increased ngen and npop, intended for osx systems to pass tests

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.1)

## v0.1.2-alpha

- Tests:
  - pygmo: Fixing test such that it passes on all systems

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.2](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.2)

## v0.1.3-alpha

- Core:
  - Support for multi objective optimizations
  - Splitting `OptResults` into `SingleObjOptResults` and `MultiObjOptResults`
- Interfaces:
  - `pymoo`: Adding `nsga2`
- Examples:
  - New: `multi_obj_chain`, demonstrating multi objective optimization using pymoo's NSGA2
- Notebooks:
  - New: `multi_obj_chain.ipynb`, demonstrating multi objective optimization using pymoo's NSGA2
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.3](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.3)

## v0.1.4-alpha

- Interfaces:
  - Improved: `pymoo` now supports mixed int/float problems and algorithm `MixedVariableGA`
  - Changed: `pymoo`algorithms now selectable via same name as in package, i.e., GA, PSO, NSGA2, MixedVariableGA
- Examples:
  - New: `mixed`, demonstrating how to solve a mixed int/float problem with pymoo
- Notebook:
  - New: `mixed`, demonstrating how to solve a mixed int/float problem with pymoo
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.4](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.4)

## v0.1.5-alpha

- Small bug fixes
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.5](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.5)

## v0.1.6-alpha

- Interfaces
  - New interface to `scipy`: Calling the `scipy.optimize.minimize` function
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.6](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.6)

## v0.1.7-beta

- Utils
  - New `import_module` utility, dynamically loading packages and modules
- Interfaces
  - Improved dynamic imports of `pygmo` and `pymoo`
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.7](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.7)

## v0.1.8-beta

- Bug fixed in setup.cfg
  
**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.8](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.8)

## v0.1.9-beta

- Bug fixes
  - Fixes for bugs in tests for Python 3.7 and 3.11

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.1.9](https://github.com/FraunhoferIWES/iwopy/commits/v0.1.9)