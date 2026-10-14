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

## v0.2-beta

- Interfaces:
  - Improved `pymoo`: Dynamical derivation from classes from optional package import
- Documentation:
  - Completely new sphinx style

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.2](https://github.com/FraunhoferIWES/iwopy/commits/v0.2)

## v0.2.1

- Documentation improvements

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.2.1](https://github.com/FraunhoferIWES/iwopy/commits/v0.2.1)

## v0.2.2

- Restricting numpy to <2, due to dependencies

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.2.2](https://github.com/FraunhoferIWES/iwopy/commits/v0.2.2)

## v0.2.3

- Recovering support for Python 3.7

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.2.3](https://github.com/FraunhoferIWES/iwopy/commits/v0.2.3)

## v0.3

- removing restriction numpy <2, since now latest numpy is supported again by indirect dependency `autograd`

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.3](https://github.com/FraunhoferIWES/iwopy/commits/v0.3)

## v0.3.1

- Added factories for allowing automatized setups

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.3.1](https://github.com/FraunhoferIWES/iwopy/commits/v0.3.1)

## v0.4.0

- Dropping support for Python 3.8

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.4.0](https://github.com/FraunhoferIWES/iwopy/commits/v0.4.0)

## v0.5.0

- Core:
  - Added optimization pipelines through the new abstract `Pipeline` base class.
  - Added native optimizer callbacks with normalized intermediate states for
    `GG`, `SLSQP`, SciPy, pymoo, and PyGMO. Callback validation and finalization
    are now consistent, callback-free runs avoid adapter overhead, and unsupported
    PyGMO IPOPT callbacks fail explicitly.
  - Unified problem summaries across optimizer output.
- Optimizers:
  - Added `SLSQP` for constrained single-objective problems, with automatic
    dimensionless scaling, vectorized iwopy Jacobians, evaluation-free progress
    output, and reuse of cached function values in `LocalFD`.
  - Hardened `GG` for zero and non-finite gradients, general constraint bounds,
    stalled constraints, hard iteration limits, and convergence edge cases.
- Interfaces:
  - Extended pymoo factories with `DE`, `NSGA3`, and `CMAES`, and improved
    integer offspring handling, reusable configurations, and bounded
    single-objective termination.
  - Expanded PyGMO algorithm support and made IPOPT option handling more robust.
- Problems:
  - Added `DiscreteLocalMoveOptProblem` for applying one small discrete move per
    wind turbine.
- Examples:
  - Added runnable SLSQP and pymoo examples for general constraints and callbacks,
    including live progress and objective-history plots.
- Bugs:
  - Fixed lower-bound second-order finite differences in `LocalFD`, maximization
    gradient orientation in PyGMO, and callback leakage after PyGMO solves.
  - Fixed regular-grid index and interpolation error handling, SciPy mixed-variable
    bounds and cache eviction, disabled pymoo problem-result storage access, and
    empty PyGMO gradient sparsity handling.
  - Rejected boolean values for integer-only `GG` iteration settings and `GG` or
    `SLSQP` memory sizes.
  - Fixed package-version fallback lookup on Python versions without `tomllib`.
- Dependencies:
  - Added the `iwopy[scipy]`, `iwopy[pymoo]`, `iwopy[pygmo]`, and `iwopy[opt]`
    extras; dropped Python 3.9 and raised runtime, PyGMO, test, and documentation
    dependency floors.
- Development:
  - Added package-wide type annotations and strict mypy checking, enforced by
    pre-commit.
  - Standardized API docstrings and Sphinx configuration on the foxes NumPyDoc
    style, with a Python 3.10 TOML fallback for documentation builds.

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v0.5.0](https://github.com/FraunhoferIWES/iwopy/commits/v0.5.0)

## v1.0.0

- Documentation:
  - Corrected chain constraint counts, callback failure semantics, and the GG
    architecture link; aligned token-review guidance with the outside-pytest policy
- Python versions:
  - Added support for Python 3.14
- Core:
  - Replaced dense Jacobian assembly and optimizer caches with SciPy CSR storage.
    Matrix APIs are now `Problem.get_jacobian()` and `calc_jacobian()`;
    scalar gradients and backend-required native names retain their meaning.
    Analytical and numerical assembly preserves implicit zeros, missing-derivative
    diagnostics, mapped variables, and selected component/variable order.
  - Added standalone `iwopy.create_optimizer` selection by backend family
    and algorithm, including a PyGMO-backed `nlopt` family. All maintained
    examples and notebooks import it from the root; core owns the same function,
    and public optimizer class names are unchanged.
  - Added restart result seeding for selected pipeline stage ranges and ensured
    stage-reported failures clear running state before finalization.
- Benchmarks:
  - Published `QuadraticProblem`, `ParetoProblem`, and `MixedProblem` through
    `iwopy.benchmarks`, replacing the optimizer examples' local shared-case module.
    All catalog scripts now use installed-package imports without path setup.
- Interfaces:
  - Standardized supported run limits as `max_iterations` and `max_evaluations`,
    with native counter units, unsupported-limit errors, and no legacy aliases.
    Kept algorithmic `pop_size`, pending-work `batch_size`, and SciPy DE's
    `population_multiplier` distinct; DEAP now exposes generation limits.
    Composed pymoo limits with custom criteria and added scalar pycma evaluation.
    Published shared-parameter and termination explanation/support tables and
    migrated maintained iwopy and FOXES Optimization callers.
  - Kept `trust-constr` constraint Jacobians sparse through the SciPy callback,
    defaulting native `sparse_jacobian` to true while preserving explicit options.
    SLSQP retains its required dense callback boundary; PyGMO reads sparse
    fitness-Jacobian entries without densification.
  - Replaced optimizer parameter dictionaries with explicit per-class controls
    and flat native keyword forwarding. Aligned `optimizer`, `vectorize`,
    `memory_size`, and absolute `pop_size` where applicable; SciPy DE preserves
    a per-dimension `population_multiplier`. Removed the class factory entry point
    and old signatures without compatibility paths.
  - Added population-vectorized SciPy differential evolution with mixed integer/
    float variables, cached nonlinear constraint evaluations, and callbacks.
  - Added lazy optional Nevergrad, Optuna, and DEAP interfaces with bounded
    single-objective population evaluation, general constraints, evaluated-candidate
    results, and deterministic batching/lifecycle tests. The Nevergrad extra
    constrains NumPy below 2.4 for upstream metamodel compatibility.
  - Added default-true `vectorize` flags to SciPy DE, Nevergrad, Optuna, and DEAP.
    Individual evaluation preserves native batches, budgets, constraints, callbacks,
    and selection while avoiding population evaluation.
  - Added the lazy optional `cma` interface with population-vectorized pycma
    CMA-ES, native augmented-Lagrangian constraints, constraint-owned
    tolerances, callbacks, finalized results, and termination diagnostics.
  - Added cached, population-vectorized iwopy Jacobians to every
    gradient-capable SciPy minimizer, with grouped general constraints and an
    explicit continuous-variable contract.
  - Added bounded vectorized finite-difference batches through
    `LocalFD.max_population_size`.
  - Constraint objects are now the sole tolerance owners; SLSQP and PyGMO
    derive component tolerances directly from registered constraints, and
    SLSQP applies them to translated solver bounds.
- Bugs:
  - Aligned GG progress rows with the completed iteration state reported to
    callbacks. Kept feasible GG iterates when all trial candidates violate constraints,
    backtracking instead of accepting an infeasible point with lower quality.
    Preserved recovery from initially infeasible layouts in both evaluation modes.
  - Accounted for floating-point roundoff at constraint feasibility-tolerance
    boundaries.
  - Fixed combined analytical Jacobians for functions with disjoint variable
    dependencies, restored constraint bounds through problem wrappers, and
    cleared SciPy value and Jacobian caches between solves. Skipped structurally independent components when assembling analytical Jacobians.
  - Avoided evaluating analytically differentiated function components during
    vectorized `LocalFD` Jacobians.
- Dependencies:
  - Required wheel >=0.46.2 for builds to exclude known path-traversal and
    regular-expression denial-of-service vulnerabilities.
- Development:
  - Added a complete optimizer catalog with 79 main-guarded factory-call
    scripts, package-aware execution/import tests, and selector coverage.
    Each row now includes capability classifications and a practical recommendation;
    detailed class references link to the API rather than being embedded.
    Split rows into non-empty objective/constraint/local-global categories and
    added constrained NSGA2/NSGA3 Pareto flows with explicit weighted selection,
    bringing the catalog to 81 runnable scripts.
    The Vectorized Interfaces overview now covers all adapters with equal API-linked
    descriptions and distinguishes search, Jacobian, and batch-fitness vectorization.
  - Reworked every optimizer setup with important controls on individually annotated
    value lines. Kept demo runs short and added production starting suggestions,
    protected by format/budget regressions. Local PyGMO solvers use one starting
    point; GG/SLSQP examples and documentation now identify their source as iwopy.
  - Added iwopy-specific agentic coding policy, architecture, naming,
    development, docstring, data-classification, decision, and Fraunhofer IWES
    visual-design guidance.

**Full Changelog**: [https://github.com/FraunhoferIWES/iwopy/commits/v1.0.0](https://github.com/FraunhoferIWES/iwopy/commits/v1.0.0)
