# iwopy Architecture

## Purpose

This document is the durable technical map of iwopy, the Fraunhofer IWES
optimization tools in Python. iwopy is an established in-process library for
defining optimization problems, objectives, constraints, derivatives, callbacks,
pipelines, and optimizer integrations. It is not an application scaffold and it
does not own a particular scientific domain.

Use [naming conventions](naming-conventions.md) for domain vocabulary, types,
array shapes, and identifiers. Use [docstring conventions](docstrings.md) for
public Python documentation and [development](development.md) for setup,
quality gates, test selection, documentation builds, and navigation. Record the
reason for a consequential change in an [ADR](adr/README.md).

## Authoritative Sources

When sources disagree, use this order:

1. `AGENTS.md` and any applicable scoped repository instructions
2. Accepted ADRs in `docs/adr/`, for decisions within those policies
3. This architecture document and `docs/naming-conventions.md`
4. Public contracts and configuration in `iwopy/` and `pyproject.toml`
5. Tests that exercise the contract
6. Examples, notebooks, and external references

Code and tests establish current runtime behavior and can expose documentation
drift, but they do not silently replace a documented durable decision. Resolve
the conflict and update the record in the same change.

## System Context

iwopy serves developers, researchers, and engineers who need one problem model
across iwopy optimizers and external solver libraries. A caller defines a
`Problem`, adds objective and constraint functions, selects an `Optimizer`, and
receives an iwopy result container. Interfaces translate the same problem into
SciPy, pycma, pymoo, PyGMO, Nevergrad, Optuna, or DEAP contracts.

```mermaid
flowchart LR
    Caller[Problem variables and application data] --> Problem
    Functions[Objectives and constraints] --> Problem
    Problem --> Optimizer
	Optimizer --> Iwopy[GG or SLSQP]
	Optimizer --> Interfaces[SciPy, pycma, pymoo, PyGMO, Nevergrad, Optuna, or DEAP]
    Native --> Results[Single or multi-objective results]
    Interfaces --> Results
    Optimizer --> Callbacks[Snapshots and optimization history]
```

- Runtime: Python 3.10 through 3.14 on operating-system-independent Python
	environments.
- Required scientific stack: NumPy, SciPy, and matplotlib.
- Default local development setup uses
	`uv sync --extra dev --extra test --extra opt --upgrade`, selecting contributor,
	test, and all optimizer integrations explicitly.
- Optional optimizer integrations: pycma, pymoo, PyGMO, Nevergrad, Optuna, and DEAP, selected through package
	extras. The `opt` and `test` extras aggregate supported solver integrations.
- iwopy is an in-process library. It has no application server, authentication
	boundary, persistent database, browser frontend, or package console script.
- External optimizer adapters own translation to backend variable, constraint,
	callback, and result conventions. Core problems remain backend-independent.

## Primary Execution Flow

1. Construct a `Problem` and declare integer and floating optimization variables.
2. Construct objectives and constraints and add them before problem
	initialization.
3. Initialize the problem, which fixes function/component metadata and variable
	mappings.
4. Construct an optimizer with `iwopy.create_optimizer(family, optimizer,
   problem, **kwargs)`, then initialize it separately.
5. Call `solve(callbacks=...)`; the optimizer evaluates individuals or
	populations through the problem contract.
6. Solver implementations finalize selected individuals or populations through
	the problem before constructing the result container. Optionally call
	`optimizer.finalize(opt_results)` afterward to print the result summary.

Examples may use `SimpleProblem` and simple functions to define compact callable
problems, or wrappers such as `LocalFD` and `DiscretizeRegGrid` to adapt an
existing problem. Those paths still use the same core lifecycle.

## Module Boundaries

| Module | Owns | Main interfaces and dependencies |
|---|---|---|
| `iwopy.core` | Base lifecycle, problems, functions, memory, optimizer creation, callbacks, results, and pipelines | `Base`, `Problem`, `OptFunction`, `Objective`, `Constraint`, `Optimizer`, `create_optimizer`, result containers, `Pipeline` |
| `iwopy.wrappers` | Reusable adaptations of problems and functions | `SimpleProblem`, `SimpleObjective`, `SimpleConstraint`, `ProblemWrapper`, `LocalFD`, `DiscretizeRegGrid` |
| `iwopy.interfaces.scipy` | Translation to SciPy optimization methods | SciPy optimizer adapter, bounds, constraints, callbacks, and Jacobians |
| `iwopy.interfaces._population` | Private shared contract for the four new bounded population adapters | Validation, float-before-integer conversion, constraint residuals, evaluated-candidate selection, callback snapshots, and finalization; no external backend imports |
| `iwopy.interfaces.cma` | Population-vectorized pycma CMA-ES integration | Continuous single-objective ask/tell, native augmented-Lagrangian constraints, callbacks, and finalized results |
| `iwopy.interfaces.nevergrad` | Nevergrad population ask/tell integration | Eight explicit batching-safe algorithm selectors, violation feedback, and evaluated-candidate results |
| `iwopy.interfaces.optuna` | Optuna population ask/tell integration | TPE/random samplers, fresh in-memory studies, current trial constraint API, and failed-batch cleanup |
| `iwopy.interfaces.deap` | Serial DEAP population evolution | Three variation/replacement schemes, bounded operators, integer repair, and constraint-first fitness |
| `iwopy.interfaces.pymoo` | Single- and multi-objective pymoo integration | Mixed/integer/float representations, algorithm factories, population evaluation, result conversion |
| `iwopy.interfaces.pygmo` | Serial PyGMO integration | UDP translation, algorithms, fitness/batch fitness, callbacks, and result conversion |
| `iwopy.optimizers` | iwopy optimizer implementations | `GG` and `SLSQP`; depends on core contracts and SciPy mathematics |
| `iwopy.benchmarks` | Reusable analytical example problems | Branin, Rosenbrock, and public continuous/Pareto/mixed quadratic problem definitions |
| `iwopy.utils` | Shared discretization, loading, subclass discovery, and output suppression | `RegularDiscretizationGrid` and focused infrastructure helpers |
| `examples` and `notebooks` | Executable public workflows | Public APIs and declared optimizer extras |
| `tests` | Core, optimizer, backend, package, and callback-example coverage | May inspect internals only when the internal contract itself is under test |

The package root exposes `create_optimizer`, `Problem`, `Objective`, `Constraint`, `Memory`, pipeline
and callback APIs, wrappers, and the principal subpackages. Lower-level
`Optimizer`, `OptFunction`, and optimization result classes are curated through
`iwopy.core`. Internal helpers do not become public merely because another
module imports them.

## Runtime Contracts

### Lifecycle And Registration

Core lifecycle-aware objects derive from `Base`. Initialize them before
evaluation or solving, and finalize them when resources or mutable problem state
must be released. `Base.finalize()` resets initialization state. Optimizer
callbacks instead implement their own `initialize`, `notify`, and `finalize`
protocol without deriving from `Base`.

Add objectives and constraints before problem initialization. Function lists
reject later additions because variable mappings, component names, bounds, and
counts are fixed by initialization. Optimizers verify that both optimizer and
problem are initialized before solving.

`ProblemWrapper.initialize()` shares and rebinds the underlying problem's
function lists to the wrapper. It is an adapter, not an independent deep copy;
document and test mutations that callers can observe through either object.

### Variables, Individuals, And Populations

iwopy separates integer and floating-point optimization variables:

- One individual uses `vars_int` with shape `(n_vars_int,)` and `vars_float`
	with shape `(n_vars_float,)`.
- A population uses `(n_pop, n_vars_int)` and `(n_pop, n_vars_float)`.
- A missing variable family uses a zero-width array, including for populations;
	the population axis is not omitted.
- Integer unbounded limits use `Problem.INT_INF` sentinels. Floating bounds use
	positive or negative `np.inf`.

Variable names and order are public. A function receives only its mapped
problem variables, in the function's declared order. Maps may use names,
indices, or uniquely matching patterns; ambiguous or missing matches are input
errors.

### Evaluation And Memory

Individual and population evaluation first applies variables to the problem,
then maps variables into each function and calculates objective and constraint
components. Evaluation returns `(objs, cons)` or
`(objs, cons, problem_results)` when application results are requested.
Finalization returns `(problem_results, objs, cons)`; preserve this distinct
ordering.

Objective values use `(n_objectives,)` for an individual and
`(n_pop, n_objectives)` for a population. Constraint values follow the same rule
with `n_constraints`. These counts refer to scalar components, not the number of
function objects.

Optional `Memory` caches objective and constraint values by variable assignment.
An evaluation requesting application-specific `problem_results` bypasses cached
lookup because those results are not reconstructed from cached function values.
Cache changes require tests for key identity, copying, and individual/population
equivalence.

### Functions, Bounds, And Feasibility

`OptFunction` is the shared function contract; `Objective` adds optimization
direction and `Constraint` adds component bounds and tolerances. Function lists
concatenate components in registration order.

Constraint bounds are component-wise. The default contract is
`-np.inf <= value <= 0.0` with tolerance `1e-5`. Feasibility checks both lower
and upper bounds with tolerance; a finite calculation is not necessarily a
feasible one.

Each `Constraint` is the sole owner of its scalar tolerance. `Problem` does not
cache a second tolerance vector. Backends that require component-wise values
expand the registered constraints in function/component order when their
adapter initializes. iwopy SLSQP applies those tolerances to its translated
bounds, while zero tolerance requests exact bounds. See
[ADR-0003](adr/0003-constraint-owned-solver-tolerances.md).

`Constraint.active_components()` may certify components as feasible throughout
enforced floating-variable bounds. `Problem.get_active_constraint_components()`
maps these selections to sorted global indices and rejects invalid selections.
The default retains all components. Selection never changes registered counts,
names, full evaluation, or public feasibility checks.

Simple objectives and constraints expose `f(*x)` and optional analytical
partial derivatives through `g(var, *x, components)`. Integer function arguments
precede floating arguments. Keep the
callable convention aligned with the function's variable mapping.

### Derivatives

`ana_deriv()` returns selected-component derivatives for one floating variable.
Unavailable analytical values use `NaN`, allowing the Jacobian assembler to
distinguish them from exact zero. Dependency masks identify structurally
independent component/variable pairs.

`Problem.get_jacobian()` returns a SciPy CSR matrix with shape
`(n_selected_components, n_selected_float_variables)`. `calc_jacobian()` assembles
analytical entries directly into sparse storage without allocating a dense
Jacobian. Independent entries and exact zeros are implicit; unresolved `NaN`
entries remain explicit until numerical wrapping resolves them or validation
rejects them. Default combined-function Jacobians order objective components
before constraint components. Selection order, including repeated variables,
is preserved. Use `.toarray()` only at a boundary requiring dense storage.

A scalar function's gradient is a vector over variables, not the complete
component-by-variable Jacobian. The matrix APIs use singular `jacobian` names;
the former plural-gradient matrix APIs are removed, without aliases. Dependency
masks currently remain dense boolean arrays.

`LocalFD` supplies numerical derivatives for wrapped functions, with bounded
finite-difference population batches where configured. Numerical wrapping must
preserve the original component and variable order. It fills only unresolved
stored entries, preserving analytical derivatives and structural zeros, and
reuses center values in selected-component order.

`Problem.get_jacobian(max_deltas=...)` accepts positive finite caps in complete
problem float-variable order. `LocalFD` uses the smaller of its configured delta
and the cap for that call without mutating configuration. Analytical and
fixed-grid derivatives are unchanged. A second-order one-sided stencil can
sample at twice the capped delta. GG supplies its current per-variable step at
existing fresh-Jacobian calls before resetting the step; cached backtracking
does not add Jacobian evaluations. See
[ADR-0017](adr/0017-gg-call-local-fd-caps.md).

### Results And Callbacks

`SingleObjOptResults` represents one selected solution. Variable, objective, and
constraint arrays can be absent when a backend fails to produce them.
`MultiObjOptResults` represents a Pareto population with leading `n_pop`; its
success array marks feasibility per candidate. Both containers retain variable
and component names and application-specific `problem_results`.

Optimizer callbacks follow `initialize`, ordered `notify`, and `finalize`.
`OptimizerCallbackData` normalizes snapshots to two-dimensional population
arrays, with integer variables stored as `int32` and numeric values as
`float64`. Events distinguish `iteration` and `evaluation`; counters and values
may be unavailable. `OptimizationHistory` records snapshots and plots best
objective values.

### Pipelines

`PipelineStage.run()` accepts `prev_stage`, `prev_results`, verbosity, and stage
parameters, and returns `(success, results)`. Stages own directories named
`NN_stage_name`. `Pipeline.run()` executes the selected half-open stage range,
stops after a failed stage, and optionally finalizes initialized stages. Changes
to result propagation must be tested directly; do not infer pipeline behavior
from a downstream application package.

`Pipeline.run(initial_results=...)` supplies an application-defined restart
payload to the first selected stage. A later-stage restart also identifies the
immediately preceding registered stage without running it. Run cleanup restores
the configured stage range and clears running state after success, a
stage-reported failure, or an exception. See
[ADR-0004](adr/0004-pipeline-restart-results.md).

## Optimizer And Backend Boundaries

### Creation And Configuration

`iwopy.core.optimizer_factory` owns the standalone `create_optimizer` function,
re-exported from `iwopy` and `iwopy.core` as the same function object. Maintained
examples and notebooks import it from the package root.
Its case-sensitive family/algorithm selection dispatches only to known iwopy
classes and returns an uninitialized optimizer. It neither initializes the
problem nor solves it. Maintained examples and notebooks use this function;
public optimizer class names and direct constructors remain available.

Families are `iwopy`, `scipy`, `cma`, `pymoo`, `pygmo`, `nlopt`, `nevergrad`,
`optuna`, and `deap`. SciPy minimizes by method name, callable, or default;
`differential_evolution` selects its separate population adapter. The `nlopt`
family uses PyGMO internally and supplies both its driver and native solver
selector, without requiring a nested selector from the caller.

Every optimizer has explicit parameters and, where appropriate, flat native
algorithm keywords. There are no common parameter dictionaries or class factory
entry points. `vectorize` governs iwopy population evaluation or Jacobian
batching, including ordered individual evaluation of pycma generations when
false. `pop_size` is an absolute search-population count; `batch_size` sizes
pending Nevergrad asks or Optuna trials, not an algorithmic population.
SciPy DE uses `population_multiplier` per free dimension.
`memory_size` controls visited-point memory for GG and evaluation caching for
SLSQP and SciPy minimization.
Adapter-private dictionaries translate explicit controls to backend contracts;
they are not a public grouped-configuration API. Numerical algorithms, callback
behavior, and dependency-loading timing remain adapter-owned. See
[ADR-0013](adr/0013-optimizer-creation.md).

`Optimizer` validates canonical `max_iterations` and `max_evaluations` controls;
adapters translate them only to meaningful native or adapter counters. Unsupported
limits fail rather than approximating iterations with evaluations. Native
convergence remains backend-specific; no global problem-evaluation interception
enforces a universal hard budget. pymoo composes explicit limits with custom
criteria using OR semantics. DEAP also exposes generation limits. Native
initialization, generation-boundary overshoot, and finalization counting are
documented in the public API table. Superseded limit names are rejected rather
than retained as aliases. This supersedes the pycma population-only and SciPy DE
native-name portions of ADR-0013; see
[ADR-0015](adr/0015-shared-optimizer-controls.md).

### iwopy Optimizers

`GG` is a constrained local Greedy Gradient optimizer. `SLSQP` uses SciPy with
iwopy Jacobians and scaling. Both require continuous, single-objective problems.
SLSQP derives and applies component tolerances from the registered constraint
objects. They own optimizer-specific convergence and step behavior, not problem
application semantics.

SLSQP retains the best finite, fully feasible evaluated point independently of
the raw primary SciPy endpoint. Returning that point does not turn failure into
convergence. Optional `feasibility_margin` reduces positive solver tolerances;
`restoration_iterations` bounds each pre/post restoration phase; and
`stall_iterations` stops repeated negligible cumulative best-feasible progress
using `progress_tol`, starting only after the first feasible point. Before
feasibility, native stopping criteria and iteration limits remain responsible.
Restoration minimizes full squared violations divided by explicit positive
`restoration_scales`, required when enabled for constrained problems. Scales
follow full constraint order and have each component's units; they must change
with constraint units. Restoration requests only violated-constraint Jacobian
rows, not objective derivatives.

The controls use the standard objective direction and evaluation contracts,
including for future scalar objectives. Progress tolerance is in evaluated
objective units; restoration uses an objective-independent constraint-merit
tolerance. Finalization may rescale returned function values.

With `reduce_constraints=True`, SLSQP assembles only bound-certified active rows
of the objective/constraint Jacobian and SciPy constraints. `move_limit` defines
a coordinate box around the initial point for the whole solve. The box and
selected rows are rebuilt between solves and stay fixed during each SciPy call.
All original constraints remain evaluated and checked. This is a local solve,
not an adaptive in-solve active-set method. See
[ADR-0018](adr/0018-slsqp-feasible-progress-and-guarded-components.md).

GG keeps feasible iterates feasible and accepts only improving feasible trials
after feasibility is reached. A batch without feasible candidates leaves the
current point unchanged and triggers step reduction through the Jacobian cache.
Infeasible fallback steps are limited to recovery from an initially infeasible
point. See [ADR-0005](adr/0005-gg-feasible-iterates.md).

GG uses lazy CSR Jacobian ring-buffer slots and sparse matrix-vector products
for constraint prediction, expanding only individual rows for projections.
SLSQP caches CSR Jacobians and converts only requested rows to the dense native
callbacks. Native SLSQP workspace remains dense. The existing core problem and
optimizer orchestration stays in its owning modules rather than undergoing an
unrelated size-only refactor; sparse assembly and finite-difference routines
exceed the 40-line guideline to keep selection, numerical state, and ordering
checks together. See [ADR-0014](adr/0014-sparse-jacobians.md).

### SciPy

`Optimizer_scipy` requires one objective, no integer variables, and at least one
floating variable. It translates bounds and constraints, dispatches callbacks,
and supplies iwopy Jacobians to methods that consume gradients. SciPy is a base
dependency, not an optional import.

SciPy minimization caches CSR Jacobians. Objective callbacks return dense
gradient vectors. `trust-constr` constraint callbacks retain CSR storage and
default native `sparse_jacobian` to true; explicit native settings remain
authoritative. Dense-only factorizations require `sparse_jacobian=False`.
Other native methods receive dense constraint Jacobians where needed. Sparse
constraint Jacobians do not remove dense Hessian approximations or guarantee
that a native factorization never densifies.

`Optimizer_scipy_de` separately supports bounded floating, integer, and mixed
population search. It uses deferred single-worker vectorization without polishing
and shares evaluated population rows across objective and constraint calls.
Success requires native convergence and final core feasibility. Callbacks report
survivor populations and actual iwopy candidate counts excluding finalization;
constraint values and infeasible objective energies are unavailable. Native
execution/representation options are managed by iwopy; evaluation exceptions
retain their original diagnostic cause.

### Nevergrad, Optuna, And DEAP

These adapters and SciPy DE use the private `_PopulationOptimizer` boundary,
not a new core contract. They require one objective, finite strictly ordered
bounds, valid bounded int32 variables, and finite in-bound initial values. Each
evaluated population retains separate integer/float arrays; fixed variables and
zero-tolerance exact equalities are rejected. Constraint tolerances relax bounds;
finite lower residuals precede finite upper residuals in component order.
The shared validation method intentionally exceeds the 40-line guideline to keep
shape, sentinel, bounds, and initial-value checks together in one ordered boundary.
`vectorize=True` evaluates an ordered batch as a population by default; false
uses individual calls without changing native search, counters, or callback batches.
See [ADR-0011](adr/0011-selectable-interface-vectorization.md).

Nevergrad exposes eight batching-safe selectors, evaluating outstanding asks
together before ordered tells. It receives signed losses and positive violations;
no executor is created. The `nevergrad>=1.0.12` extra alone constrains NumPy below
2.4 for NGOpt's metamodel scalar-conversion limitation, exercised by its catalog case.

Optuna >=5.0.0 supports TPE/random sampling with a fresh in-memory study and the
current `Trial.set_constraint` API. Ordered ask/tell batches use native objective
directions; failed batch evaluations fail outstanding trials and propagate.
TPE accounts for pending trials. Logging stays under caller control.

DEAP >=1.4.4 exposes three native variation/replacement schemes, bounded operators,
integer repair, and constraint-first fitness. Reproduced offspring are reevaluated
and counted. Its temporary Python-random seed is restored even on failure;
execution must be serial because upstream operators use that global generator.
No global creator classes are registered.

Nevergrad, Optuna, and DEAP choose the best feasible evaluated objective, otherwise
the smallest total positive violation with objective tie-breaking. Roundoff
residuals accepted by core feasibility do not bias feasible objective ranking.
The result is finalized through the problem; success means finite final
feasibility rather than convergence. Budgets include initialization and final
partial batches. Each completed batch emits an immutable original-direction
iteration snapshot with candidate counts excluding finalization. Repeated solves
reset counters, history, selection, and native state. See
[ADR-0007](adr/0007-vectorized-interfaces-and-optimizer-catalog.md).

### pymoo

The pymoo adapter supports pure integer, pure floating, and mixed variable
representations, individual or population execution, and single- or
multi-objective result conversion. Algorithm names resolved by the factory are
public selectors. Constraint conversion and representation changes require
focused adapter tests against supported pymoo versions.

### pycma

`Optimizer_cma` requires one objective, no integer variables, and at least one
floating variable. Each CMA-ES ask/tell generation makes one
`Problem.evaluate_population()` call for objectives and constraints by default,
or ordered individual calls when `vectorize=False`. The adapter
negates maximization objectives only at the backend boundary, translates box
bounds, and passes nonpositive constraint residuals to pycma's native augmented
Lagrangian. Finite lower bounds precede finite upper bounds, with component
registration order retained within each group. Constraint-owned tolerances
expand the bounds; equality constraints require positive tolerance.

Each completed generation emits an `iteration` callback with the evaluated
population, original objective signs and constraint values, and cumulative
backend evaluation count excluding finalization. A solve starts a fresh strategy
from the problem's initial values. The best feasible solution is finalized
through the problem; if none was feasible, the backend's favorite point is
returned with `success=False`. Success also requires convergence rather than
only a budget stop, and finite, feasible final values. `stop_conditions` records
the backend termination reasons.

Backend logging stays disabled regardless of iwopy verbosity, although pycma's
constraint logger may create an empty `outcmaes/` directory. Native options are
trusted configuration; iwopy manages variable representation, bounds, callbacks,
verbosity, and logging. See [ADR-0006](adr/0006-vectorized-pycma-interface.md).

### PyGMO

The PyGMO adapter is serial. Decision vectors place floating variables before
integer variables, while fitness vectors place objectives before constraints;
batch fitness is flattened according to PyGMO's contract. Current finalization
produces a single champion in `SingleObjOptResults`. Do not infer multi-objective
result support from UDP metadata alone.

The PyGMO UDP expands each registered constraint's tolerance into its `c_tol`
vector when the adapter is initialized; the core problem does not own that
backend representation.

pycma, pymoo, PyGMO, Nevergrad, Optuna, and DEAP imports are lazy and report installation guidance when
their extras are unavailable. Importing base `iwopy` must not require any of
these packages. pycma, Nevergrad, Optuna, and DEAP are loaded during optimizer initialization.

The optimizer catalog covers every named public selector, SciPy's default and
callable extension point, and PyGMO's NLopt subsolvers. Each row links to an explicit
factory-call script under `examples/optimizers/`, guarded by `__main__`; only
synthetic problem definitions are shared through `iwopy.benchmarks.quadratic`.
The package exports `QuadraticProblem`, `ParetoProblem`, and `MixedProblem` directly
from `iwopy.benchmarks`; constructors initialize silently, imports do not instantiate
problems, and no example-local helper or path injection is needed. See
[ADR-0010](adr/0010-public-quadratic-benchmarks.md). Compact capability labels distinguish
single/multi-objective results, bounds versus nonlinear restrictions, mixed
variables, and derivative needs. A recommendation gives a starting point rather
than a performance ranking; API links replace embedded class reference blocks.
Six non-empty tables partition objective count, nonlinear restrictions, and
local/global search. Boxes alone are not nonlinear constraints; constraint-capable
methods can also solve unconstrained problems. Single-run covariance/hill-climbing
methods are local adaptation here, not claimed global optimizers. Unconstrained
and constrained NSGA2/NSGA3 Pareto flows show actual iwopy results and explicit
weighted compromise selection; their demonstrated restriction determines their
table. There are 81 factory-call scripts covering 79 selectors plus two constrained
Pareto flows. Empty local multi-objective categories are omitted.
Tests execute every script and verify
inert imports, links, and selector coverage. Optional installations and native
PyGMO features guard their respective runs with explicit skips. PyGMO Pareto
examples evolve and return the native population instead of invoking unsupported
champion finalization.

Examples expose important controls one value per line, with inline explanations
only on value lines. Executable candidate/iteration limits keep the catalog quick;
comments give production starting suggestions rather than convergence guarantees.
Tests protect short explicit limits, comment placement, and production guidance.
GG/SLSQP use the source label and example prefix `iwopy`, not `native`.

The catalog's Vectorized Interfaces section covers every adapter with API links,
distinguishing population search, Jacobian batching, and optional batch fitness.
SciPy DE, pycma, Nevergrad, Optuna, and DEAP expose `vectorize=True` by default. False
routes each ordered batch through individual evaluation without changing native
batch generation, caches, budgets, callbacks, constraints, or result selection.
Empty batches invoke neither evaluation path. SciPy's native deferred/vectorized
DE mode remains enabled independently. See
[ADR-0011](adr/0011-selectable-interface-vectorization.md).

## Extension Points

### Problems And Functions

Implement the narrowest core base class. A reusable problem declares variable
names, initial values, bounds, application results, and individual/population
behavior. A function declares component count/names, variable dependencies,
values, and derivatives. Add both individual and population implementations
where the public API advertises vectorization.

### Optimizers And Interfaces

A new optimizer owns solver lifecycle, compatibility validation, callbacks, and
conversion to an iwopy result container. A backend interface remains isolated
under `iwopy.interfaces`; core must not import an optional backend eagerly.
Test backend ordering and failure conversion, not only a successful scalar
example.

### Wrappers

Wrappers adapt an existing public contract. Preserve the wrapped problem's
variable and component identity unless the wrapper explicitly transforms it.
`DiscretizeRegGrid` is the problem wrapper;
`RegularDiscretizationGrid` is its lower-level utility. Keep those concepts
distinct.

## Data And Integration Boundaries

iwopy does not own a persistent database. It receives caller-defined Python
objects, arrays, callables, paths, and backend settings; holds optimization state
in memory; and writes files only through explicit examples, callbacks, pipelines,
or caller code. The pycma constraint logger may create an empty `outcmaes/`
directory during a constrained solve, but the adapter disables backend log files.

- Validate variable maps, shapes, bounds, component selections, backend options,
	and paths at the owning boundary.
- Preserve lower-level exceptions as causes when adding context. Errors should
	identify the problem, function, variable, component, optimizer, backend, or
	stage involved.
- Keep optional backend imports isolated and fail with actionable installation
	guidance.
- Caller application data retains its own classification. Objective inputs,
	constraint data, variable values, and application-specific `problem_results`
	may require the classification question in
	[AGENTS.md](../AGENTS.md#data-classification). Synthetic benchmarks and public
	examples do not authorize inspection of another data set.
- No data-classification exception is currently recorded; see
	[data-classification-exceptions.md](data-classification-exceptions.md).

## Performance And Vectorization

Population evaluation is a primary architectural capability. Preserve leading
population axes and backend ordering instead of looping through candidates solely
for implementation convenience.

- Compare population values with stacked individual values.
- Avoid copying complete populations for each function component when mapped
	views or shared calculations are sufficient.
- Keep memory-cache behavior deterministic and independent of evaluation order.
- Keep analytical Jacobians aligned with selected component and variable order.
- Measure representative variable, population, component, and finite-difference
	batch sizes before accepting a performance optimization.

## Public Interfaces

Public contracts include exported Python classes and functions, lifecycle and
evaluation return ordering, variable and component names/order, array shapes,
bounds/tolerances, callback events/snapshots, result containers, pipeline stage
contracts, wrapper behavior, and backend algorithm selectors.

Compatibility-sensitive means identifying the complete impact and moving all
maintained callers, tests, examples, notebooks, and documentation to the target
contract in one change. It does not imply retaining a legacy path; development
is forward-only by default under
[ADR-0001](adr/0001-forward-only-development.md).

## Cross-Cutting Decisions

- Component-by-variable derivatives use the singular Jacobian terminology and
	CSR storage throughout core, wrappers, and caches. Native scalar gradients
	remain dense vectors; PyGMO's required `gradient` protocol names are retained
	for its sparse fitness-Jacobian entries. `trust-constr` receives sparse
	constraint callbacks, while SLSQP retains its required dense boundary. See
	[ADR-0014](adr/0014-sparse-jacobians.md).
- Optimizer creation uses a standalone `iwopy.create_optimizer` function owned by core,
	with explicit per-class keywords and flat native forwarding, unchanged public
	class names, and a dedicated PyGMO-backed `nlopt` family. All maintained
	examples and notebooks use it without compatibility entry points. See
	[ADR-0013](adr/0013-optimizer-creation.md).
- Optimizer catalog examples use annotated value lines and short explicit runs,
	with production starting suggestions in comments. GG/SLSQP are sourced as
	`iwopy`; external native behavior keeps its existing terminology. See
	[ADR-0012](adr/0012-optimizer-example-guidance.md).
- SciPy DE, Nevergrad, Optuna, and DEAP use true population search with explicit
	bounded single-objective contracts. The catalog links every public selector to
	a guarded factory-call script, tested with package-aware skips. New optional extras
	are lazy; Nevergrad alone constrains NumPy <2.4 based on observed upstream
	behavior. See [ADR-0007](adr/0007-vectorized-interfaces-and-optimizer-catalog.md).
- Shared quadratic catalog cases are public `iwopy.benchmarks` definitions with
	silently initialized constructors, not example-local helpers. See
	[ADR-0010](adr/0010-public-quadratic-benchmarks.md).
- The four population adapters default to `vectorize=True` and support ordered
	individual evaluation without changing native batches or search. See
	[ADR-0011](adr/0011-selectable-interface-vectorization.md).
- pycma CMA-ES uses genuine population evaluation and native augmented-Lagrangian
	constraint handling, with a lazy `cma>=4.4.0` extra and explicit continuous,
	single-objective support. See [ADR-0006](adr/0006-vectorized-pycma-interface.md).
- Development is strictly forward-looking by default. Superseded code and
	contracts are removed rather than kept behind compatibility shims; a bounded
	legacy exception requires explicit authorization and a recorded removal
	condition. See [ADR-0001](adr/0001-forward-only-development.md).
- Code changes require focused and full tests. Every change requires full
	pre-commit, affected public docstrings, the current-version final
	`CHANGELOG.md` section, and synchronized iwopy documentation as one completion
	gate. See [ADR-0001](adr/0001-forward-only-development.md).
- Regular and version-tag GitHub Actions run the complete test extra on Linux
	for Python 3.10 through 3.14. A backend-import preflight and explicit
	optimizer/catalog/benchmark regression stage precede the full suite.
	Missing solver packages cannot silently hide coverage; optional PyGMO native
	features retain documented skips. Manual regular-workflow runs do not publish.
- `pyproject.toml` is the source of package metadata, supported Python versions,
	dependencies, extras, build configuration, and mypy settings.
	Runtime and installed distribution versions, the final changelog section, and
	software `version` in `CITATION.cff` mirror the project version. Sphinx reads it
	directly. Citation schema and preferred-article metadata remain independent.
- Builds require wheel >=0.46.2 to exclude known path-traversal and
	regular-expression denial-of-service vulnerabilities. Direct dependency
	audit scope and advisory links are recorded in
	[development](development.md#security-floors).
- Production Python under `iwopy/` is type checked. Tests, examples, notebooks,
	and docs are not currently part of the mypy target.
- Pytest repository-file inputs are limited to `iwopy/`, `examples/`,
	`notebooks/`, and `tests/`; temporary test outputs remain allowed.
	Documentation, release metadata, and CI configuration are checked outside
	pytest. See [ADR-0019](adr/0019-test-file-dependency-boundary.md).
- Public Python APIs follow [docstring conventions](docstrings.md): NumPy-style
	docstrings expose optimization, array, lifecycle, and backend contracts while
	types remain in annotations.
- Ruff formatting/linting and mypy run through pre-commit. Pytest is the test
	runner; Sphinx with AutoAPI, Napoleon's NumPy-style parsing, and MyST-NB builds
	the documentation. The incompatible numpydoc extension is not loaded.
- iwopy follows Fraunhofer corporate design under
	[ADR-0002](adr/0002-corporate-design.md). Sphinx adapts authoritative tokens
	through `docs/source/_static/iwes-tokens.css` and `iwes.css`, reviewed outside
	pytest. Body, navigation, code, and tables use the 16px
	base scale; the 14px footer is secondary text. No browser application or shared
	matplotlib token adapter is introduced. The original color logo is restored
	on the white homepage without artwork changes; README uses the original white
	variant for dark color schemes. Both logos fill the available content line width
	with automatic proportional height, retained clear space, and no viewport overflow.
	Repeated installation/release prose is removed.
	Existing token `needs-review` flags remain authoritative. See
	[ADR-0008](adr/0008-readable-corporate-documentation.md), with the logo decision
	superseded by [ADR-0009](adr/0009-restored-logo-and-catalog-guidance.md).

## ADR Triggers

Create or supersede an ADR when a change alters lifecycle or module ownership,
variable/function/result shapes or ordering, bounds and feasibility semantics,
derivative assembly, callback events, pipeline contracts, backend translation,
supported runtime range, dependency strategy, data boundaries, UI design policy,
or a naming rule used across the package. A local implementation change that
preserves these contracts does not need an ADR.
