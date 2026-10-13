# iwopy Naming And Typing Conventions

## Purpose

This document defines stable iwopy vocabulary and naming rules for Python code,
optimization arrays, functions, results, callbacks, wrappers, pipelines,
backend interfaces, tests, documentation, and public contracts. Use the same
term at every layer unless an external optimizer requires a mapping. See
[architecture](architecture.md) for ownership and runtime contracts and
[development](development.md) for contributor commands.

Record a broadly consequential change to these rules in an
[ADR](adr/README.md). Do not rename an established public identifier solely to
match a newer style preference; Python exports, variable names, function
components, callback events, result attributes, and backend selectors are
compatibility-sensitive.

## General Rules

- Prefer a precise optimization term over a generic noun such as `item`,
	`manager`, `handler`, or `data_object`.
- Use singular names for one object and plural names for collections. Preserve
	established abbreviations such as `objs`, `cons`, and `n_pop` at their public
	boundaries.
- Keep the same noun in code, docstrings, tests, examples, notebooks, and user
	documentation.
- Name booleans as predicates or flags, counts with `n_`, scalar indices with
	`_i` or `_index`, and index collections with `_inds` where established.
- Include physical units in application docstrings, not in generic iwopy
	variable names, unless an external contract already includes the unit.
- Preserve backend names at interface boundaries and translate them once into
	iwopy ordering and result conventions.

## Domain Vocabulary

| Preferred term | Meaning | Avoid or distinguish from |
|---|---|---|
| iwopy | Fraunhofer IWES optimization tools in Python | A particular solver or scientific application |
| problem | A `Problem` defining variables, application behavior, objectives, and constraints | Optimizer or backend problem adapter |
| optimization variable | One named integer or floating degree of freedom | Function component or application result |
| individual | One complete variable assignment | One scalar variable or objective component |
| population | An ordered batch of individuals | Function components or optimization history |
| problem results / `problem_results` | Application-specific payload returned after variables are applied | Final optimizer result |
| optimization result / `opt_results` | `SingleObjOptResults` or `MultiObjOptResults` returned by an optimizer | Application-specific problem results |
| optimization function / `OptFunction` | Shared base for objective and constraint functions | Python callback or solver algorithm |
| objective | Function whose components are minimized or maximized | Constraint or complete problem |
| constraint | Function whose component bounds define feasibility | Objective or variable bound |
| function object | One registered objective or constraint instance | Scalar component |
| component | One scalar output of a function | Function object or optimization variable |
| feasible / valid | Satisfies all constraint component bounds within tolerance | Merely finite or successfully evaluated |
| optimizer | Object that solves an initialized problem | Problem, objective, or backend problem adapter |
| backend interface | Translation between iwopy and SciPy, pycma, pymoo, PyGMO, Nevergrad, Optuna, or DEAP | iwopy optimizer implementation |
| champion | Single solution selected by a backend such as PyGMO | Pareto population |
| Pareto front | Non-dominated objective values for a multi-objective result | All evaluated population values |
| wrapper | Adapter around a problem or function contract | Independent deep copy |
| memory | Optional cache of function values by variable assignment | Application persistence |
| callback | Observer of optimizer events | Objective, constraint, or pipeline stage |
| callback snapshot | Immutable normalized `OptimizerCallbackData` event payload | Mutable live backend state |
| pipeline | Ordered collection of optimization stages | One optimizer run |
| stage | One pipeline step returning `(success, results)` | Optimizer iteration or callback event |

## Python Symbols And Files

- Modules, functions, methods, parameters, and local variables use
	`snake_case`; classes use `PascalCase`; module constants use `UPPER_CASE`.
- Preserve established public spellings such as `OptFunction`,
	`SingleObjOptResults`, `MultiObjOptResults`, `Optimizer_scipy`, `LocalFD`, and
	`DiscretizeRegGrid`.
- Prefix non-public implementation details with one underscore. Do not export a
	private helper through a package `__init__.py`.
- Keep the root public API curated. Import lower-level optimizer and result
	contracts through `iwopy.core`; do not widen root exports incidentally.
- Package and source directories use lowercase names. Example directories use
	descriptive `snake_case`. Tests use `test_<subject>.py` and
	`test_<behavior>()` or an equally explicit observable behavior name.
- Put code in the module that owns the concept described in
	[architecture](architecture.md#module-boundaries). Do not add a generic
	`helpers.py` when an existing focused module owns the behavior.
- Use `from __future__ import annotations` consistently with the surrounding
	module. Production annotations must remain valid for Python 3.10.

## Common Type Annotations

Infer types from the owning base class and call site rather than from a name
alone, but use these established meanings unless the local contract says
otherwise.

| Name | Usual type | Notes |
|---|---|---|
| `problem` | `iwopy.core.Problem` | Application problem, not a backend adapter |
| `optimizer` | `iwopy.core.Optimizer` | Initialized against a problem before solving |
| `objective` | `iwopy.core.Objective` | May expose multiple scalar components |
| `constraint` | `iwopy.core.Constraint` | Bounds and tolerance are component-wise |
| `func` | `iwopy.core.OptFunction` | Use a narrower objective/constraint type where required |
| `vars_int` | `numpy.ndarray` | Integer variables for one individual or population |
| `vars_float` | `numpy.ndarray` | Floating variables for one individual or population |
| `objs` | `numpy.ndarray` | Objective components for one individual or population |
| `cons` | `numpy.ndarray` | Constraint components for one individual or population |
| `problem_results` | `object | None` | Application-defined payload from applying variables |
| `prob_res` | `object | None` | Established concise local for `problem_results`; not final optimizer results |
| `opt_results` | `SingleObjOptResults` or `MultiObjOptResults` | Backend solve result |
| `callbacks` | `list[OptimizerCallback]` | Ordered observers supplied to solve |
| `callback_data` | `OptimizerCallbackData` | Immutable normalized event snapshot |
| `ax` | `matplotlib.axes.Axes` | Preserve a caller-supplied axes |
| `fig` | `matplotlib.figure.Figure` | Add `None` only when allowed by the contract |

Use `ArrayLike`, `Sequence`, or `Mapping` only when the implementation accepts
that abstraction. Before using `Any` or `object`, inspect the core base class and
call sites. Application-specific `problem_results` legitimately remain
`object | None` at generic boundaries; this does not justify broadening arrays,
functions, optimizers, or callbacks with concrete contracts.

## Optimization Arrays And Ordering

Names and shapes are observable contracts:

| Name | Individual shape | Population shape |
|---|---|---|
| `vars_int` | `(n_vars_int,)` | `(n_pop, n_vars_int)` |
| `vars_float` | `(n_vars_float,)` | `(n_pop, n_vars_float)` |
| `objs` | `(n_objectives,)` | `(n_pop, n_objectives)` |
| `cons` | `(n_constraints,)` | `(n_pop, n_constraints)` |

Use zero-width arrays for an absent integer or floating variable family. Keep the
population axis even when the trailing variable count is zero.

`n_objectives` and `n_constraints` count scalar components. `n_functions`
counts function objects. Function lists concatenate components in registration
order. Combined gradient/function ordering places objectives before constraints.

Use `var_names_int` and `var_names_float` for ordered problem variable names.
Existing methods such as `var_names_int()` remain callable APIs; do not convert
them to attributes without a deliberate contract change.

## Bounds, Feasibility, And Derivatives

- Integer variable infinity uses `Problem.INT_INF`; floating variable infinity
	uses `np.inf`.
- Constraint components have lower bounds, upper bounds, and tolerance. The
	default is `-np.inf <= value <= 0.0` with tolerance `1e-5`.
- Each constraint object owns its scalar tolerance. A backend may derive a
	component vector in constraint registration order, but `Problem` does not own
	or cache that vector.
- `success` on result containers and callback snapshots denotes the owning
	optimizer's solution/candidate status. State whether it represents solver
	completion, feasibility, or both for a backend-specific value.
- `vardeps_float()` returns a component-by-variable dependency mask.
- `ana_deriv(..., var, components)` returns one derivative per selected
	component for one floating variable. Use `NaN` for unavailable analytical
	derivatives and zero only for known independence.
- A complete Jacobian from `get_gradients()` has component rows and selected
	floating-variable columns. Never rely on the word `gradient` to communicate
	orientation.

## Results And Evaluation Names

- `evaluate_individual()` and `evaluate_population()` return `(objs, cons)` or
	`(objs, cons, problem_results)`.
- Finalization returns `(problem_results, objs, cons)`. Preserve this different
	order and document it explicitly.
- `SingleObjOptResults` stores one solution and may use `None` when a backend
	did not produce variables or values.
- `MultiObjOptResults` stores Pareto-population arrays with leading `n_pop` and
	a per-candidate success array.
- Use `problem_results` only for application payloads and `opt_results` for final
	iwopy containers. Do not call SciPy's native `OptimizeResult` an iwopy result
	without identifying the adapter boundary.

## Optimizers, Interfaces, And Factories

- iwopy optimizer classes belong in `iwopy.optimizers`; external translations
	belong in `iwopy.interfaces.<backend>`.
- Preserve external backend algorithm spellings at selection boundaries.
- PyGMO decision vectors use floating variables before integers and fitness
	vectors use objectives before constraints.
- pymoo representations may be integer, floating, or mixed. Do not describe the
	adapter as floating-only or non-mixed.
- Factory `new()` methods resolve established class or algorithm names. Check
	examples, tests, and backend availability messages before renaming a selector.
- Optional backend imports remain lazy and name the corresponding package extra
	in errors and documentation.
- The external project is pycma; its distribution, import, iwopy interface
	package, and optional extra are named `cma`. The public adapter is
	`Optimizer_cma`, its backend options are `cma_pars`, and its initial search
	standard deviation is `sigma0`. `stop_conditions` records backend termination
	reasons. pycma inequality residuals group finite lower bounds before finite
	upper bounds, preserving iwopy component order within each group.
- The vectorized SciPy differential-evolution adapter is `Optimizer_scipy_de`,
	with native options in `scipy_pars` and its native result in `scipy_results`.
	Distinguish search-population vectorization from `Optimizer_scipy` gradient batching.
- The distribution, import, interface package, and extra names are `nevergrad`,
	`optuna`, and `deap`; public classes are `Optimizer_nevergrad`, `Optimizer_optuna`,
	and `Optimizer_deap`. Preserve native `algorithm`/`sampler` selector spellings.
- `budget` counts candidate evaluations, not function calls or generations;
	`batch_size` is the outstanding-ask or offspring count. `n_evaluations` excludes
	selected-point finalization. Native settings are `optimizer_pars` for Nevergrad
	and `sampler_pars` for Optuna; native state is `optimizer` or `study`, respectively.
- Optuna distribution identifiers are `float_<index>` followed by `int_<index>`
	in family order. Trial constraint identifiers are `constraint_<index>` in the
	finite-lower-then-finite-upper residual ordering shared by the new adapters.
- Catalog scripts use `<backend>_<optimizer>.py`, lowercase snake_case, with
	fully explicit constructors in a `__main__` guard. `iwopy.benchmarks.quadratic`
	owns the synthetic `QuadraticProblem`, `ParetoProblem`, and `MixedProblem` classes,
	curated through `iwopy.benchmarks`; do not use example-local helpers or path injection.
- SciPy DE, Nevergrad, Optuna, and DEAP use the optimizer-level `vectorize` flag,
	defaulting to true, for population versus ordered individual evaluation of native
	batches. Distinguish this from SciPy minimization's `vectorized` gradient option,
	pymoo's problem-level `vectorize`, and PyGMO's problem-level `pop` batch-fitness flag.
- Documentation styles are `iwes-tokens.css` and `iwes.css`. The adapter maps
	authoritative token paths to `--iwes-` prefixed kebab-case CSS variables;
	`tests/test_documentation.py` verifies every resolved value and reference.
- Catalog capabilities name objective/result workflows: `Single`, `Multi`, and
	`Multi (native)`. `bounds` means variable boxes, `constraints` means additional
	restrictions, and `mixed` means integer plus floating variables. Recommendations
	are starting points, not optimizer performance rankings. Keep detailed class
	references in the API rather than embedding them in the catalog.
- Catalog subsections use objective count, nonlinear-constraint support, and
	`Local`/`Global` search strategy. Omit empty categories. Explicit Pareto workflows
	use `_constrained` script suffixes and `selected_index` for a weighted front choice;
	keep native PyGMO populations distinct from iwopy Pareto result workflows.
- Name GG/SLSQP's source `iwopy`, with `iwopy_gg.py` and `iwopy_slsqp.py` example
	filenames. Reserve `native` for an external backend's own behavior or results.
	Catalog setup parameters use one value per line, explaining values rather than
	commenting `problem` or parameter dictionaries. Short demo limits and populations
	include production starting suggestions. See
	[ADR-0012](adr/0012-optimizer-example-guidance.md).

## Wrappers, Pipelines, And Callbacks

- `ProblemWrapper` wraps and rebinds an underlying problem; names should expose
	whether state is shared.
- `DiscretizeRegGrid` is a problem wrapper.
	`RegularDiscretizationGrid` is the lower-level grid utility.
- Pipeline classes end in `Pipeline`; composable stages derive from
	`PipelineStage`. Use `prev_stage` and `prev_results` for stage inputs and
	return `(success, results)`. Use `initial_results` for an application payload
	seeded into the first selected stage of a restart.
- Stage directories use two-digit index prefixes: `NN_stage_name`.
- Callback events are `iteration` or `evaluation`. Distinguish event counters
	from population indices and objective evaluations.
- `OptimizationHistory` is a callback/history recorder, not an optimizer result
	container.

## Documentation Names

Use [docstring conventions](docstrings.md) for required coverage, NumPy-style
structure, optimization contracts, examples, and review. This document remains
authoritative for names used inside those docstrings.

- Section entries use exact signature parameters or semantic return names.
- State array shapes with the count names above and identify ordering across
	variables, components, and populations.
- Distinguish problem results, iwopy optimization results, and native backend
	results.
- Link to the owning public API rather than inventing a prose-only synonym that
	users cannot search for.

## Test Names

Test directories retain their package ownership:

- `tests/core/`: lifecycle, mappings, derivatives, memory, callbacks, and grids;
- `tests/optimizers/`: iwopy optimizer behavior;
- `tests/scipy/`, `tests/cma/`, `tests/pymoo/`, and `tests/pygmo/`: backend translation;
- `tests/interfaces/`: shared new population-adapter contracts and complete,
	package-guarded catalog execution, import safety, and selector/link coverage;
- root tests: package imports and executable callback examples.

Name focused tests after observable behavior. Parametrization identifiers expose
the optimizer, backend, variable representation, component, or population mode
being tested.

## ADR Naming

- File format: `NNNN-short-kebab-case-title.md`
- Start at `0001` and increment monotonically.
- Keep an ADR title stable after merge unless a later ADR supersedes the
	decision.
