# iwopy Docstring Conventions

## Purpose

This document defines the documentation contract for public Python APIs in
iwopy. It applies to modules, classes, constructors, methods, functions,
properties, callbacks, wrappers, pipeline stages, result containers, and backend
interfaces.

Use NumPy-style docstrings compatible with numpydoc and Sphinx AutoAPI. Keep
Python types in annotations and describe semantic contracts in prose: lifecycle,
optimization meaning, shapes, ordering, units, bounds, tolerances, mutation,
side effects, derivatives, backend translation, return structure, and failures.

See [naming conventions](naming-conventions.md) for iwopy vocabulary and array
names, [architecture](architecture.md) for ownership and runtime contracts, and
[development](development.md) for required review and validation.

## Required Coverage

Document every public:

- module whose purpose or boundary is not obvious from its exported objects;
- class and abstract base class;
- constructor with public parameters;
- function and method;
- property and descriptor;
- callback hook, pipeline stage contract, factory, and backend adapter entry
	point; and
- public attribute whose meaning cannot be recovered from the constructor and
	type alone.

An inherited docstring is sufficient only when the inherited contract remains
exactly true. Override it when a subclass narrows supported variables, changes
component ordering, adds side effects, changes lifecycle requirements,
translates to a backend, or returns a more specific result.

Private helpers do not need docstrings when their name, annotations, and local
context are sufficient. Add a short docstring where a private helper owns a
non-obvious numerical, ordering, caching, backend, or state transition contract.

## General Rules

- Start with one imperative or declarative summary line ending in a period.
- Describe what the API guarantees, not a line-by-line implementation.
- Use exact iwopy terms: problem, optimization variable, function object,
	component, individual, population, problem results, optimization result,
	callback snapshot, pipeline stage, and backend interface.
- Keep types in annotations. Do not repeat them in `Parameters`, `Returns`,
	`Yields`, or `Attributes` entries.
- Use parameter names exactly as written in the signature, including `*args`
	and `**kwargs` without the asterisks in section entries.
- Document default behavior when it changes semantics. Do not repeat a literal
	default that is already clear from the signature unless its meaning needs
	explanation.
- State array shape and axis order with names from
	[naming conventions](naming-conventions.md#optimization-arrays-and-ordering).
- State component order, variable mapping, tuple order, and backend order where
	a caller must interpret positional values.
- State whether an operation requires initialization, mutates problem state,
	uses memory, calls external solver code, creates files, prints progress, or
	finalizes another object.
- State physical units only when the API establishes them. Generic iwopy arrays
	do not impose application units.
- Document real exceptions under `Raises`; do not list impossible or purely
	internal failures.
- Keep examples deterministic, minimal, and runnable with declared extras.
- Do not promise broader optimizer, backend, vectorization, or derivative support
	than the implementation and tests establish.

## Section Order

Use only sections that add information, in this order:

1. Summary
2. Extended Summary
3. Parameters
4. Returns or Yields
5. Raises
6. Warns
7. Attributes
8. Notes
9. See Also
10. References
11. Examples

`__init__` has no `Returns` section. A method returning `None` normally omits
`Returns` unless the absence of a value is itself important to the protocol.

## Parameters

List every public parameter that needs semantic explanation. Keep descriptions
specific enough to reject the wrong value before reading implementation code.

```python
def evaluate_population(
    self,
    vars_int: np.ndarray,
    vars_float: np.ndarray,
    ret_prob_res: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, object | None]:
    """
    Evaluate objective and constraint components for a population.

    Parameters
    ----------
    vars_int
        Integer optimization variables with shape ``(n_pop, n_vars_int)``.
        Use a zero-width second axis when the problem has no integer variables.
    vars_float
        Floating optimization variables with shape
        ``(n_pop, n_vars_float)``. The population order must match
        ``vars_int``.
    ret_prob_res
        Whether to include application-specific problem results. Requests for
        problem results use the application path rather than cached objective
        and constraint values alone.
    """
```

For mappings and selectors, describe accepted forms and resolution rules:

- names, integer indices, or patterns;
- whether order is preserved;
- whether duplicate matches are allowed;
- ambiguity and missing-match behavior; and
- whether the selection is relative to problem variables, function variables,
	or scalar components.

For backend settings, identify the backend option name, supported algorithms,
or pass-through boundary. Do not restate an external package's entire API.

For path parameters, state whether the path must exist, is created, is a file or
directory, and whether existing content is overwritten.

## Returns And Yields

Name each return semantically. Describe shape, ordering, optionality, and
ownership. A tuple is not self-documenting.

```python
Returns
-------
objs
    Objective components with shape ``(n_pop, n_objectives)``. Components are
    concatenated in objective registration order.
cons
    Constraint components with shape ``(n_pop, n_constraints)``. Components
    are concatenated in constraint registration order.
problem_results
    Application-specific results produced while applying the variables, or
    ``None`` when the problem has no result payload. Returned only when
    ``ret_prob_res`` is true.
```

Preserve iwopy's distinct tuple contracts in prose:

- evaluation returns `(objs, cons)` or `(objs, cons, problem_results)`;
- finalization returns `(problem_results, objs, cons)`; and
- pipeline stages return `(success, results)`.

For generators, use `Yields`. State iteration order, whether yielded values are
views or copies, and what ends iteration when those details affect callers.

For plots, return the actual matplotlib object:

```python
Returns
-------
ax
    Axes containing the Pareto-front artists. This is ``ax`` when the caller
    supplied one; otherwise it belongs to the figure created by this method.
```

## Raises And Diagnostics

Document failures a caller can cause or recover from. Explain the condition,
not the implementation line that raises it.

```python
Raises
------
ValueError
    If a selected component does not exist or if the derivative result has the
    wrong number of selected components.
RuntimeError
    If an analytical derivative remains unavailable after dependency handling.
```

Include lifecycle failures, invalid shape/order, unresolved variable maps,
incompatible optimizer/problem combinations, unsupported backend
representations, unavailable optional packages, invalid bounds/tolerances, and
I/O failures when they are part of the public boundary.

An implementation assertion does not excuse vague documentation. Describe the
required precondition in the summary or relevant parameter and list the public
failure mechanism that the API currently exposes.

When adding context to a lower-level exception, preserve it as the cause and
name the problem, function, variable, component, optimizer, backend, callback,
or pipeline stage involved.

## Attributes

Use `Attributes` on a class when callers read public state that is not already
obvious from constructor parameters or properties. Include shape and semantic
meaning, but do not duplicate property documentation or every internal field.

```python
Attributes
----------
success
    Per-candidate success or feasibility flags with shape ``(n_pop,)``.
objs
    Pareto-front objective values with shape
    ``(n_pop, n_objectives)``.
problem_results
    Application-specific result payload associated with the population.
```

For callback snapshots, state immutability, normalized dtypes, and optional
values. For result containers, distinguish single-solution optional arrays from
multi-objective population arrays.

## Notes, See Also, And References

Use `Notes` for numerical or protocol details that do not belong to one
parameter or return value:

- objective and constraint component ordering;
- finite-difference formulas and step selection;
- memory/cache semantics;
- backend variable or fitness translation;
- convergence or scaling assumptions;
- wrapper state sharing; and
- callback or pipeline lifecycle.

Use `See Also` for directly related iwopy APIs, such as a problem wrapper and its
underlying grid utility, or an analytical derivative method and `LocalFD`. Use
`References` for an algorithm, paper, or external specification on which the
implementation materially depends.

Put mathematics in clear notation and define symbols. For example, a centered
finite difference can be written as

$$
\frac{\partial f}{\partial x_j}
\approx
\frac{f(x + h e_j) - f(x - h e_j)}{2h},
$$

where $h$ is the configured step and $e_j$ selects the floating variable. State
how bounds alter the stencil if the implementation switches to a one-sided
difference.

## Examples

Examples are executable contracts. Keep them small enough to run in a doc build
or copied session.

- Use public imports and deterministic values.
- Include all imports and setup required to understand the example.
- Keep arrays small and make expected shape/order visible.
- Seed stochastic backends.
- Avoid network access and local private data.
- Identify an optional package extra when the example needs pymoo or PyGMO.
- Avoid brittle exact optimizer output unless deterministic convergence is the
	behavior under test.
- Use temporary directories for files and close figures created only for the
	example.

```python
Examples
--------
>>> import numpy as np
>>> vars_int = np.empty((2, 0), dtype=np.int32)
>>> vars_float = np.array([[0.0, 1.0], [2.0, 3.0]])
>>> vars_float.shape
(2, 2)
```

Do not use `...` to hide setup that determines lifecycle, variable mappings,
shape, ordering, or backend selection.

## Lifecycle Documentation

For lifecycle-aware objects, document the relevant state transition:

- what can be configured before `initialize()`;
- what metadata initialization fixes;
- whether evaluation or solving requires initialization;
- whether the method initializes lazily;
- what `finalize()` applies or releases;
- whether repeated initialization/finalization is valid; and
- whether the operation initializes or finalizes owned/wrapped objects.

Example class summary:

```python
class Optimizer(Base):
    """
    Solve one initialized iwopy problem.

    Initialize the optimizer after registering and initializing the problem's
    objectives and constraints. Subclasses validate their supported variable,
    objective, constraint, and derivative contracts before invoking a backend.
    """
```

Do not imply that a constructor performs initialization unless it does. Do not
describe an initialized function list as mutable when later additions are
rejected.

## Problems And Variables

A public `Problem` docstring states:

- integer and floating variable families;
- ordered variable names;
- initial values and bounds;
- individual and population support;
- how variables are applied;
- whether application returns `problem_results`;
- memory/caching behavior when enabled;
- lifecycle requirements; and
- finalization behavior and tuple order.

Document individual shapes as `(n_vars_int,)` and `(n_vars_float,)`; population
shapes as `(n_pop, n_vars_int)` and `(n_pop, n_vars_float)`. State that absent
families retain zero-width arrays. Do not call a one-dimensional variable array
a population.

For bounds, distinguish integer infinity sentinels from floating `np.inf`. If a
problem narrows a backend's supported bound types or requires finite values,
state it at the owning API.

For variable mappings, identify the source and destination order. A function
receives its mapped problem variables in the function's declared variable order,
not arbitrary problem order.

## Optimization Functions And Components

An `OptFunction` subclass docstring states:

- objective or constraint role;
- number and names of scalar components;
- required integer and floating variables and their mapping order;
- individual and population input/output shapes;
- whether population evaluation is vectorized or inherited;
- units or scaling when established;
- dependencies and derivative support; and
- errors for invalid component selections or shapes.

`n_objectives` and `n_constraints` count components, not function objects.
Document both when confusion is possible.

For an objective, describe component-wise minimization/maximization semantics.
For a constraint, describe lower bounds, upper bounds, tolerance, and feasibility.
The generic default is `-np.inf <= value <= 0.0` with tolerance `1e-5`, but a
subclass docstring must state its actual contract rather than relying on the
default.

Simple callable objectives and constraints document argument order: integer
function variables precede floating function variables. State the expected
return for one or multiple components.

## Derivatives And Gradients

For derivative APIs, document:

- differentiation variable and selected component order;
- derivative/Jacobian orientation;
- individual variable point and required shapes;
- exact, analytical, finite-difference, or hybrid behavior;
- dependency-mask treatment;
- `NaN` as unavailable analytical derivative, distinct from exact zero;
- step size, relative/absolute interpretation, and bound handling;
- population batching used by finite differences; and
- errors for unresolved or incorrectly shaped derivatives.

```python
Returns
-------
gradients
    Derivatives with shape
    ``(n_selected_components, n_selected_float_variables)``. Rows follow the
    requested component order and columns follow the requested floating-variable
    order.
```

If a combined function list is accepted, state that objective components precede
constraint components. If integer derivatives are unsupported, say so rather
than omitting the limitation.

## Memory And Caching

Memory-related docstrings state:

- the values forming a cache key;
- whether arrays are copied or referenced;
- which objective/constraint results are stored;
- whether populations are stored by individual;
- cache lookup/update side effects; and
- why requests for application `problem_results` bypass or supplement cached
	values.

Do not use persistence terminology for in-memory caching. iwopy `Memory` is not
a database or checkpoint format.

## Optimizers And Backends

An optimizer docstring states:

- supported integer/floating variables;
- supported objective count and constraint forms;
- initialization and compatibility checks;
- derivative requirements or numerical fallback;
- callback behavior;
- backend option pass-through;
- verbosity/progress behavior;
- result-container type and failure representation; and
- optional dependency extra where applicable.

For backend adapters, document translation at the boundary:

- SciPy accepts continuous single-objective problems and consumes iwopy bounds,
	constraints, and Jacobians according to the selected method.
- pymoo can use integer, floating, or mixed representations and individual or
	population execution.
- PyGMO decision vectors place floating variables before integer variables and
	fitness vectors place objectives before constraints.

Do not describe a backend capability merely because an external library supports
it. Document only the representations and result conversions implemented by the
iwopy adapter.

## Result Containers

For `SingleObjOptResults`, document scalar success, individual variable arrays,
objective/constraint arrays, retained names, and `None` for unavailable backend
results.

For `MultiObjOptResults`, document leading population axes, the per-candidate
success array, Pareto semantics, retained names, and application-specific
problem results. State whether plotting marks feasible/valid and infeasible/
invalid points and which objectives are selected for each axis.

Do not call these containers SciPy `OptimizeResult` objects. When an interface
uses a native result internally, identify the conversion into iwopy's public
container.

## Callbacks And History

Callback docstrings state:

- `initialize`, ordered `notify`, and `finalize` lifecycle;
- emitted event names (`iteration` or `evaluation`);
- whether counters, values, and success flags may be absent;
- normalized two-dimensional population shapes;
- integer `int32` and numeric `float64` snapshot dtypes;
- immutability of `OptimizerCallbackData`;
- callback ordering and error propagation; and
- any files, figures, or accumulated history produced.

`OptimizationHistory` is callback/history state, not an optimization result.
Plot methods document the selected objective, aggregation across populations,
caller-supplied axes, labels, and returned matplotlib object.

## Wrappers

A wrapper docstring identifies:

- the wrapped object type;
- whether state and function lists are shared or copied;
- which names, variables, components, bounds, results, derivatives, or lifecycle
	steps are transformed;
- what remains delegated unchanged;
- initialization/finalization ownership; and
- errors introduced by the adaptation.

For `ProblemWrapper`, make shared function-list behavior explicit. For
`DiscretizeRegGrid`, distinguish the problem wrapper from the lower-level
`RegularDiscretizationGrid` utility. For `LocalFD`, document numerical stencil,
steps, bounds, selected functions, and batching.

## Pipelines

Pipeline docstrings state:

- stage registration and unique-name requirements;
- required initialization;
- base/stage directory creation;
- selected half-open range `[start_stage, end_stage)`;
- `prev_stage` and `prev_results` inputs;
- `(success, results)` return semantics;
- stopping behavior on failure or exception;
- finalization behavior; and
- forwarded keyword arguments.

Stage result payloads are application-defined `object | None`. Do not name them
`opt_results` unless they are specifically iwopy optimization result containers.

## Plotting And File-Producing APIs

For plotting functions:

- accept and preserve caller-provided axes where supported;
- state when a figure is created;
- document selected objectives, cases, validity masks, scales, labels, and units;
- return the axes or figure that callers need for composition;
- distinguish data through labels, markers, line styles, or hatch patterns in
	addition to color; and
- avoid claiming corporate-design conformance unless the implementation follows
	the applicable repository policy.

For file-producing APIs, state output path, format, overwrite behavior, directory
creation, content ordering, and return value. Do not hide side effects in an
extended summary.

## Classes, Properties, And Factories

Class docstrings describe role, lifecycle, and durable behavior. Constructor
parameters belong on `__init__` in the established iwopy style; avoid duplicating
the full list on the class.

Property summaries describe the value, not the getter implementation:

```python
@property
def n_stages(self) -> int:
    """Number of registered pipeline stages."""
```

Factory methods document accepted selector names, discovery behavior, optional
dependency failures, and the exact returned base type. Do not promise that every
subclass can be constructed through a factory unless discovery and tests prove
it.

## Abstract And Overridden Methods

Abstract methods define the complete subclass contract: arguments, shape/order,
returns, lifecycle preconditions, side effects, and allowed failures. A concrete
override may inherit the docstring only when it preserves all of those details.

When a backend or wrapper narrows the abstract contract, document the narrowing
on the override. Do not copy stale prose from the base and change only the
summary.

## Documentation Review Checklist

Before completing a code change, review every affected public class and function:

- Does the summary identify the optimization role and observable behavior?
- Do parameter names match the final signature?
- Are individual/population shapes and axis order explicit?
- Are variable and component selection/order explicit?
- Are objective direction, constraint bounds/tolerance, and feasibility clear?
- Are evaluation, finalization, and pipeline tuple orders correct?
- Are lifecycle preconditions and mutations documented?
- Are analytical/numerical derivative semantics and unavailable values clear?
- Are problem results distinguished from iwopy and native backend results?
- Are callback events, optional values, and normalized snapshot shapes clear?
- Are wrapper state sharing and backend translations explicit?
- Are file/plot side effects and returned objects documented?
- Are raised exceptions accurate and contextual?
- Are examples deterministic and runnable with declared extras?
- Do related API docs, architecture, naming, examples, notebooks, and the final
	current-version changelog section remain accurate?
