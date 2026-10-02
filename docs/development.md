# iwopy Development Guide

## Purpose

This guide collects repeatable repository workflows for human contributors and
coding agents. [AGENTS.md](../AGENTS.md) remains the policy source;
[architecture](architecture.md) explains runtime ownership and contracts; and
[naming conventions](naming-conventions.md) defines iwopy vocabulary, types,
arrays, results, and identifiers. [Docstring conventions](docstrings.md) define
the public Python documentation contract.

Prefer a focused check that can disprove the current change before running a
broad suite. When code files change, finish with the full test suite. Every
development change finishes with repository-wide pre-commit, current docstrings
and iwopy documentation, and an entry in the final current-version changelog
section.

## Forward-Only Development

iwopy development is strictly forward-looking by default under
[ADR-0001](adr/0001-forward-only-development.md). Implement the target contract
directly, update all maintained callers, and remove superseded code, tests,
examples, aliases, formats, fallback behavior, and documentation in the same
change.

Do not add a compatibility shim, deprecation branch, dual input format, or other
legacy path for convenience. A legacy exception requires explicit user
authorization, a bounded scope, and an ADR with an objective removal condition.
Identifying downstream impact remains mandatory; preserving the old contract
does not.

This policy governs changes from ADR-0001 onward. It does not require unrelated
historical compatibility code to be removed during an otherwise focused task.

## Environment And Dependencies

iwopy supports Python 3.10 through 3.14. Keep production syntax compatible with
3.10 even when developing on a newer interpreter.

Use `uv` for local Python workflows:

```console
uv sync --extra dev --extra test --upgrade
uv run pytest tests
uv run pre-commit run --all-files
```

The dependency sets in `pyproject.toml` are optional extras, not uv dependency
groups. Install them with `--extra`; `--dev` does not select the optional `dev`
extra. Multiple extras are combined by repeating `--extra`.

| Dependency set | Use |
|---|---|
| Main dependencies | NumPy, SciPy, and matplotlib runtime support |
| `scipy` | Explicit SciPy integration selection; SciPy is also a base dependency |
| `pymoo` | pymoo optimizer interface |
| `pygmo` | PyGMO optimizer interface |
| `opt` | Aggregate SciPy, pymoo, and PyGMO integrations |
| `test` | All solver integrations, pytest, notebook execution, pre-commit, and mypy |
| `dev` | Contributor hooks, mypy, object-size inspection, and Jupyter |
| `doc` | Sphinx, AutoAPI, numpydoc, Immaterial, MyST-NB, and documentation notebooks |

Put a new dependency in the narrowest justified set. A package imported by base
`iwopy` belongs in main dependencies. An optional backend must remain lazy and
belong in its own extra. `uv.lock` is currently ignored and is not a repository
source of dependency versions.

Install the repository hooks once per checkout:

```console
uv run pre-commit install
```

## Finding The Owning Code

Start from the smallest behavior owner, then inspect one abstraction boundary
and the nearest tests.

| Change | Start in | Then inspect |
|---|---|---|
| Base lifecycle or initialization | `iwopy/core/base.py` | Derived core types and lifecycle tests |
| Variable mapping or evaluation | `iwopy/core/problem.py` | Function lists/subsets, memory, interfaces, and core tests |
| Objective or constraint contract | `iwopy/core/` | Problem registration, component ordering, gradients, and tests |
| Result container or Pareto plotting | `iwopy/core/opt_results.py` | Backend finalization, examples, and output tests |
| Callback or history behavior | `iwopy/core/optimizer_callback.py` | Optimizer dispatch and callback/example tests |
| Pipeline behavior | `iwopy/core/pipeline.py` | Stage implementations and pipeline users |
| Simple callable API | `iwopy/wrappers/simple_*.py` | Core function/problem contract and examples |
| Finite differences | `iwopy/wrappers/local_fd.py` | Gradient assembly and analytical/numerical tests |
| Problem wrapper or discretization | `iwopy/wrappers/` | Underlying core problem, grid utility, and core tests |
| SciPy translation | `iwopy/interfaces/scipy/` | Core gradients/results and `tests/scipy/` |
| pymoo translation | `iwopy/interfaces/pymoo/` | Factory, representation, results, and `tests/pymoo/` |
| PyGMO translation | `iwopy/interfaces/pygmo/` | UDP ordering, callbacks/results, and `tests/pygmo/` |
| Native optimizer | `iwopy/optimizers/` | Core optimizer contract and optimizer tests |
| Public import | Owning package `__init__.py` | Root exports, AutoAPI output, examples, and import tests |

Useful searches include the class or method name, variable/component name,
shape count, backend selector, callback event, and result attribute. Search
examples and notebooks before changing constructors or factory names; they are
public call sites.

Do not treat `build/lib/iwopy/` as source. The maintained package is `iwopy/`.

## Focused Validation

Run the narrowest applicable command first:

```console
# One behavior
uv run pytest tests/path/test_module.py::test_behavior -q

# One test module or package area
uv run pytest tests/core/test_gradients_ana.py -q

# Production type checking, as configured in pyproject.toml
uv run mypy iwopy

# Repository hooks on touched files
uv run pre-commit run --files iwopy/path.py tests/path/test_module.py
```

Before closure, broaden to the repository gates:

```console
# Full Python test suite, required when code files changed
uv run pytest tests

# Notebook execution used by CI
uv run pytest --nbmake notebooks

# All formatting, linting, type, and file-hygiene hooks
uv run pre-commit run --all-files
```

Pre-commit owns the configured Ruff and mypy invocations. A hook can modify a
file and still exit non-zero; inspect the diff and rerun it.

## Development Closure

A change is not complete until all closure evidence describes the final working
tree:

1. When code files changed, add or update tests for changed logic. Run the most
	discriminating focused tests first, then run `uv run pytest tests`
	successfully. Documentation-only changes run their applicable link, render,
	version, or structure checks instead.
2. Run `uv run pre-commit run --all-files` after the last edit. If a hook changes
	a file, inspect it and rerun the entire command.
3. Review every affected public Python API against
	[docstring conventions](docstrings.md). Update variables, components, shapes,
	bounds, lifecycle, side effects, returns, errors, derivatives, backend
	behavior, and examples to match the final implementation.
4. Update `docs/architecture.md`, `docs/naming-conventions.md`, this guide,
	`docs/source/`, examples, notebooks, ADRs, and any other iwopy information
	affected by the change. Do not knowingly leave stale prose, diagrams,
	commands, or contracts.
5. Read `project.version` from `pyproject.toml`. Confirm the final version
	section in `CHANGELOG.md` is exactly `## v<version>` and add a
	style-consistent entry for the change to that section.

Run notebook tests when notebooks change. A Sphinx build is not part of default
closure; use it as an explicit targeted check when requested or when rendered
Sphinx/AutoAPI output itself is under investigation. A documentation-only
change does not run the full runtime test suite, but it still runs applicable
documentation checks and all other closure gates.

## Test Structure

The test directories encode package ownership:

| Area | Purpose |
|---|---|
| `tests/core/` | Lifecycle, variable maps, functions, gradients, memory, callbacks, and discretization |
| `tests/optimizers/` | Native GG and SLSQP behavior |
| `tests/scipy/` | SciPy adapter values, constraints, gradients, and results |
| `tests/pymoo/` | pymoo variable representations, population evaluation, algorithms, and results |
| `tests/pygmo/` | PyGMO vector/fitness ordering, algorithms, callbacks, and results |
| `tests/test_package.py` | Package import and metadata surface |
| `tests/test_callback_examples.py` | In-process callback example execution and plot output |
| `notebooks/` | Executable public workflows checked separately with nbmake |

Logic tests cover a successful path, a meaningful error path, and a relevant
edge case. For numerical code, assert values, shapes, order, finite or NaN
behavior, bounds, feasibility, and tolerances as applicable. Use deterministic
synthetic problems or public benchmarks. Do not use private application,
customer, or research data as a fixture.

### Lifecycle, Mapping, And Memory

Test behavior before initialization, after initialization, during repeated
evaluation, and after finalization where the lifecycle changes. Registration
tests prove that initialized function lists reject mutation.

Variable-map tests cover names, indices, patterns, ordering, ambiguity, and
missing variables. Memory tests compare cached and uncached values, verify key
identity, and confirm that requests for `problem_results` take the application
path rather than returning incomplete cached data.

### Individuals, Populations, And Functions

Test one individual before testing population vectorization. Population tests
compare `evaluate_population()` with stacked individual results and use at least
two candidates whose values expose ordering mistakes. Cover zero-width integer
or floating arrays where a backend supports a single variable family.

Assert function-object counts separately from scalar component counts. Selection
tests prove that requested components remain in requested order. Constraint tests
cover lower/upper bounds, tolerance, finite-but-infeasible values, and mixed
boundedness when supported.

### Derivatives

For analytical derivatives:

- compare with centered finite differences at differentiable points;
- cover structurally independent entries through dependency masks;
- verify selected component and variable order;
- cover `NaN` as the unavailable-derivative signal;
- verify unresolved derivatives raise with useful context; and
- compare `LocalFD` with analytical expectations for supported step modes and
	population batch sizes.

A derivative unit test does not replace an optimizer integration test when the
change affects backend Jacobian assembly.

### Backends And Optimizers

Backend tests assert compatibility validation, variable/fitness ordering,
constraint translation, callback events, success/feasibility conversion, and
iwopy result containers. Test all affected representations: integer, floating,
mixed, individual, population, single-objective, or multi-objective as relevant.

Optional-backend tests use the declared `test` extra. Keep tests deterministic
and small; do not depend on backend default randomness without an explicit seed.

### Callbacks, Pipelines, And Plots

Callback tests assert `initialize`, ordered `notify`, and `finalize`; normalized
snapshot shapes/dtypes; optional counters/values; and event names. File/plot
tests use temporary directories and assert meaningful artists, labels, values,
or returned objects instead of brittle whole-image pixels.

Pipeline tests assert stage initialization/finalization, selected half-open
range, result propagation, failure stopping, directory naming, and exceptions.

## Examples And Notebooks

Examples and notebooks are public workflows, not scratch space. Keep their
commands, imports, backend extras, and result interpretation current when an API
changes. Run a changed example directly and a changed notebook through nbmake:

```console
uv run python examples/path/run_example.py
uv run pytest --nbmake notebooks/path.ipynb
```

Use deterministic seeds and non-interactive plotting in automated checks. Write
generated files to a temporary or documented result path. Examples must not
depend on confidential data, network access, or an undeclared local checkout.

## Documentation

Public Python APIs follow [docstring conventions](docstrings.md). Types remain
in annotations; NumPy-style docstrings describe the optimization, array,
lifecycle, and backend contract. Sphinx uses numpydoc and AutoAPI for package
references and MyST-NB for Markdown/notebook content.

When an explicit Sphinx rendering check is useful, use the same command as CI:

```console
uv sync --extra doc --upgrade
uv run sphinx-build -E -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` for local inspection. `docs/build/` and
generated AutoAPI pages under `docs/source/_*` are build output; do not edit or
commit them as source.

Update the relevant `docs/source/` page when public behavior, setup, backend
support, result contracts, or examples change. Keep examples runnable. The root
`CHANGELOG.md` is the changelog source; `docs/source/CHANGELOG.md` is a symlink
to it.

Every development change adds a concise bullet to the final version section,
whose heading must match `project.version` from `pyproject.toml`. Preserve the
section's current style and keep the full-changelog link current when release
metadata changes.

## CI Parity

GitHub Actions runs tests and notebooks on every supported Python version,
currently 3.10 through 3.14. GitLab CI runs the same test and notebook commands
in its configured Python image. A local change that passes only on the newest
interpreter is not sufficient when syntax, typing, dependencies, or numerical
behavior can differ by version.

The normal CI commands are:

```console
uv sync --extra test
uv run pre-commit run --all-files
uv run pytest tests
uv run pytest --nbmake notebooks
```

Some pre-commit jobs currently invoke `uv sync --extra test --dev`. The `--dev`
flag is a uv dependency-group flag, not the optional `dev` extra. Local
development uses `--extra dev` as documented above.

Distribution-build jobs deliberately use the PyPA build frontend directly.
That CI implementation is an exception to the local `uv run` rule, not a model
for ordinary development commands.

## Release-Sensitive Changes

`pyproject.toml` is the package-version source and Sphinx reads it directly.
Release tags use `v<version>`; publish CI requires the tag and project version to
agree.

Before a release-sensitive change is complete, check:

1. Public root and subpackage imports.
2. Variable names/order, function component names/order, bounds, and callback
	events.
3. Individual/population shapes and evaluation/finalization tuple ordering.
4. Result-container attributes and backend representation/fitness ordering.
5. Optional import behavior, extras, algorithm selectors, and dependency floors.
6. The matching changelog section, API docs, examples, and notebooks.
7. Tests across the supported Python floor and ceiling where the change is
	version-sensitive.

## Generated And Transient Paths

Do not edit these as implementation sources:

- `build/`, `dist/`, and `iwopy.egg-info/`
- `docs/build/` and generated AutoAPI output
- Python, pytest, mypy, Ruff, notebook, uv, and pre-commit caches
- local results, logs, and optimizer/backend output

Some result-like files are intentional public example or reference assets.
Confirm that a tracked test, example, or documentation page owns the file before
changing it, and regenerate it only when the task explicitly changes that
expected contract.

## Change Checklists

### Core Problem Or Function

- Preserve lifecycle, variable mapping, component order, shapes, and tuple order.
- Declare bounds, dependencies, derivatives, and failure conditions.
- Add focused value/error/edge coverage and population equivalence where
	supported.
- Update exports, API docs, examples, notebooks, and the current changelog
	section where affected.

### Optimizer Or Backend Interface

- Validate problem compatibility before invoking the backend.
- Preserve backend variable, fitness, constraint, and callback ordering.
- Test success, infeasibility/failure, result conversion, and optional imports.
- Cover analytical/numerical gradients and population behavior where supported.
- Update extras, backend docs, examples, and the current changelog section where
	affected.

### Wrapper, Callback, Or Pipeline

- Identify shared versus copied state and lifecycle ownership.
- Preserve wrapped names, shapes, components, and ordering unless transformation
	is the feature.
- Test event/stage order, normalized data, side effects, result propagation, and
	failure behavior.
- For plots or files, follow the applicable visual policy under
	`docs/fraunhofer-design/` and use temporary paths in tests.
