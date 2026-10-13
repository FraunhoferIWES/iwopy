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
uv run pre-commit run --all-files
uv run pytest tests
```

The dependency sets in `pyproject.toml` are optional extras, not uv dependency
groups. Install them with `--extra`; `--dev` does not select the optional `dev`
extra. Multiple extras are combined by repeating `--extra`.

| Dependency set | Use |
|---|---|
| Main dependencies | NumPy, SciPy, and matplotlib runtime support |
| `scipy` | Explicit SciPy integration selection; SciPy is also a base dependency |
| `cma` | pycma CMA-ES interface, requiring `cma>=4.4.0` for constrained ask/tell |
| `nevergrad` | Nevergrad >=1.0.12 population ask/tell; NumPy <2.4 for upstream metamodel compatibility |
| `optuna` | Optuna >=5.0.0 population ask/tell with the current trial constraint API |
| `deap` | DEAP >=1.4.4 population evolution using native variation/selection |
| `pymoo` | pymoo optimizer interface |
| `pygmo` | PyGMO optimizer interface |
| `opt` | Aggregate all supported optimizer integrations |
| `test` | All solver integrations, pytest, notebook execution, pre-commit, and mypy |
| `dev` | Contributor hooks, mypy, object-size inspection, and Jupyter |
| `doc` | Sphinx, AutoAPI, Napoleon, Immaterial, MyST-NB, and documentation notebooks |

Put a new dependency in the narrowest justified set. A package imported by base
`iwopy` belongs in main dependencies. An optional backend must remain lazy and
belong in its own extra. `uv.lock` is currently ignored and is not a repository
source of dependency versions.

### Security Floors

The 2026-10-13 audit checked all published, non-yanked stable versions admitted
by direct build, runtime, and extra requirements against public OSV advisories.
Builds require wheel >=0.46.2 to exclude
[GHSA-8rrh-rw8j-w5fx](https://github.com/advisories/GHSA-8rrh-rw8j-w5fx)
(path traversal) and
[GHSA-qwmp-2cf2-g9g6](https://github.com/advisories/GHSA-qwmp-2cf2-g9g6)
(regular-expression denial of service). The existing setuptools >=84.0.0
requirement remains unchanged. No other external direct requirement admitted a
release flagged by that audit.

These declarations do not replace an audit of a resolved environment: transitive
dependencies, native libraries, and later advisories remain outside this check.

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
| Objective or constraint contract | `iwopy/core/` | Problem registration, component ordering, Jacobians, and tests |
| Result container or Pareto plotting | `iwopy/core/opt_results.py` | Backend finalization, examples, and output tests |
| Callback or history behavior | `iwopy/core/optimizer_callback.py` | Optimizer dispatch and callback/example tests |
| Optimizer creation/configuration | `iwopy/core/optimizer_factory.py` and the selected constructor | `tests/core/test_optimizer_factory.py`, explicit keywords, optional loading, and examples/notebooks |
| Pipeline behavior | `iwopy/core/pipeline.py` | Stage implementations and pipeline users |
| Simple callable API | `iwopy/wrappers/simple_*.py` | Core function/problem contract and examples |
| Finite differences | `iwopy/wrappers/local_fd.py` | Jacobian assembly and analytical/numerical tests |
| Problem wrapper or discretization | `iwopy/wrappers/` | Underlying core problem, grid utility, and core tests |
| SciPy translation | `iwopy/interfaces/scipy/` | Core Jacobians/results and `tests/scipy/` |
| pycma translation | `iwopy/interfaces/cma/` | Ask/tell batching, constraint bounds/tolerances, callbacks/results, and `tests/cma/` |
| Bounded population adapters | `iwopy/interfaces/_population.py` and `scipy/de.py`, `nevergrad/`, `optuna/`, `deap/` | Shared constraints, selection/finalization, diagnostics, and `tests/interfaces/` |
| Optimizer catalog | `docs/source/optimizer_catalog.rst` and `examples/optimizers/` | Guarded factory calls, package-aware runs, and selector coverage |
| Catalog benchmark problems | `iwopy/benchmarks/quadratic.py` | Public exports, individual/population values, lifecycle, Jacobians, and `tests/test_benchmarks.py` |
| pymoo translation | `iwopy/interfaces/pymoo/` | Factory, representation, results, and `tests/pymoo/` |
| PyGMO translation | `iwopy/interfaces/pygmo/` | UDP ordering, callbacks/results, and `tests/pygmo/` |
| iwopy optimizer | `iwopy/optimizers/` | Core optimizer contract and optimizer tests |
| Public import | Owning package `__init__.py` | Root exports, AutoAPI output, examples, and import tests |

Useful searches include the class or method name, variable/component name,
shape count, backend selector, callback event, and result attribute. Search
examples and notebooks before changing constructors or factory names; they are
public call sites.

Use `create_optimizer` from the `iwopy` root in maintained examples and notebooks.
Keep explicit class keywords aligned where their meanings match, and forward
additional native options as flat keywords. Do not introduce common parameter
dictionaries, class factory wrappers, or compatibility signatures. Factory tests
cover dispatch without initialization, optional loading, and real solves; catalog
tests enforce factory-only creation throughout all examples and notebook cells.
See [ADR-0013](adr/0013-optimizer-creation.md).

Use `max_iterations` and `max_evaluations` for supported run limits,
`pop_size` for absolute algorithmic populations, `batch_size` for pending
Nevergrad/Optuna work, and `population_multiplier` for SciPy DE. Do not forward
native limit aliases or equate these units. Canonical-control tests protect
mapping, invalid values, unsupported limits, and pymoo criterion composition;
catalog guards enforce the same names in all maintained workflows. See
[ADR-0015](adr/0015-shared-optimizer-controls.md).

Do not treat `build/lib/iwopy/` as source. The maintained package is `iwopy/`.

## Focused Validation

Run the narrowest applicable command first:

```console
# One behavior
uv run pytest tests/path/test_module.py::test_behavior -q

# One test module or package area
uv run pytest tests/core/test_jacobian_ana.py -q

# Production type checking, as configured in pyproject.toml
uv run mypy iwopy

# Repository hooks on touched files
uv run pre-commit run --files iwopy/path.py tests/path/test_module.py
```

Before closure, broaden to the repository gates:

```console
# All formatting, linting, type, and file-hygiene hooks, before final tests
uv run pre-commit run --all-files

# Full Python test suite, required when code files changed
uv run pytest tests

# Notebook execution used by CI
uv run pytest --nbmake notebooks

```

Pre-commit owns the configured Ruff and mypy invocations. A hook can modify a
file and still exit non-zero; inspect the diff and rerun it.
New untracked files need an explicit `pre-commit run --files ...` invocation:
`--all-files` alone lists only tracked files. Run final tests after hook edits.

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

## GitHub Actions Coverage

Both `.github/workflows/test.yml` and `.github/workflows/publish.yml` test Python
3.10 through 3.14 on Linux with the complete `test` extra. Before the full suite,
they import every required solver backend and run a named regression stage for
`tests/interfaces`, SciPy DE, and public benchmarks.
This makes catalog execution and scalar/vectorized evaluation coverage visible;
missing backend packages fail preflight instead of silently skipping their cases.
Optional PyGMO IPOPT/NLopt native features may still receive explicit skips.

The regular workflow runs on pushes, pull requests to main/dev, the weekly
schedule, and manual `workflow_dispatch`; manual runs do not publish packages.
Version-tag validation remains in the release workflow. Pre-commit setup uses
`--extra dev --extra test`, not the unrelated uv `--dev` group option.

To check dependency metadata and Linux wheel availability without installing:

```console
uv pip compile pyproject.toml --extra test --python-version 3.14 --python-platform x86_64-unknown-linux-gnu --only-binary=:all: --no-header
```

Repeat for each matrix version. A successful resolution proves availability of a
compatible dependency set, not runtime correctness; the actual CI matrix runs
the regression and full suites. `tests/test_package.py` protects workflow test
paths, dependency preflight, ordering, extra selection, and manual-run publish guards.

## Test Structure

Pytest tests must not read or depend on files under `docs/`. Documentation
validation belongs to separate build and review workflows, not the runtime
suite. See [ADR-0016](adr/0016-documentation-independent-tests.md).

When reviewing rendered documentation, use a warnings-as-errors build:

```console
uv run --extra doc --extra dev --extra test sphinx-build -q -E -b html -W --keep-going -D nb_execution_mode=off docs/source docs/build/html
```

This targets documentation rendering without rerunning unchanged notebooks.
Sphinx uses AutoAPI and Napoleon for NumPy-style API documentation. Check body,
navigation, code, and table text at 320, 768, and 1280 CSS pixels; base text is
16px and content should not overflow the page. Keep entry pages focused on
users/developers, and implementation details in reference pages or decision records.
Review catalog capability/recommendation fields, iwopy Pareto versus native-only
PyGMO results, logo aspect ratios, and API links when checking documentation.

The test directories encode package ownership:

| Area | Purpose |
|---|---|
| `tests/core/` | Lifecycle, variable maps, functions, Jacobians, memory, callbacks, and discretization |
| `tests/optimizers/` | iwopy GG and SLSQP behavior |
| `tests/scipy/` | SciPy adapter values, constraints, Jacobians, and results |
| `tests/cma/` | pycma population batching, objective signs, bounds/tolerances, callbacks, lifecycle, and stop/success conversion |
| `tests/pymoo/` | pymoo variable representations, population evaluation, algorithms, and results |
| `tests/pygmo/` | PyGMO vector/fitness ordering, algorithms, callbacks, and results |
| `tests/interfaces/` | New population-adapter contracts plus complete catalog execution, guarded optional installations, inert imports, and selector coverage |
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

`Problem.get_jacobian()` and `calc_jacobian()` use CSR storage. Matrix tests
assert shape, order, `nnz`, missing-derivative errors, and implicit zeros.
Compare numerical values through `.toarray()` only in small expectations;
storage regressions sum `data.nbytes`, `indices.nbytes`, and `indptr.nbytes`
without claiming process-wide or peak-memory measurements. Solver tests verify
sparse caches, sparse `trust-constr` constraint callbacks, required dense SLSQP
callbacks, and PyGMO's non-densifying native `gradient` protocol.

### Backends And Optimizers

Backend tests assert compatibility validation, variable/fitness ordering,
constraint translation, callback events, success/feasibility conversion, and
iwopy result containers. Test all affected representations: integer, floating,
mixed, individual, population, single-objective, or multi-objective as relevant.

Optional-backend tests use the declared `test` extra. Keep tests deterministic
and small; do not depend on backend default randomness without an explicit seed.

For pycma, test a solve with individual evaluation disabled, the fitness and
constraint rows passed to `tell`, maximization signs, general constraint bounds,
equality tolerance bands, and unsuccessful budget or infeasible stops. Include
silent and verbose solves in a temporary directory; no backend log files may be
written, though pycma can create an empty `outcmaes/` directory. Import tests block
the optional backend in a fresh process. Check the minimum supported backend:

```console
uv run --with 'cma==4.4.0' pytest tests/cma tests/test_package.py -q
```

For the new adapters and complete catalog, run:

```console
uv run pytest tests/interfaces tests/scipy/test_de.py tests/test_package.py -q
uv run pytest tests/interfaces/test_optimizer_examples.py -q
uv run python examples/optimizers/scipy_de.py
```

The catalog suite executes all 81 explicit scripts in isolated directories and
also imports each script without running its `__main__` block. `importorskip`
guards each backend's cases; absent PyGMO IPOPT/NLopt native features and
SciPy versions without COBYQA get explicit skips. Six simulated missing-backend
tests verify that unrelated SciPy cases remain available. With the `test` extra,
all available catalog cases run; skips must not hide algorithm failures.
The original 79 selector setups are joined by constrained NSGA2/NSGA3 Pareto
flows. All four pymoo Pareto flows check two-objective shape, feasible constraints,
and explicit weighted selection. Category tests require every setup exactly once
in the six non-empty objective/constraint/search groups and omit empty categories.
Scripts import shared cases from `iwopy.benchmarks` without adding an example
directory to `sys.path`. Benchmark tests cover initialized construction, known
values, individual/population agreement, mixed Jacobians, constraint boundaries,
and rejected late registration. The interface overview covers all adapters rather
than distinguishing new integrations from established ones.

Keep catalog runs short with explicit iteration/evaluation limits and small
populations. Tests inspect the actual published values; they do not silently
reduce production-sized settings during execution. AST/token checks protect one
commented value per line, uncommented argument/dictionary containers, and
production starting suggestions beside demo limits. Those suggestions are
heuristics: tune for dimension, scaling, cost, constraints, and numerical noise.
The GG/SLSQP cases and source labels use `iwopy`, not `native`.
Use `--durations=10` on the catalog suite when checking observed run times.

For SciPy DE, Nevergrad, Optuna, and DEAP, `vectorize=True` selects population
evaluation; false evaluates candidates individually in native batch order. Tests
forbid the opposite path, compare seeded search results/callback counters across
modes, and cover empty/singleton batches and malformed scalar results. Candidate
budgets, partial batches, constraint selection, and native generation remain unchanged.

Population regressions forbid individual search evaluation, verify full/partial
batches, mixed and pure integer variables, objective directions, general and
equality constraints, tolerances/roundoff, result payloads, failed-trial cleanup,
callback failures, repeated solves, malformed/nonfinite outputs, and disabled
callback allocations. The continuous NGOpt case reaches its upstream metamodel
and verifies the Nevergrad-only NumPy <2.4 dependency constraint. Upstream NumPy
deprecation warnings and existing SciPy ignored-bound warnings are documented;
do not silence them by changing application-global logging or warning settings.

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
lifecycle, and backend contract. Sphinx uses Napoleon and AutoAPI for package
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

Keep the final changelog heading and link, installed distribution version,
runtime `iwopy.__version__`, and `CITATION.cff` software `version` synchronized.
The citation's `cff-version` is its schema version, not the package version;
preserve the preferred article's publication metadata. Package regression tests
check the current release values against project metadata.

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
- Cover analytical/numerical Jacobians and population behavior where supported.
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
