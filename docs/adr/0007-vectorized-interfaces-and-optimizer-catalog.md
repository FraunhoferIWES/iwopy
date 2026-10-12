# ADR-0007: Vectorized Interfaces And Explicit Optimizer Catalog

- Status: Accepted
- Date: 2026-10-12
- Supersedes: None

## Context

Population evaluation is iwopy's primary acceleration contract. SciPy minimize
only batches gradients; Nevergrad/Optuna executors alone still invoke scalar
objectives; DEAP's standard evaluation mapping is scalar. Users need additional
genuine population-search integrations and a catalog demonstrating every named
optimizer exposed by iwopy, including algorithm and method subselectors.

The user confirmed that examples must show explicit optimizer constructors,
without optimizer setup helpers, inside `if __name__ == "__main__":` guards.
Every catalog script must run in tests when its backend is installed, with
package-aware skips otherwise. README and documentation homepage remain brief.

## Decision

- Add `Optimizer_scipy_de` with native vectorization, deferred updates, one worker,
  no polishing, mixed variables, nonlinear constraints, and a last-population
  row cache. Restore original evaluation exceptions when SciPy wraps them.
- Add bounded, single-objective `Optimizer_nevergrad`, `Optimizer_optuna`, and
  `Optimizer_deap`. A private interface-owned population helper validates and
  translates representations/tolerances, tracks evaluated candidates, and
  finalizes through core contracts. No core or existing adapter contract changes.
- Expose eight explicitly validated Nevergrad selectors, two Optuna samplers,
  and three DEAP variation/replacement schemes. Do not expose unsupported or
  sequential-only upstream registries merely because they exist.
- Use tolerance-adjusted finite lower residuals before finite upper residuals.
  Nevergrad receives violations through ordered tell calls; Optuna >=5.0.0 uses
  `Trial.set_constraint`, not deprecated sampler callbacks; DEAP fitness orders
  violation before signed objective loss. Feasible candidate ranking respects
  core roundoff slack and then uses objective alone.
- For Nevergrad, Optuna, and DEAP, success means finite finalized feasibility,
  not convergence. Return the best feasible evaluated candidate, otherwise the
  least-violating unsuccessful one. SciPy DE also requires native convergence.
- Reset state on each solve; evaluate complete/partial batches; dispatch immutable
  original-direction snapshots without extra evaluations. Restore DEAP's Python
  random state in `finally` and require serial execution of its native operators.
- Add lazy narrow extras for Nevergrad >=1.0.12, Optuna >=5.0.0, and DEAP >=1.4.4,
  included in `opt` and `test`. The Nevergrad extra alone requires NumPy <2.4:
  its NGOpt metamodel demonstrably calls `float` on a one-element prediction
  array, which NumPy 2.4 rejects. The continuous catalog case exercises that path.
- Keep one optimizer catalog page with all 79 named/default/extension-point
  entries, each linked to its complete direct constructor script. Share only
  synthetic problem classes. Importing any script or shared problem module
  performs no problem initialization or solving. Remove the superseded runner.
- Test every script's execution/import, explicit constructor/main guard, links,
  and complete selector coverage. Guard runs by installed backends and optional
  PyGMO native features; guard COBYQA by its SciPy version. Simulate missing
  packages without changing the developer's installed environment.
- Retain PyGMO's existing champion-only result contract. Its multi-objective
  catalog cases return native evolved populations and state the limitation;
  they do not pretend to return iwopy Pareto results.

## Consequences

- Expensive objective and constraint calculations use actual iwopy populations.
  Candidate counters and finalization semantics are explicit and tested.
- New adapters deliberately exclude unbounded/fixed variables and multi-objective
  results. Existing pymoo remains the complete iwopy Pareto-result integration.
- Eighty short example modules (79 scripts and shared problem definitions) are
  justified by the user's explicit readability requirement. The single catalog
  page exceeds 700 lines to keep every row's actual script and the new API
  reference together, as requested; production modules remain below 700 lines.
- Native Optuna logging stays caller-controlled. DEAP's temporary global random
  state makes concurrent Python-random consumers unsupported.
- NumPy deprecation warnings remain observable on supported Nevergrad versions.
  TODO(minor): Remove the Nevergrad-only NumPy <2.4 constraint after a supported
  upstream release passes the continuous NGOpt catalog case on NumPy >=2.4.
- Missing packages skip only their own catalog cases; algorithm failures are
  still failures. Native NLopt/IPOPT availability remains build-dependent.

## Alternatives Considered

- Scalar objective calls behind executors: rejected because they do not satisfy
  the population-evaluation requirement.
- All upstream registry entries: rejected because the requested scope is every
  user-selectable iwopy optimizer, not unsupported upstream algorithms.
- Parameterized optimizer setup helpers: rejected by the user's demo-code
  requirement; direct guarded scripts expose every selected parameter.
- Monkey-patching Nevergrad or retaining Optuna's deprecated constraint callback:
  rejected in favor of declared dependencies and current forward-only APIs.
- Expanding PyGMO Pareto finalization: rejected as an unrequested result-contract
  change; documented native evolution demonstrates the available selectors.

## References

- [Architecture](../architecture.md)
- [Naming conventions](../naming-conventions.md)
- [Development and validation](../development.md)
- [Optimizer catalog](../source/optimizer_catalog.rst)
- [Package metadata](../../pyproject.toml)
- [ADR-0001](0001-forward-only-development.md)
