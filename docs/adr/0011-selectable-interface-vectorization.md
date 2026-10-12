# ADR-0011: Selectable Interface Vectorization

- Status: Accepted
- Date: 2026-10-12
- Supersedes: ADR-0007's population-only evaluation requirement for its four adapters.

## Context

Users need the SciPy DE, Nevergrad, Optuna, and DEAP interfaces to support
individual-only problems while retaining genuine population evaluation by default.

## Decision

Add a boolean `vectorize=True` constructor option to all four adapters. Their
shared evaluator routes each native batch to population evaluation when true,
or ordered individual calls when false. Validate individual component shapes
before aggregation and retain common finiteness, feasibility, and selection checks.
Empty batches invoke neither path. No executor or parallelism is introduced.

SciPy's native deferred vectorized DE mode remains enabled independently so
generation, cache, and callback behavior is identical in both iwopy modes.

## Consequences

- Existing callers keep population evaluation by default.
- Batches, partial budgets, constraints, selection, and callback counters do not
  change; deterministic tests compare both native search paths.
- The catalog describes all interfaces equally while naming their distinct
  controls: `vectorize`, SciPy gradient `vectorized`, and PyGMO `pop`.

## Alternatives Considered

- Separate scalar optimizers would duplicate native search and lifecycle code.
- Changing native SciPy generation mode would alter search semantics unnecessarily.

## References

- [Architecture](../architecture.md)
- [ADR-0007](0007-vectorized-interfaces-and-optimizer-catalog.md)
