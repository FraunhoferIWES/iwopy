# ADR-0010: Public Quadratic Benchmarks

- Status: Accepted
- Date: 2026-10-12
- Supersedes: ADR-0007's example-local shared-case ownership only.

## Context

The catalog's local `case_problems` imports required a sibling file and test path
injection. Users need the cases available from an installed iwopy package.

## Decision

Move the continuous, Pareto, and mixed quadratic definitions to
`iwopy.benchmarks.quadratic`. Export `QuadraticProblem`, `ParetoProblem`, and
`MixedProblem` from `iwopy.benchmarks`, keeping their formulas, bounds, ordering,
and silently initialized constructors. Update every catalog caller and remove
the local module without an alias or fallback. Imports create no problem instances.

## Consequences

- Catalog setups work without an example-local dependency or path injection.
- These classes are public benchmark contracts, covered by types, docstrings,
  API documentation, and deterministic evaluation/lifecycle tests.
- Construction fixes registered functions; the optional Pareto restriction is
  selected before initialization, not added afterward.

## Alternatives Considered

- Retaining local helpers would keep examples dependent on repository layout.
- Exporting from the iwopy root would unnecessarily widen its curated API.

## References

- [Architecture](../architecture.md)
- [ADR-0007](0007-vectorized-interfaces-and-optimizer-catalog.md)
