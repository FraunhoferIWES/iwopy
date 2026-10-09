# ADR-0005: Preserve Feasible GG Iterates

- Status: Accepted
- Date: 2026-10-08
- Supersedes: None

## Context

GG predicts constraint crossings from local derivatives and then evaluates
candidate points. Nonlinear constraints can make every evaluated candidate
infeasible even when the linear prediction was feasible. Accepting the first
such trial discards a feasible solution and may lose objective improvement
during subsequent recovery.

## Decision

Once GG reaches feasibility, accept only feasible improving candidates.
If every trial is infeasible, retain the current variables, objective,
constraints, and feasibility. The existing gradient-memory path reduces the
step on the next iteration. Apply the same policy to individual and population
evaluation. Permit infeasible fallback trials only while recovering from an
initially infeasible point.

## Consequences

- Feasible iteration histories remain feasible and do not lose objective quality.
- Initially infeasible problems retain their existing recovery behavior.
- Difficult constraint geometry may require more backtracking iterations.
- This does not change GG's convergence criterion or certify local optimality.

## Alternatives Considered

- Accepting an infeasible trial and recovering was rejected because it discards
  the current feasible solution without a guaranteed improvement.
- Returning only a best-feasible archive was rejected because it would hide
  the unnecessary infeasible excursions rather than prevent them.

## References

- [Architecture](../architecture.md#native-optimizers)
- [GG implementation](../../iwopy/optimizers/gg.py)
- [Regression tests](../../tests/optimizers/test_gg.py)
