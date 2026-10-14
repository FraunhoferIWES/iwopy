# ADR-0017: GG Call-Local Finite-Difference Caps

- Status: Accepted
- Date: 2026-10-09
- Supersedes: None

## Context

GG computes gradients at newly visited coordinates and reuses cached gradients
while reducing rejected-trial steps. Local finite differences previously used
their configured deltas even when the current optimizer step was smaller.
Refreshing a full numerical gradient during each backtracking reduction can be
expensive. The developer explicitly requires no additional gradient evaluations.

## Decision

Add `max_deltas` to the problem Jacobian API, in complete problem float-variable
order. LocalFD applies `min(configured_delta, max_delta)` per variable for that
call only. Analytical derivatives and fixed-grid stencils remain unchanged.

At its existing fresh-gradient calls, GG supplies its current per-variable step
before resetting it to `step_max`. Do not change cache lookup, invalidate cached
gradients when steps shrink, or add gradient evaluations during backtracking.
The feasible-iterate rule in ADR-0005 remains unchanged.

## Consequences

- Fresh local finite differences cannot use deltas larger than the current GG
  step, and configured deltas remain available unchanged to subsequent callers.
- Backtracking retains its existing gradient-evaluation schedule. Changed
  numerical directions can still change the overall optimization trajectory
  and the number of iterations required.
- Cached derivative deltas may exceed later, smaller backtracking steps.
- Smaller differences can amplify evaluation noise and cancellation; this
  rule is not an accuracy or convergence guarantee.
- The cap limits delta, not stencil extent. Second-order one-sided boundary
  formulas can sample at twice the capped delta.
- Fixed-grid derivatives keep their mesh resolution rather than being
  redefined as local derivatives of an interpolated problem.

## Alternatives Considered

- Recompute gradients whenever the cap shrinks: rejected by the developer
  because it adds gradient evaluations.
- Mutate LocalFD's configured deltas: rejected because the cap belongs to one
  call and must not leak into other solvers or subsequent uncapped calls.
- Cap after resetting the step to `step_max`: rejected because it loses the
  reduced current step at existing gradient calls following accepted trials.

## References

- [Architecture](../architecture.md#derivatives)
- [Naming conventions](../naming-conventions.md#bounds-feasibility-and-derivatives)
- [ADR-0005](0005-gg-feasible-iterates.md)
