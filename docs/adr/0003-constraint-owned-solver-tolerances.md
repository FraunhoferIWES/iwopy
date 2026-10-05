# ADR-0003: Constraint-Owned Solver Tolerances

- Status: Accepted
- Date: 2026-10-05
- Supersedes: None

## Context

Each iwopy `Constraint` already defines its scalar feasibility tolerance and
uses it in individual and population checks. `Problem.add_constraint()` also
copied that value into a component array, making the problem a second owner.
The copy became stale when a constraint tolerance changed after registration.
Backend behavior was inconsistent: PyGMO consumed the problem copy, while the
native SLSQP optimizer previously solved against exact bounds unless callers
enabled a separate compatibility switch.

A solver candidate accepted at the configured tolerance boundary can also land
a few floating-point units outside iwopy's feasibility comparison.

## Decision

`Constraint` is the sole owner of its feasibility tolerance. `Problem` no longer
stores or exposes a tolerance array. Optimizer backends that need component-wise
tolerances derive them from `problem.cons.functions`, expanding each scalar in
constraint registration and component order when the backend adapter is
initialized.

Native SLSQP always applies those tolerances to translated lower and upper
bounds. A positive-tolerance equality is translated to two inequalities. A
zero-tolerance equality remains an equality. Relaxed bounds use a small inward
floating-point reserve so the numerical solution remains inside iwopy's
feasibility check. PyGMO derives its `c_tol` vector through the same constraint
ordering.

There is no switch for exact legacy SLSQP bounds. A caller that requires exact
bounds sets `constraint.tol = 0.0` on the owning constraint.

## Consequences

- Constraint mutation after registration remains authoritative until a backend
  adapter initializes.
- Feasibility checks and solver translations use one source of tolerance truth.
- SLSQP now solves the feasible region iwopy reports, including the default
  `1e-5` tolerance.
- External callers that used `Problem.constraints_tol` must inspect registered
  constraints instead.
- Backend adapters retain derived vectors only where the external backend
  requires that representation.

## Alternatives Considered

- Keep a synchronized tolerance cache on `Problem`. Rejected because it creates
  duplicate ownership and requires mutation tracking.
- Make tolerance application optional in each optimizer. Rejected because it
  permits solver success and iwopy feasibility to describe different regions.
- Relax bounds by the full tolerance without a reserve. Rejected because a
  backend can return a value a few floating-point units beyond the accepted
  boundary.

## References

- [Architecture](../architecture.md#functions-bounds-and-feasibility)
- [Naming conventions](../naming-conventions.md#bounds-feasibility-and-derivatives)
- `iwopy/core/constraint.py`
- `iwopy/optimizers/slsqp.py`
- `iwopy/interfaces/pygmo/problem.py`
- `tests/optimizers/test_slsqp.py`
- `tests/pygmo/test_pygmo.py`
