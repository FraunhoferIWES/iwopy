# ADR-0006: Population-Vectorized pycma Interface

- Status: Accepted
- Date: 2026-10-12
- Supersedes: None

## Context

iwopy needs a direct derivative-free CMA-ES interface that preserves genuine
population evaluation of both objectives and constraints. pycma provides an
ask/tell loop, box-bound handling, and, since version 4.4.0, native constrained
ask/tell through an augmented Lagrangian. The adapter must preserve iwopy
lifecycle, objective direction, constraint tolerance ownership, callbacks, and
selected-solution finalization without an eager optional dependency.

## Decision

- Own the adapter in `iwopy.interfaces.cma` as `Optimizer_cma`, with a lazy
  `cma>=4.4.0` extra included in the `opt` and `test` aggregates.
- Support continuous single-objective problems. Reject integer variables,
  multiple objectives, fixed variables, and conflicting native options.
- Require an explicit positive finite `sigma0`; accept trusted native options
  through `cma_pars`, while iwopy owns bounds, representation, termination
  callbacks, verbosity, and logging.
- Evaluate each complete generation once through `evaluate_population`, then
  translate maximization signs and pass fitness and constraint rows to `tell`.
- Translate each finite constraint side to a nonpositive residual using its
  owning constraint's tolerance. Group lower residuals before upper residuals,
  preserving component order. Equalities require a positive tolerance band.
- Use pycma's native augmented Lagrangian, not an adapter-specific penalty.
- Notify callbacks once per completed generation with original iwopy values,
  normalized population arrays, and counters excluding finalization.
- Start a fresh strategy for every solve. Finalize the best feasible candidate
  through the problem and retain its application results. If none was feasible,
  finalize the backend's favorite point and report failure.
- Report success only for a convergence stop with finite, feasible final values;
  budget-only termination is unsuccessful. Expose native `stop_conditions`.
- Disable backend log writing independently of iwopy progress display. Accept
  and document the backend's possible empty `outcmaes/` directory creation.

## Consequences

- Expensive vectorized problems can evaluate an entire CMA-ES generation
  without adapter-driven scalar evaluation or gradient requirements.
- Base iwopy imports remain independent of pycma installation. The minimum
  backend version is tested directly without introducing version branches.
- Constraint penalties and search adaptation remain owned by a proven backend.
  Callbacks and results retain iwopy component order and objective signs.
- Mixed-integer, multi-objective, and exact zero-tolerance equality problems
  require another supported optimizer. Variable scaling remains the caller's
  responsibility.
- pycma may warn when its optional `moarchiving` convergence archive is absent;
  the optimizer and feasibility checks remain functional.

## Alternatives Considered

- Drive pycma through a scalar callable: rejected because it would lose the
  required population batching.
- Implement fixed penalties or rejection sampling: rejected because they add
  tuning or potentially unbounded evaluation loops instead of native constraint
  handling.
- Use older pycma versions with a second constrained execution path: rejected
  because one current backend contract is sufficient and avoids compatibility
  branches.
- Translate mixed-integer or multi-objective representations in this adapter:
  rejected for this interface contract; existing pymoo/PyGMO interfaces cover
  those representations where supported.

## References

- [Architecture](../architecture.md#pycma)
- [Constraint tolerance ownership](0003-constraint-owned-solver-tolerances.md)
- [pycma release history](https://github.com/CMA-ES/pycma#version-history)
- [pycma ask/tell API](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html#tell)
