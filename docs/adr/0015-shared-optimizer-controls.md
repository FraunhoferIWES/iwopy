# ADR-0015: Shared Optimizer Controls And Native Limits

## Status

Accepted. Supersedes ADR-0013's population-only pycma evaluation and native
SciPy DE population naming, not its factory ownership or flat-option contract.

## Context

Native limit spellings obscured comparable controls, while population sizes,
pending batches, iteration counters, and evaluation budgets have different
meanings. Universal hard evaluation interception would also count Jacobian and
finalization work differently from native solver counters.

## Decision

- Use nonnegative integer `max_iterations` and positive integer
  `max_evaluations`; reject booleans, unsupported requested units, and
  superseded native keywords instead of aliases or compatibility paths.
- Translate only to existing native limits or adapter-owned counters. Keep
  native convergence criteria and result keys; do not introduce a universal
  tolerance or hard problem-evaluation interceptor.
- Preserve `None` defaults. Candidate-budget adapters Nevergrad, Optuna, and
  DEAP select their finite default of 100 for `None`.
- Keep absolute `pop_size`, pending-work `batch_size`, and SciPy DE's
  `population_multiplier` distinct. DEAP uses `pop_size` for populations and
  offspring and supports an initial-population-inclusive generation cap.
- Compose pymoo limits and custom convergence with OR semantics. Pure-limit
  selectors are `iterations` and `evaluations`; neither supplied cap is dropped.
- Honor `vectorize=False` in pycma using ordered individual calls without
  changing generation order, native ask/tell feedback, or counters.
- Publish parameter explanations and a limit-support/unit table in the core API;
  update maintained callers in iwopy and FOXES Optimization together.

## Consequences

Limit names are portable, not their cost units or convergence guarantees.
Generation-boundary overshoot, native initialization, cache reuse, derivative
work, and selected-point finalization remain explicit adapter contracts. Native
algorithm options beyond shared controls remain flat backend-specific keywords.
Existing cohesive constructors, declarative control-validation lists, and
solver/factory lifecycle methods can exceed
the 40-line guideline to keep translation and state ownership together; this
change does not split them or expand module responsibilities.
