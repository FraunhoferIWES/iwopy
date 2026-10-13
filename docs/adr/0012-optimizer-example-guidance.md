# ADR-0012: Annotated Optimizer Examples And iwopy Source Naming

- Status: Accepted
- Date: 2026-10-13
- Supersedes: Earlier use of native as GG/SLSQP's source label only.

## Context

The complete catalog must make relevant tuning controls visible without presenting
toy budgets as production recommendations or making all example runs expensive.
GG/SLSQP's source should identify iwopy rather than an ambiguous native backend.

## Decision

Keep direct constructors in main guards. Show important values on separate lines,
with inline comments only beside individual values, not the problem argument or
dictionary containers. Use short explicit run limits and small demo populations.
Comments provide production starting suggestions for deliberately small limits
and populations; these are heuristics, not certified convergence settings.

Use iwopy as GG/SLSQP's source and rename their scripts to iwopy_gg.py and
iwopy_slsqp.py. Keep native for external backend behavior and results. Historical
accepted ADRs retain their terminology; this record establishes the current label.

## Consequences

- Catalog tests run the published scripts without budget overrides and protect
  annotated values, explicit short limits, and production guidance.
- Production settings still require tuning for dimension, variable/objective
  scaling, evaluation cost, constraint difficulty, and noise.
- Local PyGMO solvers use one problem-defined start, not an accidental multistart
  population. Actual population optimizers retain their own controls.
- Current documentation, catalog links, and example filenames use iwopy labels;
  optimizer class names and package interfaces do not change.

## Alternatives Considered

- Hidden test-only budget overrides would leave published examples expensive.
- Removing all optional controls would hide the values users need to tune.
- Uniform production populations ignore algorithm and dimension differences.

## References

- [Architecture](../architecture.md)
- [Naming Conventions](../naming-conventions.md)
- [Development](../development.md)
