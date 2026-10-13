# ADR-0013: Standalone Optimizer Creation And Explicit Parameters

- Status: Accepted
- Date: 2026-10-13
- Supersedes: Construction/configuration portions of ADR-0006, ADR-0007,
  ADR-0011, and ADR-0012; their numerical, batching, and example-guidance
  decisions remain in force.

## Context

Maintained examples and notebooks need one discoverable construction boundary.
Common parameter dictionaries hide backend-specific contracts, while renaming
public classes or changing numerical algorithms is outside this change.

## Decision

Export the plain `create_optimizer(family, optimizer, problem, **kwargs)` function
from `iwopy` and `iwopy.core`, owned by core's `optimizer_factory.py`. Maintained
examples and notebooks import the same function object from the root. It dispatches to known
classes without initializing or solving. Keep existing public class names and
their direct constructors; do not add `Optimizer.new` or a factory class.

Use explicit per-class parameters, aligning `optimizer`, `vectorize`,
`memory_size`, and absolute `pop_size` where meanings match. Forward additional
native settings as flat keywords. SciPy DE keeps its native per-dimension
`popsize`; pycma always evaluates populations. Adapter-private native mappings
are implementation details, not public configuration groups.

Expose `nlopt` as a family implemented by PyGMO. Its selected leaf solver is
translated to PyGMO's NLopt driver internally, with no nested caller selector.
All maintained examples and notebooks use the function. Remove superseded
signatures, shared option groups, and class factory entry points without compatibility aliases,
fallbacks, or compatibility branches.

## Consequences

- Family/algorithm selection and backend constructor contracts are independently
  testable, including optional dependency loading and real solves.
- Existing class names, numerical settings, seeds, budgets, callbacks, result
  handling, and backend-specific import timing remain unchanged.
- Callers must supply current explicit keywords. Native subsolver objects remain
  native backend configuration, not outer iwopy optimizer construction.

## Alternatives Considered

- A class factory or `Optimizer.new`: rejected because construction needs no
  factory instance or base-class responsibility.
- Common option dictionaries: rejected in favor of discoverable explicit
  controls and backend-specific flat forwarding.
- Public class renaming or legacy signatures: rejected by the confirmed scope
  and forward-only policy.

## References

- [Architecture](../architecture.md#creation-and-configuration)
- [Naming conventions](../naming-conventions.md#optimizers-interfaces-and-factories)
- [Creation API](../source/api_core.rst)
