# ADR-0018: Feasible SLSQP Progress And Guarded Components

- Status: Accepted
- Date: 2026-10-14
- Supersedes: None

## Context

An SLSQP trajectory can improve its objective while repeatedly violating
constraints. Its final endpoint need not be its best feasible evaluation.
Large all-pair constraints also make dense Jacobians and subproblems expensive,
but pruning by current distance alone is unsafe when candidates move.

## Decision

Retain the best finite, fully feasible evaluated point independently of the
primary SciPy endpoint and status. Return that point when available without
reporting unsuccessful primary termination as convergence.

Offer explicit inward solver margins, bounded pre/post feasibility restoration,
and a patience limit on cumulative best-feasible objective improvement. Keep
constraint-owned public tolerances unchanged. Patience starts only after a
fully feasible point exists. Native stopping criteria and iteration budgets
govern the pre-feasibility trajectory.

Enabled restoration for constrained problems requires positive finite
`restoration_scales`, scalar or in full constraint-component order. Divide each
violation by its component-unit characteristic value before squaring and apply
the matching chain rule to the sparse merit Jacobian. Scales change with
constraint units; do not infer them from tolerances or current violations.
Restoration requests only violated-constraint Jacobian rows.

Keep restoration tolerance independent of objective-progress units. Best-point
ranking and stagnation use the standard scalar objective direction and
evaluation contracts, not application-specific objective classes.

Introduce a constraint-owned bound-certificate hook, defaulting to all
components. Map selections to global indices in the problem. SLSQP may reduce
Jacobian and subproblem rows only within enforced variable bounds, optionally
intersected with a solve-local movement box. Refresh before each solve and keep
the component set fixed within SciPy. Always evaluate and check full constraints.

## Consequences

- Feasible output and convergence remain distinct, observable contracts.
- Restoration and stopping budgets are explicit; neither guarantees convergence.
- Mixed constraint units do not implicitly weight restoration when values,
  bounds, tolerances, and characteristic scales are converted consistently.
- Feasible-objective patience cannot abort recovery before any feasible point.
- Safe reduction depends on an owner-supplied certificate, not a distance guess.
- Local boxes limit one solve; moving regions requires another solve.
- Full constraint evaluation remains a cost. Objective FD deltas are unchanged.
- The cohesive SLSQP module exceeds the practical 700-line guideline to keep
  scaling, caches, phases, callbacks, and public contracts under one owner.

## Alternatives Considered

- Return only the raw endpoint: discards useful feasible points.
- Tighten public tolerance globally: changes the physical acceptance contract.
- Prune currently distant components: cannot certify future candidate feasibility.
- Change row count during SciPy: invalidates its fixed subproblem structure.
- Recompute objective gradients for restoration/stopping: unnecessary derivative
  cost; stopping uses existing values and restoration uses constraint derivatives.
- Raw squared violations: implicit weighting depends on arbitrary unit choices.
- Guess scales from tolerance or current residual: zero tolerances and satisfied
  components do not supply meaningful physical characteristic values.

## References

- [Architecture](../architecture.md)
- [Constraint-owned tolerances](0003-constraint-owned-solver-tolerances.md)
