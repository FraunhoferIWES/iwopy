# Architecture Decision Records

This directory stores durable records of significant architectural decisions for
iwopy. ADRs explain why a consequential choice was made; current binding
contracts live in `AGENTS.md`, `docs/architecture.md`, and
`docs/naming-conventions.md`.

## Index

Keep one line per record so a reader can pick the relevant ADR without opening
all of them.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-forward-only-development.md) | Accepted | Develop iwopy forward-only by default and close every change with synchronized tests, documentation, records, and changelog. |
| [0002](0002-corporate-design.md) | Accepted | Apply Fraunhofer corporate design to iwopy's plots, documentation, notebooks, and brand assets. |
| [0003](0003-constraint-owned-solver-tolerances.md) | Accepted | Keep feasibility tolerance on each constraint and derive backend component vectors in registration order. |
| [0004](0004-pipeline-restart-results.md) | Accepted | Seed a selected pipeline range with restart results and always clear running state when execution stops. |
| [0005](0005-gg-feasible-iterates.md) | Accepted | Preserve feasible GG iterates and backtrack rejected infeasible trial batches. |

## When To Add An ADR

Add an ADR when a decision has lasting consequences for lifecycle or module
ownership, variable/function/result shapes or ordering, bounds and feasibility,
derivatives, callbacks, pipelines, backend translation, supported runtimes,
dependency strategy, data ownership, visual policy, or naming rules reused
across iwopy.

## How To Add One

1. Copy `ADR-TEMPLATE.md` to `NNNN-short-kebab-case-title.md`.
2. Increment the highest number and keep the title stable after merge.
3. Record context, the decision, consequences, and rejected alternatives.
4. Use `Accepted` after adoption. When a later record replaces a decision, leave
	the older record unchanged; mark the new record `Supersedes` and link it from
	the index.
5. Add the record to the index above in the same change.

Do not create ADRs for reversible local implementation details with no durable
architectural consequence.
