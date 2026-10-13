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
| [0005](0005-gg-feasible-iterates.md) | Accepted; derivative terminology superseded by 0014 | Preserve feasible GG iterates and backtrack rejected infeasible trial batches. |
| [0006](0006-vectorized-pycma-interface.md) | Accepted; configuration format superseded by 0013 | Evaluate pycma CMA-ES generations in populations and use native augmented-Lagrangian constraints with lazy optional loading. |
| [0007](0007-vectorized-interfaces-and-optimizer-catalog.md) | Accepted; ownership/evaluation/creation/terminology details superseded by 0010/0011/0013/0014 | Add four genuine population-search integrations and a complete catalog of explicit, guarded, package-tested optimizer constructors. |
| [0008](0008-readable-corporate-documentation.md) | Accepted; logo decision superseded by 0009; derivative terminology superseded by 0014; pytest requirement superseded by [0016](0016-documentation-independent-tests.md) | Apply token-backed readable Sphinx styling and accurate, concise user/developer documentation with warning-free rendering checks. |
| [0009](0009-restored-logo-and-catalog-guidance.md) | Accepted | Restore supplied logos, classify/recommend every catalog optimizer, and link class details to the API with readable responsive rows. |
| [0010](0010-public-quadratic-benchmarks.md) | Accepted | Publish reusable quadratic catalog cases through iwopy.benchmarks and remove example-local dependencies. |
| [0011](0011-selectable-interface-vectorization.md) | Accepted; cross-adapter control naming superseded by 0013; derivative terminology superseded by 0014 | Default all four population adapters to vectorize=True while supporting ordered individual evaluation without changing native search. |
| [0012](0012-optimizer-example-guidance.md) | Accepted; direct construction superseded by 0013 | Annotate important optimizer values, keep published demo runs short with production suggestions, and label GG/SLSQP's source iwopy. |
| [0013](0013-optimizer-creation.md) | Accepted | Use a standalone family/algorithm factory and explicit optimizer keywords without legacy signatures; retain public class names and native execution semantics. |
| [0014](0014-sparse-jacobians.md) | Accepted | Use singular Jacobian matrix APIs and CSR assembly/caches, retain native gradient protocols, and pass sparse constraints to SciPy trust-constr. |
| [0016](0016-documentation-independent-tests.md) | Accepted | Keep pytest independent of documentation files and review documentation separately. |

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
