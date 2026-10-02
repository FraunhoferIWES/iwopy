# AI Data Classification Exceptions

[AGENTS.md](../AGENTS.md#data-classification) permits AI tool usage for `public`
and `restricted` data. This register records each case where a user explicitly
released work involving `confidential` or `strictly confidential` data, so the
decision stays traceable. It is a record, not an authorization: other
confidential data needs its own release and entry.

## Rules For This File

- **Describe the data, never reproduce it** — no names, records, values, keys, or
	excerpts. This file must itself stay within `restricted`.
- One entry per release, newest first, identifier `DC-NNNN`, written in the same
	change as the released work.
- Never widen an entry's scope; add a new one and link the old. When a release
	ends, set `Status: Withdrawn` or `Expired` and keep the entry.

## iwopy Data Boundary

iwopy source code, public benchmarks, and published examples are public, but
data supplied to an optimization keeps its own classification. Optimization
variables, objective and constraint inputs, application-specific
`problem_results`, callback histories, backend logs, plots, and optimizer output
can expose or permit inference of the caller's source data.

Ask the classification question from
[AGENTS.md](../AGENTS.md#data-classification) before processing unpublished or
patent-relevant research, customer or partner models, contract or pricing data,
personal data, NDA-covered formulations, or output derived from them. Do not
infer that content is public because iwopy accepts its Python type, array shape,
file format, or backend representation.

Use deterministic synthetic problems, the Branin or Rosenbrock benchmarks, or
explicitly public examples for tests and reproductions. When real values are not
needed, ask for an anonymized schema, array shape, callable signature, or a
synthetic failing case. A release for one model, data set, customer, study, or
derived result does not cover another.

If a software change itself introduces processing of released confidential
data, record that boundary in
[architecture.md](architecture.md#data-and-integration-boundaries) as required
by the repository policy.

## Status

No exceptions have been released.

## Entry Template

Copy this block above the previous entries when recording a release.

```markdown
### DC-NNNN: <short title of the feature, software, or data set>

- Status: Active | Expired | Withdrawn
- Date of release: YYYY-MM-DD
- Released by: <name and role of the person who explicitly released the work>
- Recorded by: <contributor or AI tool that performed the work>

**Covered.** Which feature, software, or data set, and what is explicitly not
covered.

**Why an AI tool.** What it was needed for, and which alternative was rejected
and why.

**Data categories.** The kinds of data involved — never the data itself.

**Protective measures.** What limits the exposure: anonymization, the AI tool and
deployment used, retention, access, deletion after the task.

**Follow-ups.** Open actions, review date, or when the release ends.

**Related records.** ADR, `docs/architecture.md` section, issue, or approval
thread.
```

## Related Records

- [AGENTS.md → Data Classification](../AGENTS.md#data-classification) — classes,
	triggers, procedure.
- [architecture.md](architecture.md) — classification of data the software
	itself processes.
