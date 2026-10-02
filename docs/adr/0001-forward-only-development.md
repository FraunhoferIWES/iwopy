# ADR-0001: Forward-Only Development

- Status: Accepted
- Date: 2026-10-02
- Supersedes: None

## Context

iwopy is an established scientific Python library whose problem, function,
array, derivative, optimizer, callback, pipeline, and result contracts are used
across native and external solver integrations. A change can therefore leave
the repository internally inconsistent when implementation, tests, public
docstrings, examples, notebooks, architecture, terminology, or release notes
describe different generations of the same contract.

Maintaining old and new behavior together also expands the combinations that
must be understood and tested. Compatibility branches, deprecated aliases,
dual formats, and fallback paths can outlive their purpose unless their removal
is explicit.

The repository needs one default direction for API evolution and one definition
of when a development change is complete.

## Decision

iwopy development is strictly forward-looking by default.

When a contract changes, implement the target contract directly and update all
maintained in-repository callers, tests, examples, notebooks, and documentation
in the same change. Remove the superseded implementation and do not add or
retain a compatibility shim, deprecated alias, dual format, fallback branch, or
migration-only path by default.

A legacy path is allowed only when the user explicitly authorizes a bounded
exception. The same change must add an ADR that defines its scope, owner, and an
objective removal condition.

A development change closes only when:

1. Required focused tests and the full runtime test suite pass when code files
	changed. Documentation-only changes run their applicable documentation checks.
2. Repository-wide pre-commit passes after the final edit.
3. Every affected public Python API has an accurate NumPy-style docstring.
4. Architecture, naming, development guidance, API documentation, examples,
	notebooks, and other affected iwopy information are synchronized.
5. The final current-version section of `CHANGELOG.md` records the change and
	matches `project.version` from `pyproject.toml`.

The policy applies to changes from this decision onward. It does not require an
otherwise focused change to remove unrelated historical compatibility code.

## Consequences

- Each merged contract has one maintained implementation and one documented
	meaning.
- Backends, wrappers, examples, and notebooks move with core API changes instead
	of relying on indefinite compatibility behavior.
- Contributors must identify the full in-repository impact before completing a
	contract change.
- Changes can be larger when a public contract has many maintained callers.
- Consumers outside the repository may need to update without a deprecation
	window unless a bounded exception was explicitly approved.
- Completion takes longer than a narrow code patch because tests, docstrings,
	durable records, and changelog evidence are part of the same change.

## Alternatives Considered

- Retain deprecated aliases and compatibility branches for every public change.
	Rejected because it multiplies maintained behavior and has no reliable removal
	point.
- Decide compatibility ad hoc in each implementation. Rejected because similar
	changes would receive inconsistent treatment and hidden legacy paths would be
	easy to introduce.
- Merge code once focused tests pass and update documentation later. Rejected
	because stale optimization contracts, shapes, ordering, and backend guidance
	are correctness defects for library users.
- Require removal of every historical compatibility path immediately. Rejected
	because unrelated cleanup would undermine focused changes and make adoption of
	the policy impractical.

## References

- [Repository instructions](../../AGENTS.md)
- [Architecture](../architecture.md)
- [Development guide](../development.md)
- [Docstring conventions](../docstrings.md)
- [Naming conventions](../naming-conventions.md)
