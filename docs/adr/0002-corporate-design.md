# ADR-0002: Corporate Design

- Status: Accepted
- Date: 2026-10-02
- Supersedes: None

## Context

iwopy has no browser frontend or design-token adapter, but it exposes visual
surfaces through matplotlib Pareto-front and optimization-history plots,
examples, notebooks, Sphinx documentation, and project or brand assets.

Repository policy requires an explicit choice between Fraunhofer corporate
design and free design before visual work. Corporate design is the default when
no different choice is recorded, and the iwopy agentic documentation rollout
follows the established FOXES policy and assets.

## Decision

iwopy follows the Fraunhofer corporate design defined in
`docs/fraunhofer-design/`.

The policy applies to plots, examples, notebooks, documentation, and brand
assets. Python plotting APIs preserve caller-supplied matplotlib axes, styles,
colours, labels, and output options unless an API explicitly promises a
corporate preset. Importing iwopy must not mutate global matplotlib state or
require an unavailable corporate font.

iwopy currently has no design-token adapter. A future browser interface, shared
plotting theme, or token adapter is a new architectural feature and must record
its ownership and synchronization mechanism before implementation.

## Consequences

- Visual changes use the approved tokens, chart sequence, typography, image,
	logo, and accessibility guidance where those rules apply.
- Scientific meaning, caller composition, labels, units, and non-colour
	distinctions remain part of the public plotting contract.
- Non-visual numerical code does not acquire web-component or styling
	requirements.
- Choosing free design later requires a superseding ADR and removal of the
	corporate-design instructions and assets in the same change.

## Alternatives Considered

- Free design. Rejected because no project-specific free-design system or
	accessibility policy has been selected, and corporate design is the recorded
	default.
- Leave the choice implicit. Rejected because contributors and coding agents
	would not have a durable source for plotting, documentation, or brand changes.
- Add a browser stack or token adapter now. Rejected because iwopy has no such
	interface and the documentation rollout does not justify new runtime code.

## References

- [Repository instructions](../../AGENTS.md)
- [Architecture](../architecture.md)
- [Fraunhofer IWES UI guidelines](../fraunhofer-design/ui-guidelines.md)
- [Fraunhofer IWES design tokens](../fraunhofer-design/design-tokens.json)
