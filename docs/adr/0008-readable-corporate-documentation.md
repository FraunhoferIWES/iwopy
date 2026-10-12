# ADR-0008: Readable Corporate Documentation

- Status: Accepted
- Date: 2026-10-12
- Supersedes: None

## Context

The user requested a complete documentation check against Fraunhofer design,
using FOXES as a reference, with readable font sizes, accurate statements, less
repetition, and a user/developer focus. Corporate design is already recorded in
ADR-0002; it remains the policy, including when the user is unavailable.

iwopy's Sphinx theme had no token adapter. Rendered catalog tables inherited
12.8px text, the homepage repeated old releases and generic claims, and setup
guidance duplicated an obsolete force-removal conda workflow. The theme reports
its numpydoc extension as incompatible.

## Decision

- Adapt authoritative design tokens into `docs/source/_static/iwes-tokens.css`,
  checked by `tests/test_documentation.py`; keep corporate styling in `iwes.css`.
  Follow FOXES' existing Sphinx approach, without modifying FOXES.
- Use 16px body, navigation, code, and table-cell text; use 14px only for the
  secondary footer. Keep Roboto, approved type weights, dark readable links,
  flat white surfaces, visible focus, and reduced-motion support. Never recolor
  logo artwork to make it visible; remove the white-on-white homepage image.
- Preserve every source token's value and `needs-review` status. The source
  still flags the mono stack, layout dimensions, and touch-target values; this
  technology adapter is not a design approval. No plot preset or app stack is added.
- Keep README/homepage concise and route users to installation, the complete
  optimizer catalog, examples, callbacks, and API reference. Route contributors
  to testing/development records. Release history belongs in the changelog.
- Remove claims that all optimization paths vectorize and that the grid utility
  itself is the problem wrapper. Name `DiscretizeRegGrid` and distinguish search
  populations from gradient batching. Keep synthetic demonstration budgets and
  backend limitations explicit without expanding entry-page implementation detail.
- Use built-in Sphinx Napoleon for the unchanged NumPy-style docstring contract,
  instead of the incompatible numpydoc extension/dependency. AutoAPI continues
  generating the reference. Align the canonical site URL with package metadata.
- Validate source tokens/includes, a full warnings-as-errors Sphinx build, and
  actual responsive typography/wrapping at 320, 768, and 1280 CSS pixels. Do not
  rerun unchanged notebooks merely to inspect rendering.

## Consequences

- Users can find direct solver setups without reading contributor or backend
  internals first. Developer ownership/gates remain in their durable records.
- Styling drift and missing literal includes fail focused tests. Browser checks
  catch theme-specific typography overrides that source string checks cannot.
- Authored guides and API links build without warnings. External upstream links
  and optional native solver availability remain outside iwopy's control.
- Existing token review flags remain visible in the source and policy; no claim
  of completing an independent design approval is made.

## Alternatives Considered

- Free design: not selected; it conflicts with the user's Fraunhofer requirement.
- Arbitrary CSS values or copied colors without a test: rejected because the
  corporate tokens remain authoritative.
- Suppressing Sphinx warnings: rejected in favor of correcting headings,
  duplicated labels, and the incompatible extension at their sources.
- Repeating full setup/backend internals on entry pages: rejected for user and
  developer navigation, with detailed contracts retained in reference records.

## References

- [ADR-0002](0002-corporate-design.md)
- [Architecture](../architecture.md)
- [Development checks](../development.md)
- [Design guidance](../fraunhofer-design/ui-guidelines.md)
- [Authoritative tokens](../fraunhofer-design/design-tokens.json)
