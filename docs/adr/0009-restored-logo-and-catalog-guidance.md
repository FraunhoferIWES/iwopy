# ADR-0009: Restored Logo And Catalog Guidance

- Status: Accepted
- Date: 2026-10-12
- Supersedes: ADR-0008's logo-removal decision only; its other decisions remain.

## Context

After reviewing the documentation, the user requested recovery of the logo,
readable optimizer classifications with recommendations, and API links instead
of embedded catalog class documentation. Both supplied original SVG variants
remain available. Corporate design and the documentation adapter are unchanged.

## Decision

- Restore `Logo_IWOPY.svg` on the white documentation homepage with descriptive
  alternative text and its original aspect ratio. Restore the README image with
  the supplied white variant selected for dark color schemes. Change neither SVG.
- Add compact capabilities and a practical when-to-use recommendation to every
  catalog row. Distinguish single/multi-objective result workflows, variable
  bounds, nonlinear restrictions, mixed variables, and derivative requirements.
  Mark PyGMO native Pareto workflows and pymoo's one-sided constraint translation.
- Recommendations are starting points, not performance claims. Classification
  describes iwopy's exposed workflow rather than every upstream capability.
- Keep direct guarded runnable scripts in the catalog. Link class parameters
  and methods to generated API pages; do not duplicate rendered class blocks.
- Keep table text at 16px. Narrow viewports stack each row with visible field
  labels instead of shrinking the font. Test fields, distinctions, logo/API
  references, script execution/imports, and responsive rendering.

## Consequences

- The logo remains visible without recoloring, inversion, cropping, or distortion.
- Users can compare suitable solvers and follow a setup without reading full
  class reference documentation. Native-only results are not mislabeled as iwopy
  Pareto support.
- The longer table needs wrapping and stacked mobile rows; a small-font workaround
  is deliberately excluded. Screen-reader table headers remain available.
- ADR-0008 remains the design/content/build decision except for its superseded
  removal of the homepage artwork.

## Alternatives Considered

- Removing or manually recoloring the logo: rejected by the user's recovery
  request and the corporate artwork policy.
- Claiming all upstream capabilities or ranking performance generically: rejected
  because suitability depends on iwopy support, problem structure, and budget.
- Repeating complete class blocks in the catalog: rejected in favor of API links.
- Shrinking table typography: rejected in favor of wrapping and labeled mobile rows.

## References

- [ADR-0008](0008-readable-corporate-documentation.md)
- [Architecture](../architecture.md)
- [Optimizer catalog](../source/optimizer_catalog.rst)
- [Development checks](../development.md)
