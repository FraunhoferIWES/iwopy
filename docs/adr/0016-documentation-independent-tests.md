# ADR-0016: Documentation-Independent Tests

- Status: Accepted
- Date: 2026-10-13
- Supersedes: Documentation-dependent pytest requirements in ADR-0008.

## Context

Documentation tests require stylesheets, tokens, catalog pages, and branding
assets in isolated package test environments. The user requested their complete
removal across FOXES, FOXES Optimization, and iwopy.

## Decision

Pytest tests must not read or depend on files under `docs/`. Delete the
documentation test module and catalog-document assertions, including their CI
targets. Retain executable example, selector, package, and workflow tests.
Corporate design and separate documentation builds and review remain unchanged.

## Consequences

Package test sandboxes need runtime examples and metadata, but no documentation
fixtures. Token parity, authored links, catalog prose, and branding are reviewed
outside pytest. No documentation-dependent checks are replaced with skips.

## Alternatives

Copying documentation into test sandboxes or conditionally skipping tests
retains the dependency and was rejected.

## Verification

Run focused optimizer-example and workflow tests, then full pytest with
documentation-file reads blocked, plus pre-commit.
