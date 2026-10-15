# ADR-0019: Test File Dependency Boundary

- Status: Accepted
- Date: 2026-10-15
- Supersedes: Test-input boundary in ADR-0016.

## Context

The user requested complete removal of tests that depend on repository files
outside package code, examples, notebooks, and tests across all three packages.
The documentation-only exclusion still allowed iwopy release and CI assertions.

## Decision

Limit pytest repository-file inputs to `iwopy/`, `examples/`, `notebooks/`, and
`tests/`. Remove release-metadata and CI-workflow tests entirely, including
unused helpers and imports, without skips or replacement fixtures. Retain
runtime package tests and tests that create and read temporary outputs.

## Consequences

Documentation, release metadata, CI configuration, and other repository files
are checked by their separate build, release, or review workflows. No production
API, naming convention, or dependency changes.

## Verification

Audit test inputs, run the retained package tests, then run pre-commit and the
full pytest suite.
