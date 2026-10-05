# ADR-0004: Pipeline Restart Results

- Status: Accepted
- Date: 2026-10-05
- Supersedes: None

## Context

`Pipeline.run()` could select a later stage with `start_stage`, but the first
selected stage had no supported way to receive a persisted result from an
earlier run. Callers therefore could not restart a pipeline at an existing
application state through the generic pipeline contract.

Stopping iteration early after a stage returned `success=False` also left the
pipeline iterator's running flag set. The subsequent default finalization then
failed because a running pipeline cannot be finalized, masking the stage
failure with a lifecycle assertion.

## Decision

`Pipeline.run()` accepts `initial_results`. The first selected stage receives
that object as `prev_results`. When `start_stage > 0`, it also receives the
immediately preceding registered stage as `prev_stage`, preserving stage
identity without rerunning that stage.

The run method clears the running flag and restores the pipeline's configured
stage selection in a `finally` block. This cleanup occurs after successful
execution, a stage-reported failure, or an exception. A stage-reported failure
can therefore follow the normal optional finalization path and return its
`(False, results)` pair without being masked.

## Consequences

- Application packages can restart later stages from persisted, typed-by-
  convention result payloads without adding pipeline-specific global state.
- Restart loading remains the responsibility of the application package; iwopy
  propagates an `object | None` payload.
- The preceding stage object is available for stage logic, but it is not
  initialized or run again as part of the restart.
- Exceptions still propagate to the caller after pipeline running-state cleanup.

## Alternatives Considered

- Require callers to assign restart data directly to a stage. Rejected because
  it bypasses the pipeline result-propagation contract and couples callers to
  stage internals.
- Rerun all earlier stages to reconstruct results. Rejected because stages can
  be expensive or intentionally skipped during a restart.
- Exhaust the iterator after a failed stage. Rejected because lifecycle cleanup
  should not depend on continuing iteration after execution has stopped.

## References

- [Architecture](../architecture.md#pipelines)
- `iwopy/core/pipeline.py`
- `tests/core/test_pipeline.py`
