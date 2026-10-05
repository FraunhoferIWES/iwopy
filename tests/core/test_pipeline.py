from iwopy.core import Pipeline, PipelineStage


class _Stage(PipelineStage):
    def __init__(self, name, success=True):
        super().__init__(name=name)
        self.success = success
        self.received = None

    def run(self, prev_stage=None, prev_results=None, verbosity=1, **kwargs):
        self.received = (prev_stage, prev_results)
        return self.success, f"{prev_results}:{self.name}"


def test_pipeline_finalizes_after_failed_stage(tmp_path):
    pipeline = Pipeline(tmp_path, name="pipeline")
    pipeline.add_stage(_Stage("failed", success=False))

    success, results = pipeline.run(verbosity=0)

    assert success is False
    assert results == "None:failed"
    assert pipeline.running is False
    assert pipeline.initialized is False


def test_pipeline_restarts_with_initial_results(tmp_path):
    first = _Stage("first")
    second = _Stage("second")
    pipeline = Pipeline(tmp_path, name="pipeline")
    pipeline.add_stage(first)
    pipeline.add_stage(second)

    success, results = pipeline.run(
        start_stage=1,
        initial_results="saved",
        finalize=False,
        verbosity=0,
    )

    assert success is True
    assert results == "saved:second"
    assert first.received is None
    assert second.received == (first, "saved")
    assert pipeline.running is False
