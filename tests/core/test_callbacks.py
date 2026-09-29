import numpy as np
import pytest

from iwopy.core import (
    OptimizationHistory,
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
)


class DummyProblem:
    name = "dummy"
    initialized = True
    n_objectives = 1
    maximize_objs = np.array([False])

    class Objectives:
        component_names = ["objective"]

    objs = Objectives()


class DummyOptimizer(Optimizer):
    def solve(self, verbosity=1, callbacks=None):
        super().solve(verbosity, callbacks)
        data = OptimizerCallbackData(
            event="iteration",
            iteration=1,
            vars_int=np.array([], dtype=np.int32),
            vars_float=np.array([1.0]),
            objs=np.array([2.0]),
            cons=np.array([], dtype=np.float64),
        )
        self._notify_callbacks(data)
        return self._finalize_callbacks("results")


class RecordingCallback(OptimizerCallback):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def initialize(self, optimizer):
        super().initialize(optimizer)
        self.calls.append((self.name, "initialize"))

    def notify(self, data):
        self.calls.append((self.name, data.event))

    def finalize(self, results):
        self.calls.append((self.name, results))


class MutationAttemptCallback(OptimizerCallback):
    def __init__(self):
        super().__init__()
        self.blocked = False

    def notify(self, data):
        try:
            data.vars_float.setflags(write=True)
        except ValueError:
            self.blocked = True


class ValueRecordingCallback(OptimizerCallback):
    def __init__(self):
        super().__init__()
        self.values = []

    def notify(self, data):
        self.values.append(data.vars_float[0, 0])


def test_callback_data_is_normalized_and_immutable():
    vars_float = np.array([1.0, 2.0])

    data = OptimizerCallbackData(
        event="iteration",
        vars_int=np.array([], dtype=np.int32),
        vars_float=vars_float,
        objs=np.array([3.0]),
        cons=np.array([], dtype=np.float64),
    )
    vars_float[0] = 9.0

    assert data.vars_int.shape == (1, 0)
    assert data.vars_float.shape == (1, 2)
    assert data.objs.shape == (1, 1)
    assert data.cons.shape == (1, 0)
    assert data.vars_float[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        data.vars_float[0, 0] = 4.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        data.vars_float.setflags(write=True)


def test_callback_data_rejects_inconsistent_population_sizes():
    with pytest.raises(ValueError, match="inconsistent sizes"):
        OptimizerCallbackData(
            event="evaluation",
            vars_int=np.zeros((2, 0), dtype=np.int32),
            vars_float=np.zeros((3, 1)),
            objs=np.zeros((2, 1)),
            cons=np.zeros((2, 0)),
        )


def test_optimizer_dispatches_callback_lifecycle_in_order():
    calls = []
    callbacks = [RecordingCallback("first", calls), RecordingCallback("second", calls)]
    optimizer = DummyOptimizer(DummyProblem())
    optimizer.initialize(verbosity=0)

    results = optimizer.solve(verbosity=0, callbacks=callbacks)

    assert results == "results"
    assert callbacks[0].optimizer is optimizer
    assert calls == [
        ("first", "initialize"),
        ("second", "initialize"),
        ("first", "iteration"),
        ("second", "iteration"),
        ("first", "results"),
        ("second", "results"),
    ]


def test_callback_cannot_mutate_state_seen_by_later_callback():
    mutation = MutationAttemptCallback()
    recorder = ValueRecordingCallback()
    optimizer = DummyOptimizer(DummyProblem())
    optimizer.initialize(verbosity=0)

    optimizer.solve(verbosity=0, callbacks=[mutation, recorder])

    assert mutation.blocked
    assert recorder.values == [1.0]


def test_optimizer_rejects_invalid_callback_collection():
    optimizer = DummyOptimizer(DummyProblem())
    optimizer.initialize(verbosity=0)

    with pytest.raises(TypeError, match="list"):
        optimizer.solve(verbosity=0, callbacks=())
    with pytest.raises(TypeError, match="OptimizerCallback"):
        optimizer.solve(verbosity=0, callbacks=[object()])


def test_optimization_history_resets_and_plots_objective():
    optimizer = DummyOptimizer(DummyProblem())
    optimizer.initialize(verbosity=0)
    history = OptimizationHistory()

    optimizer.solve(verbosity=0, callbacks=[history])
    optimizer.solve(verbosity=0, callbacks=[history])
    figure = history.plot_objective()

    assert len(history.states) == 1
    np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [1])
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [2.0])