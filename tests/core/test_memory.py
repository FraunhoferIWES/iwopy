import numpy as np

from iwopy.core.memory import Memory


def test_memory_replaces_values_and_evicts_oldest_entry():
    memory = Memory(size=2, keyf=lambda vars_int, vars_float: tuple(vars_int))
    vars_float = np.array([], dtype=np.float64)
    constraints = np.array([0.0])

    memory.store_individual(
        np.array([1]), vars_float, np.array([1.0]), constraints
    )
    memory.store_individual(
        np.array([2]), vars_float, np.array([2.0]), constraints
    )
    memory.store_individual(
        np.array([2]), vars_float, np.array([20.0]), constraints
    )

    assert memory.size == 2
    assert memory.lookup_individual(np.array([1]), vars_float) is not None
    result = memory.lookup_individual(np.array([2]), vars_float)
    assert result is not None
    np.testing.assert_array_equal(result[0], np.array([20.0]))

    memory.store_individual(
        np.array([3]), vars_float, np.array([3.0]), constraints
    )

    assert memory.size == 2
    assert memory.lookup_individual(np.array([1]), vars_float) is None
    assert memory.lookup_individual(np.array([2]), vars_float) is not None
    assert memory.lookup_individual(np.array([3]), vars_float) is not None