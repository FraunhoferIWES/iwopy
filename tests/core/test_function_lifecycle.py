import pytest

from iwopy.core.function import OptFunction


class _Problem:
    def var_names_int(self):
        return ["i"]

    def var_names_float(self):
        return ["x"]


class _Function(OptFunction):
    def n_components(self):
        return 1


def test_function_metadata_requires_initialization():
    function = _Function(_Problem(), "f")

    with pytest.raises(RuntimeError, match="has not been initialized"):
        function.component_names
    with pytest.raises(RuntimeError, match="has not been initialized"):
        function.var_names_int
    with pytest.raises(RuntimeError, match="has not been initialized"):
        function.var_names_float

    function.initialize()

    assert function.component_names == ["f"]
    assert function.var_names_int == ["i"]
    assert function.var_names_float == ["x"]
