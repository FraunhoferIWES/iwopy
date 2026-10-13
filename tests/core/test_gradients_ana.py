import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

import iwopy


class Obj1(iwopy.Objective):
    def n_components(self):
        return 1

    def maximize(self):
        return [False]

    def calc_individual(self, vars_int, vars_float, problem_results):
        x, y = vars_float
        return [x**2 + 2 * np.sin(3 * y) - y * x]

    def calc_population(self, vars_int, vars_float, problem_results):
        x, y = vars_float[:, 0], vars_float[:, 1]

        def f(x, y):
            return x**2 + 2 * np.sin(3 * y) - y * x

        return f(x, y)[:, None]

    def ana_grad(self, pvars0_float):
        x, y = pvars0_float
        return np.array([2 * x - y, 6 * np.cos(3 * y) - x])

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        grad = self.ana_grad(vars_float)
        return grad[var]


class SingleVarObjective(iwopy.Objective):
    def __init__(self, problem, name, variable, scale):
        super().__init__(problem, name, vnames_float=[variable])
        self.scale = scale

    def n_components(self):
        return 1

    def maximize(self):
        return [False]

    def calc_individual(self, vars_int, vars_float, problem_results):
        return [self.scale * vars_float[0] ** 2]

    def calc_population(self, vars_int, vars_float, problem_results):
        return (self.scale * vars_float[:, 0] ** 2)[:, None]

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        return 2.0 * self.scale * vars_float[0]


class SparseObjective(iwopy.Objective):
    def __init__(self, problem):
        super().__init__(problem, "sparse")
        self.derivative_calls = []

    def n_components(self):
        return 3

    def maximize(self):
        return [False, False, False]

    def vardeps_float(self):
        return np.array(
            [
                [True, False],
                [False, True],
                [False, False],
            ]
        )

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        x, y = vars_float
        values = np.array([x, 2.0 * y, 1.0])
        return values if components is None else values[components]

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        x, y = vars_float.T
        values = np.column_stack([x, 2.0 * y, np.ones(len(x))])
        return values if components is None else values[:, components]

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        selected = np.arange(self.n_components()) if components is None else components
        selected = np.asarray(selected, dtype=int)
        expected = np.flatnonzero(self.vardeps_float()[:, var])
        np.testing.assert_array_equal(selected, expected)
        self.derivative_calls.append((var, selected.tolist()))
        return np.full(len(selected), 1.0 if var == 0 else 2.0)


def _calc(p, f, p0, o, lim, pop):
    print("p0 =", p0)

    g = p.get_jacobian(vars_int=[], vars_float=p0, pop=pop)[0].toarray().ravel()
    print("g =", g)

    a = f.ana_grad(p0)
    print("a =", a)

    d = a - g
    print("==> mismatch =", d)

    assert np.max(d) < lim


def test_o1_indi():
    print("\n\nTEST order 1 INDI")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f, varmap_float={"x": "x", "y": "y"})
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, 1, 0.01, False)


def test_om1_indi():
    print("\n\nTEST order -1 INDI")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f, varmap_float={0: 0, 1: 1})
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, -1, 0.01, False)


def test_o2_indi():
    print("\n\nTEST order 1 INDI")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f)
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, 2, 0.01, False)


def test_o1_pop():
    print("\n\nTEST order 1 POP")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f)
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, 1, 0.01, True)


def test_om1_pop():
    print("\n\nTEST order -1 POP")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f)
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, -1, 0.01, True)


def test_o2_pop():
    print("\n\nTEST order 1 POP")

    p = iwopy.SimpleProblem("test", float_vars=["x", "y"], init_values_float=[0, 0])
    f = Obj1(p, "f")
    p.add_objective(f)
    p.initialize()

    for p0 in np.random.uniform(-2.0, 2.0, (100, 2)):
        _calc(p, f, p0, 2, 0.01, True)


def test_analytical_jacobian_zero_disjoint_variable_dependencies():
    problem = iwopy.SimpleProblem(
        "disjoint",
        float_vars=["x", "y"],
        init_values_float=[0.0, 0.0],
    )
    problem.add_objective(SingleVarObjective(problem, "fx", "x", scale=1.0))
    problem.add_objective(SingleVarObjective(problem, "fy", "y", scale=2.0))
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
    )

    assert isspmatrix_csr(jacobian)
    np.testing.assert_allclose(jacobian.toarray(), [[6.0, 0.0], [0.0, 16.0]])


def test_analytical_jacobian_select_dependent_components():
    problem = iwopy.SimpleProblem(
        "sparse",
        float_vars=["x", "y"],
        init_values_float=[0.0, 0.0],
    )
    objective = SparseObjective(problem)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
    )

    assert isspmatrix_csr(jacobian)
    assert jacobian.nnz == 2
    np.testing.assert_allclose(
        jacobian.toarray(),
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ],
    )
    assert objective.derivative_calls == [(0, [0]), (1, [1])]


def test_sparse_jacobian_preserves_component_and_variable_selection_order():
    problem = iwopy.SimpleProblem(
        "selected", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    problem.add_objective(SparseObjective(problem))
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
        components=[1, 0],
        vars=["y", "x", "y"],
    )

    assert jacobian.shape == (2, 3)
    np.testing.assert_allclose(jacobian.toarray(), [[2.0, 0.0, 2.0], [0.0, 1.0, 0.0]])


def test_sparse_jacobian_respects_function_variable_mapping():
    problem = iwopy.SimpleProblem(
        "mapped", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    objective = SingleVarObjective(problem, "fy", "y", scale=2.0)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32), np.array([3.0, 4.0]), func=objective, vars=["y"]
    )

    np.testing.assert_allclose(jacobian.toarray(), [[16.0]])


@pytest.mark.parametrize(
    ("components", "variables", "shape"),
    [([], None, (0, 2)), (None, [], (3, 0)), ([2], None, (1, 2))],
)
def test_sparse_jacobian_empty_and_independent_selections(components, variables, shape):
    problem = iwopy.SimpleProblem(
        "empty", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    problem.add_objective(SparseObjective(problem))
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
        components=components,
        vars=variables,
    )

    assert isspmatrix_csr(jacobian)
    assert jacobian.shape == shape
    assert jacobian.nnz == 0


def test_sparse_jacobian_does_not_store_exact_zeros(monkeypatch):
    problem = iwopy.SimpleProblem(
        "zero", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    objective = SparseObjective(problem)
    monkeypatch.setattr(objective, "ana_deriv", lambda *args, **kwargs: 0.0)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(np.array([], dtype=np.int32), np.zeros(2))

    assert jacobian.shape == (3, 2)
    assert jacobian.nnz == 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_sparse_jacobian_reports_unresolved_variable(monkeypatch, wrapped):
    problem = iwopy.SimpleProblem(
        "missing", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    objective = SparseObjective(problem)
    monkeypatch.setattr(
        objective,
        "ana_deriv",
        lambda vars_int, vars_float, var, components=None: np.nan if var == 1 else 1.0,
    )
    problem.add_objective(objective)
    problem.initialize(verbosity=0)
    if wrapped:
        problem = iwopy.LocalFD(problem, deltas={"x": 1e-5})
        problem.initialize(verbosity=0)

    with pytest.raises(ValueError, match=r"derivatives for variables \['y'\]"):
        problem.get_jacobian(np.array([], dtype=np.int32), np.array([3.0, 4.0]))


def test_sparse_jacobian_storage_scales_with_nonzero_entries():
    n_vars = 256
    variable_names = [f"x{index}" for index in range(n_vars)]
    problem = iwopy.SimpleProblem(
        "diagonal", float_vars=variable_names, init_values_float=np.ones(n_vars)
    )
    for variable in variable_names:
        problem.add_objective(
            SingleVarObjective(problem, variable, variable, scale=1.0)
        )
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(np.array([], dtype=np.int32), np.ones(n_vars))

    assert jacobian.shape == (n_vars, n_vars)
    assert jacobian.nnz == n_vars
    np.testing.assert_allclose(jacobian.data, 2.0)
    storage = jacobian.data.nbytes + jacobian.indices.nbytes + jacobian.indptr.nbytes
    assert storage < n_vars * n_vars * np.dtype(np.float64).itemsize / 100


@pytest.mark.parametrize("pop", [False, True])
@pytest.mark.parametrize(
    ("components", "variables", "expected"),
    [
        ([1, 0], ["y", "x", "y"], [[2.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
        ([2, 0], ["x"], [[0.0], [1.0]]),
    ],
)
def test_sparse_finite_differences_preserve_selections_and_center_values(
    monkeypatch, pop, components, variables, expected
):
    problem = iwopy.SimpleProblem(
        "selected_fd", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    objective = SparseObjective(problem)
    monkeypatch.setattr(objective, "ana_deriv", lambda *args, **kwargs: np.nan)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)
    problem = iwopy.LocalFD(problem, deltas=1e-5)
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
        components=components,
        vars=variables,
        func_values=np.array([3.0, 8.0, 1.0])[components],
        pop=pop,
    )

    assert isspmatrix_csr(jacobian)
    np.testing.assert_allclose(jacobian.toarray(), expected, atol=1e-8)


@pytest.mark.parametrize("pop", [False, True])
def test_sparse_finite_differences_preserve_analytical_and_independent_entries(
    monkeypatch, pop
):
    problem = iwopy.SimpleProblem(
        "sparse_fd", float_vars=["x", "y"], init_values_float=[0.0, 0.0]
    )
    objective = SparseObjective(problem)
    monkeypatch.setattr(
        objective,
        "ana_deriv",
        lambda vars_int, vars_float, var, components=None: np.full(
            len(components), np.nan if var == 0 else 2.0
        ),
    )
    problem.add_objective(objective)
    problem.initialize(verbosity=0)
    problem = iwopy.LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32), np.array([3.0, 4.0]), pop=pop
    )

    assert isspmatrix_csr(jacobian)
    assert jacobian.nnz == 2
    np.testing.assert_allclose(
        jacobian.toarray(), [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], atol=1e-8
    )


if __name__ == "__main__":
    test_o1_indi()
    test_om1_indi()
