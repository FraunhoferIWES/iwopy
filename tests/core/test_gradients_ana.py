import numpy as np

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

    def calc_individual(self, vars_int, vars_float, problem_results):
        x, y = vars_float
        return np.array([x, 2.0 * y, 1.0])

    def calc_population(self, vars_int, vars_float, problem_results):
        x, y = vars_float.T
        return np.column_stack([x, 2.0 * y, np.ones(len(x))])

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        selected = np.arange(self.n_components()) if components is None else components
        selected = np.asarray(selected, dtype=int)
        expected = np.flatnonzero(self.vardeps_float()[:, var])
        np.testing.assert_array_equal(selected, expected)
        self.derivative_calls.append((var, selected.tolist()))
        return np.full(len(selected), 1.0 if var == 0 else 2.0)


def _calc(p, f, p0, o, lim, pop):
    print("p0 =", p0)

    g = p.get_gradients(vars_int=[], vars_float=p0)[0]
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


def test_analytical_gradients_zero_disjoint_variable_dependencies():
    problem = iwopy.SimpleProblem(
        "disjoint",
        float_vars=["x", "y"],
        init_values_float=[0.0, 0.0],
    )
    problem.add_objective(SingleVarObjective(problem, "fx", "x", scale=1.0))
    problem.add_objective(SingleVarObjective(problem, "fy", "y", scale=2.0))
    problem.initialize(verbosity=0)

    gradients = problem.get_gradients(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(gradients, [[6.0, 0.0], [0.0, 16.0]])


def test_analytical_gradients_select_dependent_components():
    problem = iwopy.SimpleProblem(
        "sparse",
        float_vars=["x", "y"],
        init_values_float=[0.0, 0.0],
    )
    objective = SparseObjective(problem)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)

    gradients = problem.get_gradients(
        np.array([], dtype=np.int32),
        np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(
        gradients,
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ],
    )
    assert objective.derivative_calls == [(0, [0]), (1, [1])]


if __name__ == "__main__":
    test_o1_indi()
    test_om1_indi()
