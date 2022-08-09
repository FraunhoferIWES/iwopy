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


if __name__ == "__main__":

    test_o1_indi()
    test_om1_indi()
