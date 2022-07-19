import numpy as np
import unittest
from pathlib import Path
import inspect

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
        return np.array([[2 * x - y], [6 * np.cos(3 * y) - x]])


class Test(unittest.TestCase):
    def setUp(self):
        self.thisdir = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 0
        np.random.seed(42)

    def print(self, *args):
        if self.verbosity:
            print(*args)

    def _calc(self, p, f, p0, o, lim, pop):

        self.print("p0 =", p0)

        g = p.calc_gradients(pvars0_float=p0, pop=pop, order=o)
        self.print("g =", g)

        a = [f.ana_grad(p0)]
        self.print("a =", a)

        d = a[0] - g[0]
        self.print("==> mismatch =", d)

        assert np.max(d) < lim

    def test_o1_indi(self):

        self.print("\n\nTEST order 1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, 1, 0.01, False)

    def test_om1_indi(self):

        self.print("\n\nTEST order -1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, -1, 0.01, False)

    def test_o2_indi(self):

        self.print("\n\nTEST order 1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, 2, 0.01, False)

    def test_o1_pop(self):

        self.print("\n\nTEST order 1 POP")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, 1, 0.01, True)

    def test_om1_pop(self):

        self.print("\n\nTEST order -1 POP")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, -1, 0.01, True)

    def test_o2_pop(self):

        self.print("\n\nTEST order 1 POP")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2.0, 2.0, (1000, 2)):
            self._calc(p, f, p0, 2, 0.01, True)


if __name__ == "__main__":
    unittest.main()
