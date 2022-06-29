
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
        return [x**2 + 2 * np.sin(3*y) - y * x]
    
    def ana_grad(self, pvars0_float):
        x, y = pvars0_float
        return np.array([[2*x - y], [6 * np.cos(3*y) - x]])

class Test(unittest.TestCase):

    def setUp(self):
        self.thisdir   = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 0
        np.random.seed(42)

    def print(self, *args):
        if self.verbosity:
            print(*args)
    
    def _calc(self, p, f, p0, o, lim):

        self.print("p0 =",p0)

        g  = p.calc_gradients(pvars0_float=p0, pop=False, order=o)
        self.print("g =", g)

        a = [f.ana_grad(p0)]
        self.print("a =", a)

        d = a[0] - g[0]
        self.print("==> mismatch =", d)

        assert(np.max(d) < lim)

    def test_o1_individual(self):

        self.print("\n\nTEST order 1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2., 2., (1000, 2)):
            self._calc(p, f, p0, 1, 0.01)

    def test_om1_individual(self):

        self.print("\n\nTEST order -1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2., 2., (1000, 2)):
            self._calc(p, f, p0, -1, 0.01)
        
    def test_o2_individual(self):

        self.print("\n\nTEST order 1 INDI")

        p = iwopy.SimpleProblem("test", float_vars=["x", "y"], deltas=0.001)
        f = Obj1(p, "f")
        p.add_objective(f)

        for p0 in np.random.uniform(-2., 2., (1000, 2)):
            self._calc(p, f, p0, 2, 0.01)

if __name__ == '__main__':
    unittest.main()