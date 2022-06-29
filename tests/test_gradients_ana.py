
import numpy as np
import unittest
from pathlib import Path
import inspect

import iwopy


class f(iwopy.Objective):

    def n_components(self):
        return 1

    def maximize(self):
        return [False]
    
    def calc_individual(self, vars_int, vars_float, problem_results):
        x, y = vars_float
        return [x**2 + 2 * np.sin(3*y) - y * x]

class Test(unittest.TestCase):

    def setUp(self):
        self.thisdir   = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 0

    def print(self, *args):
        if self.verbosity:
            print(*args)

    def test_individual(self):
        p = iwopy.SimpleProblem("test", float_vars=["x", "y"])
        f = f(p, "f")
        p.add_objective(f)
        fd = iwopy.FiniteDiff(deltas=0.01)
        g = fd.calc_gradients()

        
        

if __name__ == '__main__':
    unittest.main()