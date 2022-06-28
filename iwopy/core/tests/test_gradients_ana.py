
import numpy as np
import unittest
from pathlib import Path
import inspect

import iwopy

class MyProblem(iwopy.Problem):

    def var_names_float(self):
        return ["x", "y"]
    
    def apply_individual(self, vars_int, vars_float):
        TODO



class Test(unittest.TestCase):

    def setUp(self):
        self.thisdir   = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 0

    def print(self, *args):
        if self.verbosity:
            print(*args)

    def test_individual(self):


        
        

if __name__ == '__main__':
    unittest.main()