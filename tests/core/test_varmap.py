import numpy as np

import iwopy


class Obj1(iwopy.Objective):
    def __init__(self, problem, name):
        super().__init__(problem, name, n_vars_int=1, n_vars_float=2)

    def n_components(self):
        return 3

    def maximize(self):
        return [False, False, False]

    @classmethod
    def f(cls, a, x, y):
        return [a, x, y]

    def calc_individual(self, vars_int, vars_float, problem_results):
        a = vars_int[0]
        x, y = vars_float
        return np.array(self.f(a, x, y))

    def calc_population(self, vars_int, vars_float, problem_results):
        a = vars_int[:, 0]
        x, y = vars_float[:, 0], vars_float[:, 1]
        return np.column_stack(self.f(a, x, y))


class Con1(iwopy.Constraint):
    def __init__(self, problem, name):
        super().__init__(problem, name, n_vars_int=2, n_vars_float=3)

    def n_components(self):
        return 1

    def get_bounds(self):
        return [-2099], [-2000]

    @classmethod
    def f(cls, n, a, x, k, p):
        return n + a - x + k - p

    def calc_individual(self, vars_int, vars_float, problem_results):
        n, a = vars_int
        x, k, p = vars_float
        return np.array(self.f(n, a, x, k, p))

    def calc_population(self, vars_int, vars_float, problem_results):
        n, a = vars_int[:, 0], vars_int[:, 1]
        x, k, p = vars_float[:, 0], vars_float[:, 1], vars_float[:, 2]
        return self.f(n, a, x, k, p)[:, None]



def test_indi():

    print("\n\nTEST INDI")

    problem = iwopy.SimpleProblem(
        "test", 
        int_vars=["A", "N"], 
        float_vars=["X", "Y", "K", "P"],
        init_values_int=[0, 0],
        init_values_float=[0, 0, 0, 0],
    )
    problem.add_objective(
        Obj1(problem, "f"), varmap_int={0: "A"}, varmap_float={0: "X", 1: "Y"}
    )
    problem.add_constraint(
        Con1(problem, "g"),
        varmap_int={0: "N", 1: "A"},
        varmap_float={0: "X", 1: "K", 2: "P"},
    )
    problem.initialize(verbosity=1)

    N = 100
    ivars = np.c_[np.arange(N), -np.arange(N)]
    fvars = np.c_[
        1000 + np.arange(N),
        2000 + np.arange(N),
        3000 + np.arange(N),
        4000 + np.arange(N),
    ]
    for n in range(N):

        varsi = ivars[n]
        varsf = fvars[n]

        ovals, cvals = problem.evaluate_individual(varsi, varsf)

        ovals_direct = np.array(Obj1.f(varsi[0], *varsf[:2]))
        assert np.all(ovals == ovals_direct)

        cvals_direct = np.array(Con1.f(varsi[1], varsi[0], varsf[0], *varsf[2:]))
        assert np.all(cvals == cvals_direct)

        assert np.all(problem.check_constraints_individual(cvals))

def test_pop():

    print("\n\nTEST POP")

    problem = iwopy.SimpleProblem(
        "test", 
        int_vars=["A", "N"], 
        float_vars=["X", "Y", "K", "P"],
        init_values_int=[0, 0],
        init_values_float=[0, 0, 0, 0],
    )
    problem.add_objective(
        Obj1(problem, "f"), varmap_int={0: "A"}, varmap_float={0: "X", 1: "Y"}
    )
    problem.add_constraint(
        Con1(problem, "g"),
        varmap_int={0: "N", 1: "A"},
        varmap_float={0: "X", 1: "K", 2: "P"},
    )
    problem.initialize(verbosity=1)

    N = 100
    varsi = np.c_[np.arange(N), -np.arange(N)]
    varsf = np.c_[
        1000 + np.arange(N),
        2000 + np.arange(N),
        3000 + np.arange(N),
        4000 + np.arange(N),
    ]

    ovals, cvals = problem.evaluate_population(varsi, varsf)

    ovals_direct = np.column_stack(Obj1.f(varsi[:, 0], varsf[:, 0], varsf[:, 1]))
    assert np.all(ovals == ovals_direct)

    cvals_direct = Con1.f(
        varsi[:, 1], varsi[:, 0], varsf[:, 0], varsf[:, 2], varsf[:, 3]
    )[:, None]
    assert np.all(cvals == cvals_direct)

    assert np.all(problem.check_constraints_population(cvals))


if __name__ == "__main__":

    test_indi()
    test_pop()
