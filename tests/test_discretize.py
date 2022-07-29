import numpy as np

import iwopy

def f(x, y):
    return x + np.sin(x - 0.6*y)

def g(x, y):
    return [1 + np.cos(x - 0.6*y), -0.6*np.cos(x - 0.6*y)]

class Obj1(iwopy.Objective):
    
    def n_components(self):
        return 1

    def maximize(self):
        return [False]

    def calc_individual(self, vars_int, vars_float, problem_results):
        x, y = vars_float
        return [f(x, y)]

    def calc_population(self, vars_int, vars_float, problem_results):
        x, y = vars_float[:, 0], vars_float[:, 1]
        return f(x, y)[:, None]

    def ana_grad(self, pvars0_float):
        x, y = pvars0_float
        return np.array(g(x, y))

def test_grad():

    np.random.seed(42)

    dsl = (
        (1, 1, False, 0.01, 0.02, 0.005),
        (1, 1, False, 0.001, 0.002, 0.0005),
        (1, 1, False, 0.001, 0.001, 0.0005),

        (1, 1, True, 0.01, 0.02, 0.005),
        (1, 1, True, 0.001, 0.002, 0.0005),
        (1, 1, True, 0.001, 0.001, 0.0005),

        (2, 2, True, 0.01, 0.02, 5e-5),
        (2, 2, True, 0.001, 0.002, 5e-7),
        (2, 2, True, 0.0001, 0.0002, 5e-9),
    )
    N = 50

    for ox, oy, pop, dx, dy, lim in dsl:

        print("\nENTERING", (ox, oy, pop, dx, dy, lim), "\n")

        p = iwopy.SimpleProblem(
            "test",
            float_vars=["x", "y"],
            min_float_vars={"x": 1., "y": 0.},
            max_float_vars={"x": 2., "y": 3.},
        )
        obj1 = Obj1(p, "f")
        p.add_objective(obj1, varmap_float={"x": "x", "y": "y"})

        gp = iwopy.DiscretizeRegGrid(p, {"x": dx, "y": dy}, fd_order={"x": ox, "y": oy})
        gp.initialize(verbosity=1)

        for p0 in np.random.uniform(1.0, 2.0, (N, 2)):

            print("p0 =", p0)

            g = gp.get_gradients(vars_int=[], vars_float=p0, pop=pop)[0]
            print("g =", g)

            a = obj1.ana_grad(p0)
            print("a =", a)

            d = np.abs(a - g)
            print("==> mismatch =", d, ", max =", np.max(d))

            assert np.max(d) < lim

if __name__ == "__main__":
    test_grad()
