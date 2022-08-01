import numpy as np

from iwopy import DiscretizeRegGrid
from iwopy.benchmarks.branin import BraninProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

def run_branin_ipopt(init_vals, dx, dy, tol, pop):

    prob = BraninProblem(initial_values=init_vals)
    gprob = DiscretizeRegGrid(prob, deltas={"x": dx, "y": dy})
    gprob.initialize()

    solver = Optimizer_pygmo(
        gprob, 
        problem_pars=dict(
            grad_pop=pop
        ),
        algo_pars=dict(
            type="ipopt",
            tol=tol
        ),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    return results

def test_branin_ipopt():

    cases = (
        (0.001, 0.001, 1e-6, (1., 1.), 0.397887, (9.42478, 2.475), 5e-6, (0.0008, 0.002), False,),
        (0.0001, 0.0001, 1e-7, (1., 1.), 0.397887, (9.42478, 2.475), 5e-7, (7e-5, 1.1e-4), False,),
        (0.001, 0.001, 1e-6, (-3, 12.), 0.397887, (-np.pi, 12.275), 7e-6, (0.001, 0.0016), True,),
        (0.001, 0.001, 1e-6, (3, 3.), 0.397887, (np.pi, 2.275), 3e-6, (0.0005, 0.00015), True,),
    )
    
    for dx, dy, tol, ivals, f, xy, limf, limxy, pop in cases:

        print("\nENTERING", (dx, dy, tol, ivals, f, xy, limf, limxy, pop), "\n")

        results = run_branin_ipopt(ivals, dx, dy, tol, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0]-f)
        print("delf =", delf,", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float-np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy,", lim =", limxy)
        assert np.all(delxy < limxy)

if __name__ == "__main__":
    test_branin_ipopt()
