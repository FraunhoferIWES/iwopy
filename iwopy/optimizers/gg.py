import numpy as np

from iwopy.core import Optimizer, SingleObjOptResults


class GG(Optimizer):
    """
    Greedy Gradient (GG) optimizer, for local optimum
    search with constraints.

    Follows steepest decent, reducing step size
    in a finite number of steps on the way. Step directions
    that violate constraints are projected out or reversed.

    Parameters
    ----------
    problem: iwopy.Problem
        The problem to optimize
    step_max : float or list or dict
        The maximal steps. Either uniform float value
        or list of floats for each problem variable,
        or dict with entry for each variable
    step_min : float or list or dict
        The minimal steps. Either uniform float value
        or list of floats for each problem variable,
        or dict with entry for each variable
    step_div_factor : float
        Step size division factor until step_min is reached
    f_tol : float
        The objective function tolerance
    vectorized : bool
        Flag for running in vectorized mode
    n_max_steps : int
        The maximal number of steps without fresh gradient
    memory_size : int
        The number of memorized visited points
    name: str, optional
        The name

    Attributes
    ----------
    step_max : numpy.ndarray
        Maximal step size for each problem variable,
        shape: (n_vars_float,)
    step_min : numpy.ndarray
        Minimal step size for each problem variable,
        shape: (n_vars_float,)
    step_div_factor : float
        Step size division factor until step_min is reached
    f_tol : float
        The objective function tolerance
    vectorized : bool
        Flag for running in vectorized mode
    n_max_steps : int
        The maximal number of steps without fresh gradient
    memory_size : int
        The number of memorized visited points
    memory : tuple
        Memorized data: (x, obj, grad, all_valid), each a
        numpy.ndarray, shapes: (memory_size, n_vars),
        (memory_size, n_vars), (memory_size,), (memory_size,)

    """

    def __init__(
        self,
        problem,
        step_max,
        step_min,
        step_div_factor=10.0,
        f_tol=1e-8,
        vectorized=True,
        n_max_steps=100,
        memory_size=100,
        name="GG",
    ):
        super().__init__(problem, name)
        self.step_max = step_max
        self.step_min = step_min
        self.step_div_factor = step_div_factor
        self.f_tol = f_tol
        self.vectorized = vectorized
        self.n_max_steps = n_max_steps
        self.memory_size = memory_size
        self.memory = None

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if self.problem.n_objectives != 1:
            raise ValueError(
                f"Optimizer '{self.name}': Not applicable for multi-objective problems."
            )
        if self.problem.n_vars_int != 0:
            raise ValueError(
                f"Optimizer '{self.name}': Not applicable for problems with integer variables."
            )
        if self.problem.n_vars_float == 0:
            raise ValueError(
                f"Optimizer '{self.name}': Missing float variables in problem."
            )

        n_vars = self.problem.n_vars_float
        smax = np.zeros(n_vars, dtype=np.float64)
        if isinstance(self.step_max, dict):
            for i, vname in enumerate(self.problem.var_names_float()):
                if vname in self.step_max:
                    smax[i] = self.step_max[vname]
                else:
                    raise KeyError(
                        f"Optimizer '{self.name}': Missing step_max entry for variable '{vname}'"
                    )
        elif isinstance(self.step_max, list) or isinstance(self.step_max, np.ndarray):
            if len(self.step_max) != n_vars:
                raise ValueError(
                    f"Optimizer '{self.name}': step_max has wrong size {len(self.step_max)} for {n_vars} variables"
                )
            smax[:] = self.step_max
        else:
            smax[:] = self.step_max
        self.step_max = smax

        smin = np.zeros(n_vars, dtype=np.float64)
        if isinstance(self.step_min, dict):
            for i, vname in enumerate(self.problem.var_names_float()):
                if vname in self.step_min:
                    smin[i] = self.step_min[vname]
                else:
                    raise KeyError(
                        f"Optimizer '{self.name}': Missing step_min entry for variable '{vname}'"
                    )
        elif isinstance(self.step_min, list) or isinstance(self.step_min, np.ndarray):
            if len(self.step_min) != n_vars:
                raise ValueError(
                    f"Optimizer '{self.name}': step_max has wrong size {len(self.step_min)} for {n_vars} variables"
                )
            smin[:] = self.step_min
        else:
            smin[:] = self.step_min
        self.step_min = smin

        n_funcs = 1 + self.problem.n_constraints
        self.memory = (
            np.zeros((self.memory_size, n_vars), dtype=np.float64),
            np.zeros((self.memory_size, n_funcs, n_vars), dtype=np.float64),
            np.zeros(self.memory_size, dtype=np.float64),
            np.zeros(self.memory_size, dtype=bool),
        )

        super().initialize(verbosity)

    def print_info(self):
        """
        Print solver info, called before solving
        """
        s = f"  Optimizer '{self.name}'  "
        print(s)
        hline = "-" * len(s)
        print(hline)
        for i, vname in enumerate(self.problem.var_names_float()):
            print(
                f" ({i}) {vname}: step size {self.step_min[i]:.2e} -- {self.step_max[i]:.2e}"
            )
        print(hline)

    def _get_newx(self, x, deltax):
        """
        Helper function for new x creation
        """
        n_vars = self.problem.n_vars_float
        newx = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
        newx[:] = x[None, :]
        for i in range(self.n_max_steps):
            newx[i] += np.sum(deltax[: i + 1], axis=0)

        mi = self.problem.min_values_float()[None, :]
        sel = np.where(newx < mi)
        newx[sel[0], sel[1]] = mi[0, sel[1]]

        ma = self.problem.max_values_float()[None, :]
        sel = np.where(newx > ma)
        newx[sel[0], sel[1]] = ma[0, sel[1]]

        return newx

    def _grad2deltax(self, grad, step):
        """
        Helper function for deltax creation
        """
        j = np.argmax(np.abs(grad) / step)
        return grad * step[j] / np.abs(grad[j])

    def solve(self, verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.core.SingleObjOptResults
            The optimization results object

        """
        super().solve(verbosity)

        # prepare:
        inone = np.array([], dtype=np.int32)
        n_vars = self.problem.n_vars_float
        maximize = self.problem.maximize_objs[0]
        imem = 0
        nmem = 0

        # evaluate initial variables:
        x = np.array(self.problem.initial_values_float(), dtype=np.float64)
        obs, cons = self.problem.evaluate_individual(inone, x)
        obs0 = obs[0]
        valid = self.problem.check_constraints_individual(cons)

        if verbosity > 0:
            s = f"{'it':<5} | {'Objective':<9} | cviol | level"
            hline = "-" * (len(s) + 1)
            print("\nRunning GG")
            print(hline)
            print(s)
            print(hline)

        step = self.step_max.copy()
        count = -1
        level = 0
        while not np.all(step < self.step_min):

            count += 1
            recover = not np.all(valid)

            # check memory:
            sel = (
                np.max(np.abs(x[None, :] - self.memory[0][:nmem]), axis=1) < 1e-13
                if nmem > 0
                else np.array([False])
            )
            if np.any(sel):
                jmem = np.where(sel)[0][0]
                grads = self.memory[1][jmem]
                step /= self.step_div_factor
                level += 1
            else:

                # fresh calculation:
                grads = self.problem.get_gradients(inone, x, pop=self.vectorized)
                step = self.step_max.copy()
                level = 0

                # memorize:
                jmem = imem
                self.memory[0][jmem] = x
                self.memory[1][jmem] = grads
                self.memory[2][jmem] = obs[0]
                self.memory[3][jmem] = not recover
                imem = (imem + 1) % self.memory_size
                nmem = min(nmem + 1, self.memory_size)

            if verbosity > 0:
                print(f"{count:>5} | {obs[0]:9.3e} | {np.sum(~valid):>5} | {level:>5}")

            # project out directions of constraint violation:
            grad = grads[0].copy() if not maximize else -grads[0]
            deltax = self._grad2deltax(-grad, step)
            ncons = cons + np.einsum("cd,d->c", grads[1:], deltax)
            nvalid = ncons <= 0  # self.problem.check_constraints_individual(ncons)
            newbad = valid & ~nvalid
            newgood = ~valid & nvalid
            cnews = newgood | newbad
            for ci in np.where(~valid | cnews)[0]:
                n = grads[1 + ci] / np.linalg.norm(grads[1 + ci])
                grad -= np.dot(grad, n) * n

            # follow grad, but move downwards along violated directions:
            deltax = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
            deltax[:] = self._grad2deltax(-grad, step)[None, :]
            for ci in np.where(~valid & ~cnews)[0]:
                deltax[:] += self._grad2deltax(-grads[1 + ci], step)

            # linear approximation when crossing constraint bondary:
            for ci in np.where(cnews)[0]:
                m = np.linalg.norm(grads[1 + ci])
                if np.abs(m) > 0:
                    deltax[0] -= grads[1 + ci] * cons[ci] / m**2
            newx = self._get_newx(x, deltax)

            if not len(newx):
                continue

            """
            # for debugging
            import matplotlib.pyplot as plt
            pres = self.problem.base_problem.apply_individual(inone, x)
            fig = self.problem.get_fig(pres)
            ax = fig.axes[0]
            for i, xy in enumerate(pres):
                ax.annotate(str(i), xy)
            plt.show()
            plt.close(fig)
            """

            if self.vectorized:

                # calculate population:
                inonep = np.zeros((len(newx), 0), dtype=np.int32)
                obsp, consp = self.problem.evaluate_population(inonep, newx)
                validp = self.problem.check_constraints_population(consp)
                valc = np.all(validp, axis=1)

                # evaluate population results:
                if np.any(valc):
                    if recover:
                        i = np.where(valc)[0][0]
                        x = newx[i]
                        obs = obsp[i]
                        cons = consp[i]
                        valid = validp[i]

                    else:

                        # find best:
                        obsp = obsp[valc]
                        if maximize:
                            i = np.argmax(obsp)
                            if obsp[i][0] <= obs[0]:
                                i = -1
                        else:
                            i = np.argmin(obsp)
                            if obsp[i][0] >= obs[0]:
                                i = -1

                        if i >= 0:
                            x = newx[valc][i]
                            done = np.abs(obs[0] - obsp[i][0]) <= self.f_tol
                            obs = obsp[i]
                            cons = consp[valc][i]
                            valid = validp[valc][i]
                            if done:
                                break

                else:
                    x = newx[0]
                    obs = obsp[0]
                    cons = consp[0]
                    valid = validp[0]

            # non-vectorized:
            else:
                anygood = False
                done = False
                for i, hx in enumerate(newx):

                    obsh, consh = self.problem.evaluate_individual(inone, hx)
                    validh = self.problem.check_constraints_individual(consh)

                    if i == 0:
                        hx0 = hx
                        obsh0 = obsh
                        consh0 = consh
                        validh0 = validh

                    if np.all(validh):
                        anygood = True
                        if recover:
                            x = hx
                            obs = obsh
                            cons = consh
                            valid = validh
                            break

                        else:
                            if maximize:
                                better = obsh[0] > obs[0]
                            else:
                                better = obsh[0] < obs[0]
                            if better:
                                x = hx
                                done = np.abs(obs[0] - obsh[0]) <= self.f_tol
                                obs = obsh
                                cons = consh
                                valid = validh
                    if done:
                        break

                if not anygood:
                    x = hx0
                    obs = obsh0
                    cons = consh0
                    valid = validh0

        if verbosity > 0:
            print(f"{hline}\n")

        # final evaluation:
        pres, obs, cons = self.problem.finalize_individual(inone, x, verbosity)
        valid = self.problem.check_constraints_individual(cons, verbosity)
        if maximize:
            better = obs[0] > obs0
        else:
            better = obs[0] < obs0
        success = np.all(valid) and better

        return SingleObjOptResults(
            self.problem,
            success,
            inone,
            x,
            obs,
            cons,
            pres,
        )
