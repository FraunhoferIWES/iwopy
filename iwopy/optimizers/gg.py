import numpy as np

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)


class GG(Optimizer):
    """
    Greedy Gradient (GG) optimizer, for local optimum
    search with constraints.

    Follows steepest descent, reducing step size
    in a finite number of steps on the way. Step directions
    that violate constraints are projected out or reversed.
    Once a feasible point is reached, infeasible trial batches leave
    it unchanged and trigger step reduction. Infeasible trial points
    may be accepted only while recovering from an infeasible start.
    """

    def __init__(
        self,
        problem: Problem,
        step_max: float | list[float] | np.ndarray | dict[str, float],
        step_min: float | list[float] | np.ndarray | dict[str, float],
        step_div_factor: float = 2.0,
        f_tol: float | None = 1e-8,
        vectorized: bool = True,
        n_max_steps: int = 100,
        memory_size: int = 100,
        name: str = "GG",
        max_iterations: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The problem to optimize
        step_max
            The maximal steps. Either uniform float value
            or list of floats for each problem variable,
            or dict with entry for each variable
        step_min
            The minimal steps. Either uniform float value
            or list of floats for each problem variable,
            or dict with entry for each variable
        step_div_factor
            Step size division factor until step_min is reached
        f_tol
            The objective function tolerance
        vectorized
            Flag for running in vectorized mode
        n_max_steps
            The maximal number of steps without fresh gradient
        memory_size
            The number of memorized visited points
        name
            The name
        max_iterations
            Exit criteria based on number of iterations, None for no limit
        """
        super().__init__(problem, name)
        self.step_max = step_max
        self.step_min = step_min
        self.step_div_factor = step_div_factor
        self.f_tol = f_tol if f_tol is not None else 0.0
        self.vectorized = vectorized
        self.n_max_steps = n_max_steps
        self.memory_size = memory_size
        self.memory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.max_iterations = max_iterations
        self.n_iterations = 0

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
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
        if not np.isfinite(self.step_div_factor) or self.step_div_factor <= 1.0:
            raise ValueError(
                f"Optimizer '{self.name}': step_div_factor must be greater than 1."
            )
        if (
            isinstance(self.n_max_steps, (bool, np.bool_))
            or not isinstance(self.n_max_steps, (int, np.integer))
            or self.n_max_steps < 1
        ):
            raise ValueError(
                f"Optimizer '{self.name}': n_max_steps must be a positive integer."
            )
        if (
            isinstance(self.memory_size, (bool, np.bool_))
            or not isinstance(self.memory_size, (int, np.integer))
            or self.memory_size < 1
        ):
            raise ValueError(
                f"Optimizer '{self.name}': memory_size must be a positive integer."
            )
        if self.max_iterations is not None and (
            isinstance(self.max_iterations, (bool, np.bool_))
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 0
        ):
            raise ValueError(
                f"Optimizer '{self.name}': max_iterations must be a non-negative integer or None."
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
        elif isinstance(self.step_max, (list, np.ndarray)):
            if len(self.step_max) != n_vars:
                raise ValueError(
                    f"Optimizer '{self.name}': step_max has wrong size {len(self.step_max)} for {n_vars} variables"
                )
            smax[:] = self.step_max
        else:
            smax[:] = self.step_max

        smin = np.zeros(n_vars, dtype=np.float64)
        if isinstance(self.step_min, dict):
            for i, vname in enumerate(self.problem.var_names_float()):
                if vname in self.step_min:
                    smin[i] = self.step_min[vname]
                else:
                    raise KeyError(
                        f"Optimizer '{self.name}': Missing step_min entry for variable '{vname}'"
                    )
        elif isinstance(self.step_min, (list, np.ndarray)):
            if len(self.step_min) != n_vars:
                raise ValueError(
                    f"Optimizer '{self.name}': step_max has wrong size {len(self.step_min)} for {n_vars} variables"
                )
            smin[:] = self.step_min
        else:
            smin[:] = self.step_min
        if not np.all(np.isfinite(smax)) or not np.all(smax > 0):
            raise ValueError(
                f"Optimizer '{self.name}': step_max must contain positive finite values."
            )
        if not np.all(np.isfinite(smin)) or not np.all(smin > 0):
            raise ValueError(
                f"Optimizer '{self.name}': step_min must contain positive finite values."
            )
        if np.any(smax < smin):
            raise ValueError(
                f"Optimizer '{self.name}': step_max must be greater than or equal to step_min."
            )
        self.step_max = smax
        self.step_min = smin

        n_funcs = 1 + self.problem.n_constraints
        self.memory = (
            np.zeros((self.memory_size, n_vars), dtype=np.float64),
            np.zeros((self.memory_size, n_funcs, n_vars), dtype=np.float64),
            np.zeros(self.memory_size, dtype=np.float64),
            np.zeros(self.memory_size, dtype=bool),
        )

        super().initialize(verbosity)

    def print_info(self) -> None:
        """Print solver info, called before solving"""
        super().print_info()

        s = f"  Optimizer '{self.name}'  "
        print(s)
        hline = "-" * len(s)
        print(hline)
        assert isinstance(self.step_min, np.ndarray)
        assert isinstance(self.step_max, np.ndarray)
        for i, vname in enumerate(self.problem.var_names_float()):
            print(
                f" ({i}) {vname}: step size {self.step_min[i]:.2e} -- {self.step_max[i]:.2e}"
            )
        print(hline)

    def _get_newx(self, x: np.ndarray, deltax: np.ndarray) -> np.ndarray:
        """Helper function for new x creation"""
        n_vars = self.problem.n_vars_float
        newx = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
        newx[:] = x[None, :]
        for i in range(self.n_max_steps):
            newx[i] += np.sum(deltax[: i + 1], axis=0)

        mi = np.asarray(self.problem.min_values_float(), dtype=np.float64)[None, :]
        sel = np.where(newx < mi)
        newx[sel[0], sel[1]] = mi[0, sel[1]]

        ma = np.asarray(self.problem.max_values_float(), dtype=np.float64)[None, :]
        sel = np.where(newx > ma)
        newx[sel[0], sel[1]] = ma[0, sel[1]]

        return newx

    def _grad2deltax(self, grad: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Helper function for deltax creation"""
        if not np.any(np.abs(grad) > 0):
            return np.zeros_like(grad)
        j = np.argmax(np.abs(grad) / step)
        if np.abs(grad[j]) == 0.0:
            return np.zeros_like(grad)
        return grad * step[j] / np.abs(grad[j])

    def _constraint_side(
        self, value: float, minimum: float, maximum: float
    ) -> tuple[float, float]:
        """Return the signed constraint gradient direction toward violation."""
        if value > maximum:
            return 1.0, maximum
        if value < minimum:
            return -1.0, minimum
        return 0.0, value

    def _constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get constraint bounds from the problem or its function list."""
        minimum = self.problem.min_values_constraints
        maximum = self.problem.max_values_constraints
        if minimum is None or maximum is None:
            bounds = [f.get_bounds() for f in self.problem.cons.functions]
            if bounds:
                minimum = np.concatenate([b[0] for b in bounds])
                maximum = np.concatenate([b[1] for b in bounds])
            else:
                minimum = np.array([], dtype=np.float64)
                maximum = np.array([], dtype=np.float64)
        return minimum, maximum

    def _report_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        objs: np.ndarray,
        cons: np.ndarray,
        valid: np.ndarray,
        level: int,
        step: np.ndarray,
        verbosity: int,
    ) -> None:
        """Report the current completed iteration to stdout and callbacks."""
        if verbosity > 0:
            print(
                f"{iteration:>5} | {objs[0]:9.3e} | {np.sum(~valid):>5} | {level:>5} | {np.min(step):>5.3e} | {np.max(step):>5.3e}"
            )
        if self._has_callbacks:
            self._notify_callbacks(
                OptimizerCallbackData(
                    event="iteration",
                    iteration=iteration,
                    vars_int=np.array([], dtype=np.int32),
                    vars_float=x,
                    objs=objs,
                    cons=cons,
                )
            )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults:
        """
        Run the optimization solver.

        Feasible iterates are replaced only by feasible improving trials.
        If all trials are infeasible, keep the current point and reduce
        the step on the next iteration. Recovery steps may remain
        infeasible only until the first feasible point is reached.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for completed optimizer iterations

        Returns
        -------
        results
            The selected solution and its final objective, constraints,
            and problem results. Success requires feasibility and either
            recovery from an infeasible start, net objective improvement,
            or an objective change within ``f_tol`` of the initial point.

        Raises
        ------
        ValueError
            If an objective or constraint gradient is non-finite.
        """
        super().solve(verbosity, callbacks)
        step_max = self.step_max
        step_min = self.step_min
        memory = self.memory
        assert isinstance(step_max, np.ndarray)
        assert isinstance(step_min, np.ndarray)
        assert memory is not None

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
        initially_valid = np.all(valid)

        if verbosity > 0:
            s = f"{'it':<5} | {'Objective':<9} | cviol | level | min step | max step"
            hline = "-" * (len(s) + 1)
            print("\nRunning GG")
            print(hline)
            print(s)
            print(hline)

        step = step_max.copy()
        count = 0
        self.n_iterations = 0
        level = 0
        done = False
        stalled = False
        cmins, cmaxs = self._constraint_bounds()
        while not np.all(step < step_min):
            # exit criteria based on number of iterations:
            if self.max_iterations is not None and count >= self.max_iterations:
                if verbosity > 0:
                    print(
                        f"GG: Reached maximum number of iterations {self.max_iterations}, stopping."
                    )
                break
            recover = not np.all(valid)

            # check memory:
            sel = (
                np.max(np.abs(x[None, :] - memory[0][:nmem]), axis=1) < 1e-13
                if nmem > 0
                else np.array([False])
            )
            if np.any(sel):
                jmem = np.where(sel)[0][0]
                grads = memory[1][jmem]
                step /= self.step_div_factor
                level += 1
            else:
                # fresh calculation:
                grads = self.problem.get_gradients(inone, x, pop=self.vectorized)
                if not np.all(np.isfinite(grads)):
                    raise ValueError(
                        f"Optimizer '{self.name}': Non-finite objective or constraint gradient at current point."
                    )
                step = step_max.copy()
                level = 0

                # memorize:
                jmem = imem
                memory[0][jmem] = x
                memory[1][jmem] = grads
                memory[2][jmem] = obs[0]
                memory[3][jmem] = not recover
                imem = (imem + 1) % self.memory_size
                nmem = min(nmem + 1, self.memory_size)

            count += 1
            self.n_iterations = count

            # project out directions of constraint violation:
            grad = grads[0].copy() if not maximize else -grads[0]
            deltax = self._grad2deltax(-grad, step)
            ncons = cons + np.einsum("cd,d->c", grads[1:], deltax)
            nvalid = self.problem.check_constraints_individual(ncons)
            newbad = valid & ~nvalid
            newgood = ~valid & nvalid
            cnews = newgood | newbad
            for ci in np.where(~valid | cnews)[0]:
                cmin = cmins[ci]
                cmax = cmaxs[ci]
                value = ncons[ci] if newbad[ci] else cons[ci]
                side, _ = self._constraint_side(value, cmin, cmax)
                norm = np.linalg.norm(grads[1 + ci])
                if side == 0.0:
                    continue
                if norm == 0.0:
                    stalled = True
                    break
                n = side * grads[1 + ci] / norm
                grad -= np.dot(grad, n) * n

            if stalled:
                self._report_iteration(
                    count, x, obs, cons, valid, level, step, verbosity
                )
                break

            # follow grad, but move downwards along violated directions:
            deltax = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
            deltax[:] = self._grad2deltax(-grad, step)[None, :]
            for ci in np.where(~valid & ~cnews)[0]:
                cmin = cmins[ci]
                cmax = cmaxs[ci]
                side, _ = self._constraint_side(cons[ci], cmin, cmax)
                if side != 0.0:
                    deltax[:] += self._grad2deltax(-side * grads[1 + ci], step)

            # linear approximation when crossing constraint bondary:
            for ci in np.where(cnews)[0]:
                m = np.linalg.norm(grads[1 + ci])
                if np.abs(m) > 0:
                    cmin = cmins[ci]
                    cmax = cmaxs[ci]
                    value = ncons[ci] if newbad[ci] else cons[ci]
                    _, target = self._constraint_side(value, cmin, cmax)
                    deltax[0] += grads[1 + ci] * (target - cons[ci]) / m**2
            newx = self._get_newx(x, deltax)

            if not len(newx):
                self._report_iteration(
                    count, x, obs, cons, valid, level, step, verbosity
                )
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
                                self._report_iteration(
                                    count, x, obs, cons, valid, level, step, verbosity
                                )
                                break

                elif recover:
                    x = newx[0]
                    obs = obsp[0]
                    cons = consp[0]
                    valid = validp[0]

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

                if recover and not anygood:
                    x = hx0
                    obs = obsh0
                    cons = consh0
                    valid = validh0

            self._report_iteration(count, x, obs, cons, valid, level, step, verbosity)

        if verbosity > 0:
            print(f"{hline}")
            print(f"All steps < step_min      : {np.all(step < step_min)}")
            print(f"Objective within tolerance: {done}")
            print(f"{hline}\n")

        # final evaluation:
        pres, obs, cons = self.problem.finalize_individual(inone, x, verbosity)
        valid = self.problem.check_constraints_individual(cons, verbosity)
        if maximize:
            better = obs[0] > obs0
        else:
            better = obs[0] < obs0
        success = bool(
            np.all(valid)
            and (not initially_valid or better or np.abs(obs[0] - obs0) <= self.f_tol)
        )

        results = SingleObjOptResults(
            self.problem,
            success,
            inone,
            x,
            obs,
            cons,
            pres,
        )
        return self._finalize_callbacks(results)
