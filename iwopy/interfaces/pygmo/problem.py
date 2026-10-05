from typing import Any, Protocol

import numpy as np

from iwopy.core import (
    OptFunction,
    OptFunctionList,
    OptFunctionSubset,
    Problem,
    SingleObjOptResults,
)


class _CallbackSink(Protocol):
    def notify(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        objs: np.ndarray,
        cons: np.ndarray,
    ) -> None: ...


class UDP:
    """Adapt an iwopy problem to a PyGMO user-defined problem.

    PyGMO's component tolerance vector is derived from the tolerance owned by
    each registered iwopy constraint when the adapter is constructed.
    """

    def __init__(
        self,
        problem: Problem,
        pop: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Initialize the PyGMO problem adapter.

        Parameters
        ----------
        problem
            The problem to optimize
        pop
            Vectorized fitness computation
        verbosity
            The verbosity level, 0 = silent
        """
        self.problem = problem
        self.n_vars_all = problem.n_vars_float + problem.n_vars_int
        self.n_fitness = problem.n_objectives + problem.n_constraints

        self.c_tol = (
            np.concatenate(
                [
                    np.full(
                        constraint.n_components(),
                        constraint.tol,
                        dtype=np.float64,
                    )
                    for constraint in problem.cons.functions
                ]
            )
            if problem.n_constraints
            else np.empty(0, dtype=np.float64)
        )

        self.pop = pop
        self.verbosity = verbosity
        self.callback_sink: _CallbackSink | None = None
        self._active = False

    def fitness(self, dv: np.ndarray) -> np.ndarray:
        """
        Calculate fitness values for one decision vector.

        Parameters
        ----------
        dv
            The decision vector

        Returns
        -------
        fitness_values
            The objective and constraint values
        """
        # extract variables:
        xf = dv[: self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float :].astype(np.int32)

        # apply new variables:
        values = np.zeros(self.n_fitness, dtype=np.float64)
        objs, cons = self.problem.evaluate_individual(xi, xf)
        if self.callback_sink is not None:
            self.callback_sink.notify(xi, xf, objs, cons)
        objs *= np.where(self.problem.maximize_objs, -1.0, 1.0)
        values[: self.problem.n_objectives] = objs
        values[self.problem.n_objectives :] = cons

        return values

    def batch_fitness(self, dvs: np.ndarray) -> np.ndarray:
        """
        Calculate fitness values for a batch of decision vectors.

        Parameters
        ----------
        dvs
            The flattened decision vectors

        Returns
        -------
        fitness_values
            The flattened objective and constraint values
        """
        # extract variables:
        n_vf = self.problem.n_vars_float
        n_vi = self.problem.n_vars_int
        n_v = n_vi + n_vf
        n_pop = int(len(dvs) / n_v)
        dvs = dvs.reshape(n_pop, n_v)
        xf = dvs[:, :n_vf]
        xi = dvs[:, n_vf:].astype(np.int32)

        # apply new variables:
        values = np.zeros((n_pop, self.n_fitness), dtype=np.float64)
        objs, cons = self.problem.evaluate_population(xi, xf)
        if self.callback_sink is not None:
            self.callback_sink.notify(xi, xf, objs, cons)
        objs *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
        values[:, : self.problem.n_objectives] = objs
        values[:, self.problem.n_objectives :] = cons

        return values.reshape(n_pop * self.n_fitness)

    def has_batch_fitness(self) -> bool:
        """Check whether batch fitness evaluation is enabled."""
        return self.pop

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the decision-variable bounds.

        Returns
        -------
        lower_bounds
            The lower decision-variable bounds
        upper_bounds
            The upper decision-variable bounds
        """
        lb = np.full(self.n_vars_all, -np.inf)
        ub = np.full(self.n_vars_all, np.inf)

        if self.problem.n_vars_float:
            lb[: self.problem.n_vars_float] = self.problem.min_values_float()
            ub[: self.problem.n_vars_float] = self.problem.max_values_float()

        if self.problem.n_vars_int:
            lbi = lb[self.problem.n_vars_float :]
            ubi = ub[self.problem.n_vars_float :]

            lbi[:] = self.problem.min_values_int()
            ubi[:] = self.problem.max_values_int()

            lbi[lbi == -Problem.INT_INF] = -np.inf
            ubi[ubi == Problem.INT_INF] = np.inf

        return (lb, ub)

    def get_nobj(self) -> int:
        """Get the number of objectives."""
        return self.problem.n_objectives

    def get_nec(self) -> int:
        """Get the number of equality constraints."""
        return 0

    def get_nic(self) -> int:
        """Get the number of inequality constraints."""
        return self.problem.n_constraints

    def get_nix(self) -> int:
        """Get the number of integer decision variables."""
        return self.problem.n_vars_int

    def has_gradient(self) -> bool:
        """Check whether gradient evaluation is available."""
        return True

    def gradient(self, x: np.ndarray) -> list[float]:
        """
        Calculate the flattened fitness gradient.

        Parameters
        ----------
        x
            The decision vector

        Returns
        -------
        gradient
            The gradient entries selected by the sparsity pattern
        """
        sparsity = self.gradient_sparsity()
        if not sparsity:
            return []

        spars = np.array(sparsity, dtype=np.int32)
        cmpnts = np.unique(spars[:, 0])
        vrs = np.unique(spars[:, 1])

        if len(cmpnts) != self.problem.n_objectives + self.problem.n_constraints:
            function_list: OptFunctionList[OptFunction] = OptFunctionList(
                self.problem, "objs_cons"
            )
            for objective in self.problem.objs.functions:
                function_list.append(objective)
            for constraint in self.problem.cons.functions:
                function_list.append(constraint)
            subset = OptFunctionSubset(function_list, cmpnts)
            subset.initialize()
            func: OptFunction | None = subset
        else:
            func = None

        varsf = x[: self.problem.n_vars_float]
        varsi = x[self.problem.n_vars_float :].astype(np.int32)

        grad = self.problem.get_gradients(
            varsi,
            varsf,
            vars=vrs,
            func=func,
            verbosity=self.verbosity,
            pop=self.pop,
        )

        component_rows = {component: row for row, component in enumerate(cmpnts)}
        objective_signs = np.where(self.problem.maximize_objs, -1.0, 1.0)
        return [
            float(
                grad[component_rows[c], list(vrs).index(v)]
                * (objective_signs[c] if c < self.problem.n_objectives else 1.0)
            )
            for c, v in spars
        ]

    def has_gradient_sparsity(self) -> bool:
        """Check whether a gradient sparsity pattern is available."""
        return True

    def gradient_sparsity(self) -> list[list[int]]:
        """
        Get the fitness-gradient sparsity pattern.

        Returns
        -------
        sparsity
            The fitness-component and variable index pairs
        """
        out: list[list[int]] = []

        # add sparsity of objectives:
        out += np.argwhere(self.problem.objs.vardeps_float()).tolist()
        if self.problem.n_vars_int:
            depsi = np.argwhere(self.problem.objs.vardeps_int())
            depsi[:, 1] += self.problem.n_vars_float
            out += depsi.tolist()

        # add sparsity of constraints:
        if self.problem.n_constraints:
            depsf = np.argwhere(self.problem.cons.vardeps_float())
            depsf[:, 0] += self.problem.n_objectives
            out += depsf.tolist()
            if self.problem.n_vars_int:
                depsi = np.argwhere(self.problem.cons.vardeps_int())
                depsi[:, 0] += self.problem.n_objectives
                depsi[:, 1] += self.problem.n_vars_float
                out += depsi.tolist()

        return sorted(out)

    def has_hessians(self) -> bool:
        """Check whether Hessian evaluation is available."""
        return False

    # def hessians(self, dv):

    def has_hessians_sparsity(self) -> bool:
        """Check whether Hessian sparsity patterns are available."""
        return False

    # def hessians_sparsity(self):

    def has_set_seed(self) -> bool:
        """Check whether random-seed configuration is available."""
        return False

    # def set_seed(self, s):

    def get_name(self) -> str:
        """Get the problem name."""
        return self.problem.name

    def get_extra_info(self) -> str:
        """Get additional problem information."""
        return ""

    def finalize(self, pygmo_pop: Any, verbosity: int = 1) -> SingleObjOptResults:
        """
        Finalize the problem.

        Parameters
        ----------
        pygmo_pop
            The results from the solver
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        results
            The optimization results object
        """

        # extract variables:
        dv = pygmo_pop.champion_x
        xf = dv[: self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float :].astype(np.int32)

        if verbosity:
            print()
            print(pygmo_pop)

        # apply final variables:
        res, objs, cons = self.problem.finalize_individual(xi, xf, verbosity)

        if verbosity:
            print()
        suc = np.all(self.problem.check_constraints_individual(cons, verbosity))
        if verbosity:
            print()

        return SingleObjOptResults(self.problem, suc, xi, xf, objs, cons, res)
