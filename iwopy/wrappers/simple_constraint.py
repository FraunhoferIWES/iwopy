import numpy as np
from abc import abstractmethod

from iwopy.core import Constraint


class SimpleConstraint(Constraint):
    """
    A simple constraint that assumes the
    same variables as defined by the problem.

    :group: wrappers

    """

    def __init__(
        self,
        problem,
        name,
        n_components=1,
        mins=-np.inf,
        maxs=0.0,
        cnames=None,
        has_ana_derivs=True,
    ):
        """
        Constructor

        Parameters
        ----------
        problem: iwopy.Problem
            The underlying optimization problem
        name: str
            The function name
        n_components: int
            The number of components
        mins: float or array
            The minimal values of components,
            shape: (n_components,)
        maxs: float or array
            The maximal values of components,
            shape: (n_components,)
        cnames: list of str, optional
            The names of the components
        has_ana_derivs = bool
            Flag for analytical derivatives

        """
        if cnames is not None and len(cnames) != n_components:
            raise ValueError(
                f"Wrong number of component names, found {len(cnames)}, expected {n_components}: {cnames}"
            )
        self._n_comps = n_components

        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            cnames=cnames,
        )

        self._mins = np.zeros(self._n_comps, dtype=np.float64)
        self._maxs = np.zeros(self._n_comps, dtype=np.float64)
        self._mins[:] = mins
        self._maxs[:] = maxs
        self._ana = has_ana_derivs

    @abstractmethod
    def f(self, *x):
        """
        The function.

        Parameters
        ----------
        x: tuple
            The int and float variables in that order. Variables are
            either scalars or numpy arrays in case of populations.

        Returns
        -------
        result: float (or numpy.ndarray) or list of float (or numpy.ndarray)
            For one component, a float, else a list of floats. For
            population results, a array with shape (n_pop,) in case
            of one component or a list of such arrays otherwise.

        """
        pass

    def g(self, var, *x, components=None):
        """
        The analytical derivative of the function f, df/dvar,
        if available.

        Parameters
        ----------
        var: int
            The index of the derivation varibable within the function
            float variables
        x: tuple
            The int and float variables in that order.
        components: list of int, optional
            The selected components, or None for all

        Returns
        -------
        result: float or list of float
            For one component, a float, else a list of floats.
            The length of list is 0 or 1 in case of single component,
            or n_sel_components otherwise.

        """
        pass

    def get_bounds(self):
        """
        Returns the bounds for all components.

        Non-existing bounds are expressed by np.inf.

        Returns
        -------
        min: np.array
            The lower bounds, shape: (n_components,)
        max: np.array
            The upper bounds, shape: (n_components,)

        """
        return self._mins, self._maxs

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return self._n_comps

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_sel_components,)

        """
        results = np.array(self.f(*vars_int, *vars_float), dtype=np.float64)
        return np.atleast_1d(results)

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_sel_components,)

        """
        varsi = (vars_int[:, vi] for vi in range(self.n_vars_int))
        varsf = (vars_float[:, vi] for vi in range(self.n_vars_float))

        results = self.f(*varsi, *varsf)
        if self.n_components() == 1:
            return results[:, None]
        else:
            return np.stack(results, axis=1)

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        """
        Calculates the analytic derivative, if possible.

        Use `numpy.nan` if analytic derivatives cannot be calculated.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)
        var: int
            The index of the differentiation float variable
        components: list of int
            The selected components, or None for all

        Returns
        -------
        deriv: numpy.ndarray
            The derivative values, shape: (n_sel_components,)

        """
        cmpnts = list(range(self.n_components())) if components is None else components

        if self._ana:
            results = np.atleast_1d(self.g(var, *vars_int, *vars_float, cmpnts))
        else:
            results = None

        if results is None:
            return super().ana_deriv(vars_int, vars_float, var, cmpnts)
        else:
            return np.atleast_1d(np.array(results, dtype=np.float64))
