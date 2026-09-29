from collections.abc import Sequence

import numpy as np

from .function import OptFunction


class OptFunctionSubset(OptFunction):
    """
    A function composed of a subset of a function's
    components.
    """

    def __init__(
        self,
        function: OptFunction,
        subset: list[int] | np.ndarray,
        name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        function
            The original function
        subset
            The component choice
        name
            The function name
        """
        if name is None:
            name = f"{function.name}[" + ",".join([str(i) for i in subset]) + "]"
        super().__init__(function.problem, name)

        self.func_org = function
        self.subset = subset

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """
        f = self.func_org
        if not f.initialized:
            f.initialize(verbosity)

        self._cnames = [f.component_names[i] for i in self.subset]
        self._vdepsi: np.ndarray = f.vardeps_int()[self.subset]
        self._vdepsf: np.ndarray = f.vardeps_float()[self.subset]
        self._vnamesi = [f.var_names_int[i] for i in np.unique(self._vdepsi)]
        self._vnamesf = [f.var_names_float[i] for i in np.unique(self._vdepsf)]

        super().initialize(verbosity)

    def vardeps_int(self) -> np.ndarray:
        """
        Gets the dependencies of all components
        on the function int variables

        Returns
        -------
        deps
            The dependencies of components on function
            variables, shape: (n_components, n_vars_int)
        """
        return self._vdepsi

    def vardeps_float(self) -> np.ndarray:
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps
            The dependencies of components on function
            variables, shape: (n_components, n_vars_float)
        """
        return self._vdepsf

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        n_components
            The number of components.
        """
        return len(self.subset)

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: object,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)
        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.calc_individual(
            vars_int, vars_float, problem_results, cmpts
        )

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: object,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components,)
        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.calc_population(
            vars_int, vars_float, problem_results, cmpts
        )

    def ana_deriv(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        var: int,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculates the analytic derivative, if possible.

        Use `numpy.nan` if analytic derivatives cannot be calculated.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        var
            The index of the differentiation float variable
        components
            The selected components, or None for all

        Returns
        -------
        deriv
            The derivative values, shape: (n_sel_components,)
        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.ana_deriv(vars_int, vars_float, var, cmpts)
