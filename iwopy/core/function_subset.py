import numpy as np

from .function import OptFunction


class OptFunctionSubset(OptFunction):
    """
    A function composed of a subset of a function's
    components.

    Parameters
    ----------
    function: iwopy.OptFunction
        The original function
    subset : list of int
        The component choice
    name: str, optional
        The function name

    Attributes
    ----------
    func_org: iwopy.OptFunction
        The original function
    subset : list of int
        The component choice

    """

    def __init__(self, function, subset, name=None):

        if name is None:
            name = f"{function.name}[" + ",".join([str(i) for i in subset]) + "]"
        super().__init__(function.problem, name)

        self.func_org = function
        self.subset = subset

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        f = self.func_org
        if not f.initialized:
            f.initialize(verbosity)

        self._cnames = [f._cnames[i] for i in self.subset]
        self._vdepsi = f.vardeps_int()[self.subset]
        self._vdepsf = f.vardeps_float()[self.subset]
        self._vnamesi = [f._vnamesi[i] for i in np.unique(self._vdepsi)]
        self._vnamesf = [f._vnamesf[i] for i in np.unique(self._vdepsf)]

        super().initialize(verbosity)

    def vardeps_int(self):
        """
        Gets the dependencies of all components
        on the function int variables

        Returns
        -------
        deps : numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_int)

        """
        return self._vdepsi

    def vardeps_float(self):
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps : numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_float)

        """
        return self._vdepsf

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return len(self.subset)

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.calc_individual(
            vars_int, vars_float, problem_results, cmpts
        )

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components,)

        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.calc_population(
            vars_int, vars_float, problem_results, cmpts
        )

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        """
        Calculates the analytic derivative, if possible.

        Use `numpy.nan` if analytic derivatives cannot be calculated.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        var : int
            The index of the differentiation float variable
        components : list of int
            The selected components, or None for all

        Returns
        -------
        deriv : numpy.ndarray
            The derivative values, shape: (n_sel_components,)

        """
        cmpts = (
            self.subset if components is None else [self.subset[i] for i in components]
        )
        return self.func_org.ana_deriv(vars_int, vars_float, var, cmpts)
