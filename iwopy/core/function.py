import numpy as np
import fnmatch
from abc import ABCMeta, abstractmethod

from .base import Base


class OptFunction(Base, metaclass=ABCMeta):
    """
    Abstract base class for functions
    that calculate scalars based on a problem.

    Parameters
    ----------
    problem: iwopy.Problem
        The underlying optimization problem
    name: str
        The function name
    n_vars_int : int, optional
        The number of integer variables. If not specified
        it is assumed that the function depends on all
        problem int variables
    n_vars_float : int, optional
        The number of float variables. If not specified
        it is assumed that the function depends on all
        problem float variables
    vnames_int : list of str, optional
        The integer variable names. Useful for mapping
        function variables to problem variables, otherwise
        map by integer or default name
    vnames_float : list of str, optional
        The float variable names. Useful for mapping
        function variables to problem variables, otherwise
        map by integer or default name
    cnames : list of str, optional
        The names of the components

    Attributes
    ----------
    problem: iwopy.Problem
        The underlying optimization problem

    """

    def __init__(
        self,
        problem,
        name,
        n_vars_int=None,
        n_vars_float=None,
        vnames_int=None,
        vnames_float=None,
        cnames=None,
    ):
        super().__init__(name)

        self.problem = problem
        self._vnamesi = vnames_int
        self._vnamesf = vnames_float
        self._cnames = cnames

        if n_vars_int is not None:
            if vnames_int is not None:
                if len(vnames_int) != n_vars_int:
                    raise ValueError(
                        f"Problem '{self.name}': Mismatch between n_vars_int = {n_vars_int} and vnames_int = {vnames_int}, length {len(vnames_int)}"
                    )
            else:
                self._vnamesi = [f"{name}_n{i}" for i in range(n_vars_int)]

        if n_vars_float is not None:
            if vnames_float is not None:
                if len(vnames_float) != n_vars_float:
                    raise ValueError(
                        f"Problem '{self.name}': Mismatch between n_vars_float = {n_vars_float} and vnames_float = {vnames_float}, length {len(vnames_float)}"
                    )
            else:
                self._vnamesf = [f"{name}_x{i}" for i in range(n_vars_float)]

    @abstractmethod
    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        pass

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if self._cnames is None:
            if self.n_components() > 1:
                self._cnames = [
                    f"{self.name}_{ci}" for ci in range(self.n_components())
                ]
            else:
                self._cnames = [self.name]

        if self._vnamesi is None:
            self._vnamesi = list(self.problem.var_names_int())

        if self._vnamesf is None:
            self._vnamesf = list(self.problem.var_names_float())

        super().initialize(verbosity)

    @property
    def component_names(self):
        """
        The names of the components

        Returns
        -------
        names : list of str
            The component names

        """
        return self._cnames

    @property
    def var_names_int(self):
        """
        The names of the integer variables

        Returns
        -------
        names : list of str
            The integer variable names

        """
        return self._vnamesi

    @property
    def n_vars_int(self):
        """
        The number of int variables

        Returns
        -------
        n : int
            The number of int variables

        """
        return len(self.var_names_int)

    @property
    def var_names_float(self):
        """
        The names of the float variables

        Returns
        -------
        names : list of str
            The float variable names

        """
        return self._vnamesf

    @property
    def n_vars_float(self):
        """
        The number of float variables

        Returns
        -------
        n : int
            The number of float variables

        """
        return len(self.var_names_float)

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
        return np.ones((self.n_components(), self.n_vars_int), dtype=bool)

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
        return np.ones((self.n_components(), self.n_vars_float), dtype=bool)

    def _rename_vars(self, varmap, target, vtype):
        """
        Helper function for variable renaming
        """
        for ov, nv in varmap.items():
            if isinstance(ov, str):
                ovl = fnmatch.filter(target, ov)
                if not len(ovl):
                    raise KeyError(
                        f"Function '{self.name}': Cannot apply renaming '{ov} --> {nv}', since '{ov}' not found in {vtype} variables {target}"
                    )
                elif len(ovl) > 1:
                    raise KeyError(
                        f"Function '{self.name}': Cannot apply renaming '{ov} --> {nv}', since more than one match found in {vtype} variables: {ovl}"
                    )
                oi = target.index(ovl[0])
            elif isinstance(ov, int):
                oi = ov
                if oi < 0 or oi >= len(target):
                    raise ValueError(
                        f"Function '{self.name}': Renaming rule '{ov} --> {nv}' cannot be applied for {len(target)} {vtype} variables {target}"
                    )
            else:
                raise TypeError(
                    f"Function '{self.name}': Unacceptable source type '{type(ov)}' in renaming rule '{ov} --> {nv}', expecting str or int"
                )
            if not isinstance(nv, str):
                raise TypeError(
                    f"Function '{self.name}': Unacceptable target type '{type(nv)}' in renaming rule '{ov} --> {nv}', expecting str"
                )
            target[oi] = nv

    def rename_vars_int(self, varmap):
        """
        Rename integer variables.

        Parameters
        ----------
        varmap : dict
            The name mapping. Key: old name str,
            Value: new name str

        """
        self._rename_vars(varmap, self._vnamesi, "int")

    def rename_vars_float(self, varmap):
        """
        Rename float variables.

        Parameters
        ----------
        varmap : dict
            The name mapping. Key: old name str,
            Value: new name str

        """
        self._rename_vars(varmap, self._vnamesf, "float")

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
        raise NotImplementedError(f"Not implemented for class {type(self).__name__}")

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
            The component values, shape: (n_pop, n_sel_components)

        """
        if problem_results is not None:
            raise NotImplementedError(
                f"Not implemented for class {type(self).__name__}, results type {type(problem_results).__name__}"
            )

        # prepare:
        n_pop = (
            vars_float.shape[0]
            if vars_float is not None and len(vars_float.shape)
            else vars_int.shape[0]
        )
        vals = np.full((n_pop, self.n_components()), np.nan, dtype=np.float64)

        # loop over individuals:
        for i in range(n_pop):
            vals[i] = self.calc_individual(vars_int[i], vars_float[i], None)

        return vals

    def finalize_individual(self, vars_int, vars_float, problem_results, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int : np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The optimal float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        values : np.array
            The component values, shape: (n_components,)

        """
        return self.calc_individual(vars_int, vars_float, problem_results)

    def finalize_population(self, vars_int, vars_float, problem_results, verbosity=1):
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_components)

        """
        return self.calc_population(vars_int, vars_float, problem_results)

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
        n_cmpnts = len(components) if components is not None else self.n_components()
        return np.full(n_cmpnts, np.nan, dtype=np.float64)
