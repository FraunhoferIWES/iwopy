import numpy as np
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
    vnames_int : list of str, optional
        The integer variable names. Useful for mapping
        problem variables to function variables
    vnames_float : list of str, optional
        The float variable names. Useful for mapping
        problem variables to function variables
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
            vnames_int=None, 
            vnames_float=None, 
            cnames=None
        ):
        super().__init__(name)
        self.problem  = problem
        self._vnamesi = vnames_int
        self._vnamesf = vnames_float
        self._cnames  = cnames

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
                self._cnames = [f"{self.name}_{ci}" for ci in range(self.n_components)]
            else:
                self._cnames = [self.name]

        if self._vnamesi is None:
            self._vnamesi = self.problem.var_names_int()

        if self._vnamesf is None:
            self._vnamesf = self.problem.var_names_float()

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
    
    def pvars2fvars_int(self, pvars, varmap):
        """
        Map problem variables to function variables

        Parameters
        ----------
        pvars : list of str or int
            The problem variables to be mapped
        varmap : dict
            Mapping from function int variables to
            problem int variables. Keys: function name,
            Values: dict with mapping from str (or int)
            to str (or int)
        
        Returns
        -------
        fvars : list of int
            The function variable indices
        
        """
        if not self.name in varmap:
            raise KeyError(f"Function '{self.name}': No match in varmap keys for this function, {sorted(list(varmap.keys()))}")
        vmap = varmap[self.name]
        
        vnms = list(self.var_names_int)
        pnms = list(self.problem.var_names_int())
        geti = lambda v, nms: nms.index(v) if isinstance(v, str) else int(v)
        fvars = []
        for pv in pvars:
            if pv in vmap:
                fv = vmap[pv]
            elif isinstance(pv, str):
                qv = geti(pv, pnms)
                fv = vmap.get(qv, pv)
            else:
                qv = pnms[pv]
                fv = vmap.get(qv, qv)
            fvars.append(geti(fv, vnms))
        
        return fvars

    def pvars2fvars_float(self, pvars, varmap):
        """
        Map problem variables to function variables

        Parameters
        ----------
        pvars : list of str or int
            The problem variables to be mapped
        varmap : dict
            Mapping from function float variables to
            problem float variables. Keys: function name,
            Values: dict with mapping from str (or int)
            to str (or int)
        
        Returns
        -------
        fvars : list of int
            The function variable indices
        
        """
        TODO - STILL NEEDED?
        
        if not self.name in varmap:
            raise KeyError(f"Function '{self.name}': No match in varmap keys for this function, {sorted(list(varmap.keys()))}")
        vmap = varmap[self.name]
        
        vnms = list(self.var_names_float)
        pnms = list(self.problem.var_names_float())
        geti = lambda v, nms: nms.index(v) if isinstance(v, str) else int(v)
        fvars = []
        for pv in pvars:
            if pv in vmap:
                fv = vmap[pv]
            elif isinstance(pv, str):
                qv = geti(pv, pnms)
                fv = vmap.get(qv, pv)
            else:
                qv = pnms[pv]
                fv = vmap.get(qv, qv)
            fvars.append(geti(fv, vnms))
        
        return fvars

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

    def calc_individual(self, vars_int, vars_float, problem_results):
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

        Returns
        -------
        values : np.array
            The component values, shape: (n_components,)

        """
        raise NotImplementedError(f"Not implemented for class {type(self).__name__}")

    def calc_population(self, vars_int, vars_float, problem_results):
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

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_components,)

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

        Otherwise np.nan values are returned instead.

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
