import fnmatch
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .base import Base

if TYPE_CHECKING:
    from .problem import Problem


class OptFunction(Base, metaclass=ABCMeta):
    """
    Abstract base class for functions
    that calculate scalars based on a problem.
    """

    def __init__(
        self,
        problem: "Problem",
        name: str,
        n_vars_int: int | None = None,
        n_vars_float: int | None = None,
        vnames_int: list[str] | None = None,
        vnames_float: list[str] | None = None,
        cnames: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The function name
        n_vars_int
            The number of integer variables. If not specified
            it is assumed that the function depends on all
            problem int variables
        n_vars_float
            The number of float variables. If not specified
            it is assumed that the function depends on all
            problem float variables
        vnames_int
            The integer variable names. Useful for mapping
            function variables to problem variables, otherwise
            map by integer or default name
        vnames_float
            The float variable names. Useful for mapping
            function variables to problem variables, otherwise
            map by integer or default name
        cnames
            The names of the components
        """
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
    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        n_components
            The number of components.
        """

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
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
    def component_names(self) -> list[str]:
        """
        The names of the components

        Returns
        -------
        names
            The component names
        """
        if self._cnames is None:
            raise RuntimeError(f"Function '{self.name}' has not been initialized")
        return self._cnames

    @property
    def var_names_int(self) -> list[str]:
        """
        The names of the integer variables

        Returns
        -------
        names
            The integer variable names
        """
        if self._vnamesi is None:
            raise RuntimeError(f"Function '{self.name}' has not been initialized")
        return self._vnamesi

    @property
    def n_vars_int(self) -> int:
        """
        The number of int variables

        Returns
        -------
        n
            The number of int variables
        """
        return len(self.var_names_int)

    @property
    def var_names_float(self) -> list[str]:
        """
        The names of the float variables

        Returns
        -------
        names
            The float variable names
        """
        if self._vnamesf is None:
            raise RuntimeError(f"Function '{self.name}' has not been initialized")
        return self._vnamesf

    @property
    def n_vars_float(self) -> int:
        """
        The number of float variables

        Returns
        -------
        n
            The number of float variables
        """
        return len(self.var_names_float)

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
        return np.ones((self.n_components(), self.n_vars_int), dtype=bool)

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
        return np.ones((self.n_components(), self.n_vars_float), dtype=bool)

    def _rename_vars(
        self,
        varmap: Mapping[str | int, str],
        target: list[str],
        vtype: str,
    ) -> None:
        """Helper function for variable renaming"""
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

    def rename_vars_int(self, varmap: Mapping[str | int, str]) -> None:
        """
        Rename integer variables.

        Parameters
        ----------
        varmap
            The name mapping. Key: old name str,
            Value: new name str
        """
        self._rename_vars(varmap, self.var_names_int, "int")

    def rename_vars_float(self, varmap: Mapping[str | int, str]) -> None:
        """
        Rename float variables.

        Parameters
        ----------
        varmap
            The name mapping. Key: old name str,
            Value: new name str
        """
        self._rename_vars(varmap, self.var_names_float, "float")

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
        raise NotImplementedError(f"Not implemented for class {type(self).__name__}")

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

    def finalize_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: object,
        verbosity: int = 1,
    ) -> np.ndarray:
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float
            The optimal float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The component values, shape: (n_components,)
        """
        return self.calc_individual(vars_int, vars_float, problem_results)

    def finalize_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: object,
        verbosity: int = 1,
    ) -> np.ndarray:
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The component values, shape: (n_pop, n_components)
        """
        return self.calc_population(vars_int, vars_float, problem_results)

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
        n_cmpnts = len(components) if components is not None else self.n_components()
        return np.full(n_cmpnts, np.nan, dtype=np.float64)
