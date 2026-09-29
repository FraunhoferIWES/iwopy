from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from .function import OptFunction

if TYPE_CHECKING:
    from .problem import Problem


_OptFunctionT = TypeVar("_OptFunctionT", bound=OptFunction)


class OptFunctionList(OptFunction, Generic[_OptFunctionT]):
    """
    A list of functions.

    The main point of this class is to manage the
    variables and components of the added functions.

    Add functions to the list via the `append` function,
    and don't forget to initialize.

    Attributes
    ----------
    func_vars_int: list of lists of int
        For each added function, the subset of
        integer variables
    func_vars_float: list of lists of int
        For each added function, the subset of
        float variables
    sizes: list of int
        The components of each added function

    :group: core

    """

    def __init__(self, problem: "Problem", name: str) -> None:
        """
        Constructor

        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The function name

        """
        super().__init__(problem, name)

        self._functions: list[_OptFunctionT] = []
        self._cnames = []
        self.func_vars_int: list[list[int]] = []
        self.func_vars_float: list[list[int]] = []
        self.sizes: list[int] = []

    def append(self, function: _OptFunctionT) -> None:
        """
        Adds a function to the list.

        Parameters
        ----------
        function
            The function

        """
        if self.initialized:
            raise ValueError(
                f"FunctionList '{self.name}': Attempt to add function '{function.name}' after initialization"
            )

        if function.problem is not self.problem:
            raise ValueError(
                f"FunctionList '{self.name}': Cannot add function '{function.name}' since problems don't match. Expected '{self.problem.name}', found '{function.problem.name}'"
            )
        self._functions.append(function)
        self.component_names.extend(function.component_names)

    @property
    def functions(self) -> list[_OptFunctionT]:
        """
        The list of added funtions

        Returns
        -------
        funcs: list of iwopy.core.OptFunction
            The list of added functions

        """
        return self._functions

    @property
    def n_functions(self) -> int:
        """
        The number of added functions

        Returns
        -------
        n: int
            The total number of added functions

        """
        return len(self.functions)

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        self._vnamesi = []
        self._vnamesf = []
        self.sizes = []
        for f in self.functions:
            if not f.initialized:
                f.initialize(verbosity)
            self._vnamesi += f.var_names_int
            self._vnamesf += f.var_names_float
            self.sizes.append(f.n_components())
        self._vnamesi = list(dict.fromkeys(self._vnamesi))
        self._vnamesf = list(dict.fromkeys(self._vnamesf))

        def getv(vnames: list[str], fvnames: list[str]) -> list[int]:
            if not len(fvnames):
                return []
            return [vnames.index(v) for v in fvnames]

        self.func_vars_int = [
            getv(self._vnamesi, f.var_names_int) for f in self.functions
        ]
        self.func_vars_float = [
            getv(self._vnamesf, f.var_names_float) for f in self.functions
        ]

        super().initialize(verbosity)

    def vardeps_int(self) -> np.ndarray:
        """
        Gets the dependencies of all components
        on the function int variables

        Returns
        -------
        deps: numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_int)

        """
        deps = np.zeros((self.n_components(), self.n_vars_int), dtype=bool)

        i0 = 0
        for f in self.functions:
            i1 = i0 + f.n_components()
            deps[i0:i1] = f.vardeps_int()
            i0 = i1

        return deps

    def vardeps_float(self) -> np.ndarray:
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps: numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_float)

        """
        deps = np.zeros((self.n_components(), self.n_vars_float), dtype=bool)

        i0 = 0
        for f in self.functions:
            i1 = i0 + f.n_components()
            deps[i0:i1] = f.vardeps_float()
            i0 = i1

        return deps

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return sum(self.sizes)

    def split_individual(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Splits result values or other data into
        individual function data.

        Parameters
        ----------
        data
            The data, shape: (n_components,)

        Returns
        -------
        fdata: list of numpy.ndarray
            The data for each function, list entry
            shapes: (n_func_components,)

        """
        out = []
        i0 = 0
        for fi in range(self.n_functions):
            i1 = i0 + self.sizes[fi]
            out.append(data[i0:i1])
            i0 = i1
        return out

    def split_population(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Splits result values or other data into
        individual function data.

        Parameters
        ----------
        data
            The data, shape: (n_pop, n_components)

        Returns
        -------
        fdata: list of numpy.ndarray
            The data for each function, list entry
            shapes: (n_pop, n_func_components)

        """
        out = []
        i0 = 0
        for fi in range(self.n_functions):
            i1 = i0 + self.sizes[fi]
            out.append(data[:, i0:i1])
            i0 = i1
        return out

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
        values: np.array
            The component values, shape: (n_sel_components,)

        """
        cmpnts = np.arange(self.n_components()) if components is None else components
        values = np.full(len(cmpnts), np.nan, dtype=np.float64)

        i0 = 0
        j0 = 0
        for fi, f in enumerate(self.functions):
            i1 = i0 + self.sizes[fi]
            cts = (
                None
                if components is None
                else [i - i0 for i in cmpnts if i >= i0 and i < i1]
            )
            if cts is None or len(cts):
                j1 = j0 + (self.sizes[fi] if cts is None else len(cts))
                varsi = vars_int[self.func_vars_int[fi]]
                varsf = vars_float[self.func_vars_float[fi]]
                values[j0:j1] = f.calc_individual(varsi, varsf, problem_results, cts)
                j0 = j1
            i0 = i1

        return values

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
        values: np.array
            The component values, shape: (n_pop, n_sel_components,)

        """
        n_pop = vars_float.shape[0]
        cmpnts = np.arange(self.n_components()) if components is None else components
        values = np.full((n_pop, len(cmpnts)), np.nan, dtype=np.float64)

        i0 = 0
        j0 = 0
        for fi, f in enumerate(self.functions):
            i1 = i0 + self.sizes[fi]
            cts = (
                None
                if components is None
                else [i - i0 for i in cmpnts if i >= i0 and i < i1]
            )
            if cts is None or len(cts):
                j1 = j0 + (self.sizes[fi] if cts is None else len(cts))
                varsi = vars_int[:, self.func_vars_int[fi]]
                varsf = vars_float[:, self.func_vars_float[fi]]
                values[:, j0:j1] = f.calc_population(varsi, varsf, problem_results, cts)
                j0 = j1
            i0 = i1

        return values

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
        values: np.array
            The component values, shape: (n_components,)

        """
        values = np.full(self.n_components(), np.nan, dtype=np.float64)

        i0 = 0
        for fi, f in enumerate(self.functions):
            i1 = i0 + self.sizes[fi]
            varsi = vars_int[self.func_vars_int[fi]]
            varsf = vars_float[self.func_vars_float[fi]]
            values[i0:i1] = f.finalize_individual(
                varsi, varsf, problem_results, verbosity
            )
            i0 = i1

        return values

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
        values: np.array
            The component values, shape: (n_pop, n_components)

        """
        n_pop = vars_float.shape[0]
        values = np.full((n_pop, self.n_components()), np.nan, dtype=np.float64)

        i0 = 0
        for fi, f in enumerate(self.functions):
            i1 = i0 + self.sizes[fi]
            varsi = vars_int[:, self.func_vars_int[fi]]
            varsf = vars_float[:, self.func_vars_float[fi]]
            values[:, i0:i1] = f.finalize_population(
                varsi, varsf, problem_results, verbosity
            )
            i0 = i1

        return values

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
        deriv: numpy.ndarray
            The derivative values, shape: (n_sel_components,)

        """
        cmpnts = np.arange(self.n_components()) if components is None else components
        deriv = np.full(len(cmpnts), np.nan, dtype=np.float64)

        i0 = 0
        j0 = 0
        for fi, f in enumerate(self.functions):
            i1 = i0 + self.sizes[fi]
            if var in list(self.func_vars_float[fi]):
                cts = (
                    None
                    if components is None
                    else [i - i0 for i in cmpnts if i >= i0 and i < i1]
                )
                if cts is None or len(cts):
                    j1 = j0 + (self.sizes[fi] if cts is None else len(cts))
                    varsi = vars_int[self.func_vars_int[fi]]
                    varsf = vars_float[self.func_vars_float[fi]]
                    vi = list(self.func_vars_float[fi]).index(var)
                    deriv[j0:j1] = f.ana_deriv(varsi, varsf, vi, components=cts)
                    j0 = j1
            i0 = i1

        return deriv
