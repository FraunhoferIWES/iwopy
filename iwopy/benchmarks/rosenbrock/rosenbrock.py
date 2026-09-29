from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from iwopy import SimpleObjective, SimpleProblem
from iwopy.core import Problem


class RosenbrockObjective(SimpleObjective):
    """
    The Rosenbrock function is defined as

    f(x,y) = (a-x)^2 + b(y-x^2)^2

    Recommended values for the parameters are:
    a = 1
    b = 100

    Domain:
    x = [-inf, inf]
    y = [-inf, inf]

    The unconstraint Rosenbrock function has a global minima at

    (x,y) = (1,1)

    with a function value of

    f(x,y) = 0

    :group: benchmarks.rosenbrock

    """

    def __init__(
        self,
        problem: Problem,
        pars: tuple[float, float] = (1.0, 100.0),
        ana_deriv: bool = False,
        name: str = "f",
    ) -> None:
        """
        Construtor

        Parameters
        ----------
        problem
            The underlying optimization problem
        pars
            The a, b parameters
        ana_deriv
            Switch for analytical derivatives
        name
            The function name

        """
        super().__init__(problem, name, n_components=1, has_ana_derivs=ana_deriv)

        # (a, b)
        self._pars = pars

    def f(self, *x: ArrayLike) -> ArrayLike:
        """
        The Rosenbrock function f(x, y)
        """
        x_value, y_value = (np.asarray(value) for value in x)
        a, b = self._pars
        return (a - x_value) ** 2 + b * (y_value - x_value**2) ** 2

    def g(
        self,
        var: int,
        *x: ArrayLike,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> ArrayLike:
        """
        The derivative of the Rosenbrock function
        """
        del components
        x_value, y_value = (np.asarray(value) for value in x)
        a, b = self._pars
        if var == 0:
            return -2 * (a - x_value) - 2 * b * (y_value - x_value**2) * 2 * x_value
        else:
            return 2 * b * (y_value - x_value**2)


class RosenbrockProblem(SimpleProblem):
    """
    Problem definition of benchmark function Rosenbrock.

    Attributes
    ----------
    initial_values: list of float
        The initial values

    :group: benchmarks.rosenbrock

    """

    def __init__(
        self,
        lower: Sequence[float] | np.ndarray | None = None,
        upper: Sequence[float] | np.ndarray | None = None,
        initial: Sequence[float] | np.ndarray | None = None,
        ana_deriv: bool = False,
        name: str = "rosenbrock",
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        lower
            The minimal variable values
        upper
            The maximal variable values
        initial
            The initial values
        ana_deriv
            Switch for analytical derivatives
        name
            The name of the problem

        """
        if initial is None:
            initial = [0.0, 0.0]
        if upper is None:
            upper = [10.0, 10.0]
        if lower is None:
            lower = [-5.0, -5.0]
        super().__init__(
            name,
            float_vars={"x": initial[0], "y": initial[1]},
            min_values_float={"x": lower[0], "y": lower[1]},
            max_values_float={"x": upper[0], "y": upper[1]},
        )

        self.add_objective(RosenbrockObjective(self, ana_deriv=ana_deriv))
