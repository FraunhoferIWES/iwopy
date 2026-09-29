from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from iwopy import SimpleObjective, SimpleProblem
from iwopy.core import Problem


class BraninObjective(SimpleObjective):
    """
    The objective function for the Branin problem.

    The Branin (or Branin-Hoo) function is defined as

    f(x,y) = a(y-bx^2+cx-r)^2 + s(1-t)cos(x)+s

    Recommended values for the parameters are:
    a = 1
    b = 5.1/(4*pi^2)
    c = 5/pi
    r = 6
    s = 10
    t = 1/(8*pi)

    Domain:
    x = [-5, 10]
    y = [0, 15]

    The Branin function has three global minima at

    (x,y) = (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)

    with a function value of

    f(x,y) = 0.397887
    """

    def __init__(
        self, problem: Problem, ana_deriv: bool = False, name: str = "f"
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem
        ana_deriv
            Switch for analytical derivatives
        name
            The function name
        """
        super().__init__(problem, name, n_components=1, has_ana_derivs=ana_deriv)

        # (a, b, c, r, s, t)
        self._pars: tuple[float, float, float, float, float, float] = (
            1,
            5.1 / (4 * np.pi**2),
            5 / np.pi,
            6,
            10,
            1 / (8 * np.pi),
        )

        self._ana_deriv = ana_deriv

    def f(self, *x: ArrayLike) -> ArrayLike:
        """The Branin function f(x, y)"""
        x_value, y_value = (np.asarray(value) for value in x)
        a, b, c, r, s, t = self._pars
        return (
            a * (y_value - b * x_value**2 + c * x_value - r) ** 2
            + s * (1 - t) * np.cos(x_value)
            + s
        )

    def g(
        self,
        var: int,
        *x: ArrayLike,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> ArrayLike:
        """The derivative of the Branin function"""
        del components
        x_value, y_value = (np.asarray(value) for value in x)
        a, b, c, r, s, t = self._pars
        if var == 0:
            return 2 * a * (y_value - b * x_value**2 + c * x_value - r) * (
                -2 * b * x_value + c
            ) - s * (1 - t) * np.sin(x_value)
        else:
            return 2 * a * (y_value - b * x_value**2 + c * x_value - r)


class BraninProblem(SimpleProblem):
    """Problem definition of benchmark function Branin."""

    def __init__(
        self,
        name: str = "branin",
        initial_values: Sequence[float] | np.ndarray | None = None,
        ana_deriv: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name
            The name of the problem
        ana_deriv
            Switch for analytical derivatives
        initial_values
            The initial values
        """
        if initial_values is None:
            initial_values = [1.0, 1.0]
        super().__init__(
            name,
            float_vars={"x": initial_values[0], "y": initial_values[1]},
            min_values_float={"x": -5.0, "y": 0.0},
            max_values_float={"x": 10.0, "y": 15},
        )

        self.add_objective(BraninObjective(self, ana_deriv=ana_deriv))
