import numpy as np

from iwopy import SimpleProblem, SimpleObjective


class BraninObjective(SimpleObjective):
    """
    The objective function for the Branin problem.

    The Branin, or Branin-Hoo function is defined as

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

    Parameters
    ----------
    problem: iwopy.Problem
        The underlying optimization problem
    ana_deriv : bool
        Switch for analytical derivatives
    name: str
        The function name

    """

    def __init__(self, problem, ana_deriv=False, name="f"):
        super().__init__(problem, name, n_components=1, has_ana_derivs=ana_deriv)

        # Parameters of branin function,
        # (a, b, c, r, s, t)
        self._pars = (
            1,
            5.1 / (4 * np.pi**2),
            5 / np.pi,
            6,
            10,
            1 / (8 * np.pi),
        )

        self._ana_deriv = ana_deriv

    def f(self, x, y):
        """
        The Branin function f(x, y)
        """
        a, b, c, r, s, t = self._pars
        return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s

    def g(self, var, x, y, components=None):
        """
        The derivative of the Branin function
        """
        a, b, c, r, s, t = self._pars
        if var == 0:
            return 2 * a * (y - b * x**2 + c * x - r) * (-2 * b * x + c) - s * (
                1 - t
            ) * np.sin(x)
        else:
            return 2 * a * (y - b * x**2 + c * x - r)


class BraninProblem(SimpleProblem):
    """
    Problem definition of benchmark function Branin.

    Parameters
    ----------
    name : str
        The name of the problem
    ana_deriv : bool
        Switch for analytical derivatives
    initial_values : list of float
        The initial values

    Attributes
    ----------
    initial_values : list of float
        The initial values

    """

    def __init__(self, name="branin", initial_values=[1.0, 1.0], ana_deriv=False):

        super().__init__(
            name,
            float_vars={"x": initial_values[0], "y": initial_values[1]},
            min_values_float={"x": -5.0, "y": 0.0},
            max_values_float={"x": 10.0, "y": 15},
        )

        self.add_objective(BraninObjective(self, ana_deriv=ana_deriv))
