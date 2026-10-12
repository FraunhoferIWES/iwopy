"""Initialized synthetic quadratic benchmarks for single- and multi-objective workflows."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from iwopy import SimpleConstraint, SimpleObjective, SimpleProblem
from iwopy.core import Problem


class _Quadratic(SimpleObjective):
    def __init__(self, problem: Problem, shift: float) -> None:
        super().__init__(problem, name=f"quadratic_{shift}")
        self.shift = shift

    def f(self, *variables: ArrayLike) -> ArrayLike:
        return sum((np.asarray(values) - self.shift) ** 2 for values in variables)

    def g(
        self,
        var: int,
        *variables: ArrayLike,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> ArrayLike:
        return 2 * (np.asarray(variables[self.problem.n_vars_int + var]) - self.shift)


class QuadraticProblem(SimpleProblem):
    """Bounded two-variable, single-objective quadratic benchmark.

    Minimize ``(x - 0.25)**2 + (y - 0.25)**2`` with both floating variables
    in [-2, 2], initially zero. The objective has analytical derivatives.
    Construction initializes the problem silently; it is ready for an optimizer.
    """

    def __init__(self) -> None:
        """Create and initialize the fixed continuous quadratic benchmark."""
        super().__init__(
            "quadratic",
            float_vars={"x": 0.0, "y": 0.0},
            min_values_float=[-2.0, -2.0],
            max_values_float=[2.0, 2.0],
        )
        self.add_objective(_Quadratic(self, 0.25))
        self.initialize(verbosity=0)


class ParetoProblem(SimpleProblem):
    """Two-objective quadratic benchmark with an optional one-sided constraint.

    Both floating variables x and y have bounds [-2, 2] and start at zero.
    Minimize squared distances to (0.25, 0.25) and (-0.75, -0.75), in that
    objective order. Both objectives provide analytical derivatives.
    Construction initializes the problem silently; further function registration
    requires a different problem definition.
    """

    def __init__(self, constrained: bool = False) -> None:
        """
        Parameters
        ----------
        constrained
            Register x + y <= -0.25 with zero tolerance before initialization.
            The constraint has no analytical derivatives. Without it, only
            variable bounds restrict the problem.
        """
        super().__init__(
            "pareto_quadratic",
            float_vars={"x": 0.0, "y": 0.0},
            min_values_float=[-2.0, -2.0],
            max_values_float=[2.0, 2.0],
        )
        self.add_objective(_Quadratic(self, 0.25))
        self.add_objective(_Quadratic(self, -0.75))
        if constrained:
            restriction = _ParetoRestriction(self)
            restriction.tol = 0.0
            self.add_constraint(restriction)
        self.initialize(verbosity=0)


class _ParetoRestriction(SimpleConstraint):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem, "sum_limit", maxs=-0.25, has_ana_derivs=False)

    def f(self, x_value: ArrayLike, y_value: ArrayLike) -> ArrayLike:
        return np.asarray(x_value) + np.asarray(y_value)


class MixedProblem(SimpleProblem):
    """Bounded integer/float, single-objective quadratic benchmark.

    Minimize ``(index - 0.25)**2 + (x - 0.25)**2``. Integer index starts at
    one with bounds [0, 3]; floating x starts at zero with bounds [-2, 2].
    Analytical derivatives apply to the floating variable only.
    Construction initializes the problem silently; it is ready for an optimizer.
    """

    def __init__(self) -> None:
        """Create and initialize the fixed mixed-variable quadratic benchmark."""
        super().__init__(
            "mixed_quadratic",
            int_vars={"index": 1},
            float_vars={"x": 0.0},
            min_values_int=[0],
            max_values_int=[3],
            min_values_float=[-2.0],
            max_values_float=[2.0],
        )
        self.add_objective(_Quadratic(self, 0.25))
        self.initialize(verbosity=0)
