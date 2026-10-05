iwopy.optimizers
----------------
`iwopy`'s own optimizer implementations.

The ``SLSQP`` class solves smooth, continuous, single-objective
problems with variable bounds and lower, upper, two-sided, or equality
constraints. It supplies SciPy's SLSQP implementation with the complete
iwopy Jacobian at every iterate.

SLSQP expands the tolerance owned by each registered constraint in component
order and relaxes the corresponding lower and upper solver bounds. A
positive-tolerance equality becomes two inequalities; set a constraint's
``tol`` to zero when exact equality or inequality bounds are required. A small
inward numerical reserve keeps a boundary solution inside iwopy's feasibility
check.

Optimization variables are transformed internally to dimensionless
coordinates. Variables with finite lower and upper bounds are mapped to
``[-1, 1]``; other variables are scaled from their initial values and finite
bounds where available. Function evaluations, final results, and
``scipy_results.x`` remain in the problem's physical coordinates.

By default, all Jacobians are requested with
``Problem.get_gradients(..., pop=True)``. Analytical derivatives use their
usual iwopy interface. Missing derivatives must be provided by a wrapper such
as ``LocalFD``; its finite-difference points are evaluated together through
the problem's population interface. Set ``vectorized=False`` to use iwopy's
serial gradient evaluation instead. The optimizer never falls back to SciPy's
finite differences.

When finite differences need the unperturbed function values,
``SLSQP`` passes its cached objective and constraint values to the
gradient calculation. ``LocalFD`` then excludes the duplicate center point
from both population and serial finite-difference evaluations.

With positive ``solve`` verbosity, SLSQP reports every accepted iteration with
the original objective value and maximum exact constraint violation. Reporting
uses cached function values and does not trigger additional problem
evaluations. Set ``verbosity=0`` for silent operation. User callbacks remain
active at every verbosity level and receive the same cached iteration state.

``GG`` reports the retained current point after every counted iteration.
Callbacks for both native optimizers and all external interfaces use the common
iwopy callback API described in :doc:`callbacks`.

A complete derivative-free constrained example using ``LocalFD`` is available
in ``examples/slsqp/run.py``. Pass ``--no-pop`` to compare serial gradient
evaluation with the default population-based mode.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/optimizers/index
