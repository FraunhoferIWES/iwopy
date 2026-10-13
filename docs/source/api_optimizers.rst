iwopy.optimizers
----------------
`iwopy`'s own optimizer implementations.

Construct ``GG`` or ``SLSQP`` with explicit parameters, or use
``create_optimizer(family="iwopy", optimizer="GG", problem=..., ...)``
and the corresponding ``SLSQP`` selector. GG requires ``step_max`` and
``step_min`` and exposes ``step_div_factor``, ``f_tol``, ``n_max_steps``,
``max_iterations``, ``memory_size``, and ``vectorize``. SLSQP exposes ``tol``,
``max_iterations``, ``memory_size``, and ``vectorize``; flat native method
options such as ``ftol`` are forwarded to SciPy. Neither optimizer supports
``max_evaluations``. See :ref:`optimizer-parameters` and :ref:`optimizer-termination`.

The ``SLSQP`` class solves smooth, continuous, single-objective
problems with variable bounds and lower, upper, two-sided, or equality
constraints. It supplies SciPy's SLSQP implementation with the complete
iwopy Jacobian at every iterate.

Jacobians remain in CSR cache storage. Native SLSQP callbacks receive only
their requested rows as dense arrays; its native workspace remains dense.
GG uses lazy CSR cache slots and sparse matrix-vector constraint predictions,
expanding individual rows only for projection steps.

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
``Problem.get_jacobian(..., pop=True)``. Analytical derivatives use their
usual iwopy interface. Missing derivatives must be provided by a wrapper such
as ``LocalFD``; its finite-difference points are evaluated together through
the problem's population interface. Set ``vectorize=False`` to use iwopy's
serial Jacobian evaluation instead. The optimizer never falls back to SciPy's
finite differences.

When finite differences need the unperturbed function values,
``SLSQP`` passes its cached objective and constraint values to the
Jacobian calculation. ``LocalFD`` then excludes the duplicate center point
from both population and serial finite-difference evaluations.

With positive ``solve`` verbosity, SLSQP reports every accepted iteration with
the original objective value and maximum exact constraint violation. Reporting
uses cached function values and does not trigger additional problem
evaluations. Set ``verbosity=0`` for silent operation. User callbacks remain
active at every verbosity level and receive the same cached iteration state.

``GG`` reports the retained current point after every counted iteration.
Callbacks for both iwopy optimizers and all external interfaces use the common
iwopy callback API described in :doc:`callbacks`.

A complete derivative-free constrained example using ``LocalFD`` is available
in ``examples/slsqp/run.py``. Pass ``--no-pop`` to compare serial Jacobian
evaluation with the default population-based mode.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/optimizers/index
