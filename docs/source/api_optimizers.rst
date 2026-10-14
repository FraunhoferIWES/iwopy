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
constraints. It supplies SciPy's SLSQP implementation with the iwopy
objective and selected constraint Jacobian at every iterate.

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

Feasible Output And Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SLSQP returns the best finite, fully feasible evaluated point, or the raw
endpoint if none was feasible. ``best_feasible`` returns copies of that point's
variables, objectives, and full constraints. ``scipy_results`` keeps the primary
physical endpoint and original status. Success still requires primary SciPy
convergence at a feasible endpoint; returning an earlier feasible point alone
does not imply convergence.

These controls use iwopy's objective direction, evaluation, and derivative
contracts, not a particular objective class. New continuous scalar objectives
can use them with analytical or wrapped numerical derivatives. ``progress_tol``
is absolute in evaluated objective units, before any finalization rescaling;
choose it for the objective's scale. Multiobjective and integer problems remain
outside SLSQP's contract.

``feasibility_margin`` reduces each positive solver tolerance in its own units,
clamped at zero, without changing public constraint tolerances. For example,
``tol=0.5`` with ``feasibility_margin=0.01`` gives solver slack of 0.49. Exact
zero-tolerance equalities remain exact.

``restoration_iterations`` enables bounded minimization of full squared
constraint violations before an infeasible start and after an infeasible
primary endpoint. Each phase has its own iteration budget and uses only
violated-constraint derivatives. Restoration is not a nearest-point projection
and does not guarantee feasibility. ``restoration_results`` retains the most
recent physical restoration endpoint.

Enabled restoration for constrained problems requires ``restoration_scales``:
positive finite characteristic constraint values, scalar or an array of shape
``(n_constraints,)`` in full constraint order. Violations are divided by their
scales before squaring, and the merit Jacobian uses the same normalization.
For example, if one component changes from metres to millimetres, multiply its
scale by 1000 too. Choose scales for comparable physical significance across
components; iwopy does not infer units or substitute tolerances for scales.
Public constraints and feasibility checks are unchanged.

Restoration uses a dimensionless constraint-merit tolerance of ``1e-12``,
independent of objective units and ``progress_tol``.

``stall_iterations`` stops primary iterations when cumulative improvement of
the best fully feasible objective does not exceed ``progress_tol`` within that
patience window. Patience begins only after a feasible point exists; before
that, native stopping criteria and ``max_iterations`` govern termination.
It handles minimization and maximization, and adds no gradient
calls. ``termination_reason`` identifies primary convergence, SciPy failure, or
``feasible_stagnation``. ``n_iterations`` includes all accepted restoration and
primary iterations; primary ``scipy_results.nit`` excludes restoration.

Guarded Constraint Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``move_limit`` defines positive finite per-coordinate displacement bounds
around the initial point for the entire solve. ``reduce_constraints=True``
requests certified selections through
``Problem.get_active_constraint_components`` and each constraint's
``active_components`` hook. Defaults retain all components. Omission is allowed
only when a component is guaranteed feasible throughout enforced bounds.
``active_constraint_components`` stores sorted global constraint indices.

Selections reduce both Jacobian rows and SciPy constraint rows, never full
evaluation, final checks, or public component identities. The enforced box and
selection stay fixed during a solve and are rebuilt before the next solve. A
caller moving to another local region must start another solve; changing
constraint dimensions within a SciPy call is not supported.

Scaling And Derivatives
~~~~~~~~~~~~~~~~~~~~~~~

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

The last callback describes the returned point with finalized function values.
An objective may rescale these values at finalization, so they can differ in
units from cached optimization values and ``best_feasible``.

``GG`` reports the retained current point after every counted iteration.
Callbacks for both iwopy optimizers and all external interfaces use the common
iwopy callback API described in :doc:`callbacks`.

A complete derivative-free constrained example using ``LocalFD`` is available
in ``examples/slsqp/run.py``. Pass ``--no-pop`` to compare serial Jacobian
evaluation with the default population-based mode.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/optimizers/index
