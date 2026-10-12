iwopy.interfaces.cma
--------------------
Population-vectorized CMA-ES through the `pycma <https://github.com/CMA-ES/pycma>`_ package

Install the ``cma`` extra (``cma>=4.4.0``), or synchronize it in a checkout:

.. code-block:: console

    uv sync --extra cma

``Optimizer_cma`` supports one objective, one or more floating variables, box
bounds, and nonlinear constraints. Integer variables and multiple objectives
are rejected. It does not require derivatives. Importing iwopy or the interface
does not load pycma; optimizer initialization loads the optional backend.

Each generation calls ``ask()``, evaluates the complete ordered population in
one ``Problem.evaluate_population()`` call, and passes the corresponding fitness
and constraint rows to ``tell()``. Provide vectorized problem/function
implementations to benefit from batching. Selected-solution finalization still
uses the problem's individual finalization contract.

Example
~~~~~~~

.. code-block:: python

    import iwopy
    from iwopy.interfaces.cma import Optimizer_cma


    class Quadratic(iwopy.SimpleObjective):
        def f(self, var_x, var_y):
            return (var_x - 1.0) ** 2 + (var_y + 1.0) ** 2


    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars={"x": 3.0, "y": 2.0},
        min_values_float=[-4.0, -4.0],
        max_values_float=[4.0, 4.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.initialize(verbosity=0)

    solver = Optimizer_cma(
        problem,
        sigma0=0.8,
        cma_pars={"seed": 42, "popsize": 8, "maxiter": 200, "ftarget": 1e-9},
    )
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()
    results = solver.solve(verbosity=0, callbacks=[history])
    assert results.success, solver.stop_conditions

Options And Constraints
~~~~~~~~~~~~~~~~~~~~~~~

``sigma0`` is a positive finite initial standard deviation in variable units;
scale variables so one shared step size is meaningful. ``cma_pars`` contains
trusted native pycma options, including population size, seed, stopping budgets,
and convergence tolerances. The adapter manages ``bounds``, ``BoundaryHandler``,
``integer_variables``, ``fixed_variables``, ``transformation``,
``termination_callback``, ``verbose``, ``verb_disp``, and ``verb_log``. Those
options cannot be supplied in ``cma_pars``.

Maximization objectives are negated only for pycma. ``ftarget`` is expressed in
this minimization space; callbacks and final results retain the original signs.
Each constraint's scalar tolerance expands its component bounds. Finite lower
bounds become ``lower - tolerance - value <= 0`` and finite upper bounds become
``value - upper - tolerance <= 0``. All finite lower residuals precede all finite
upper residuals, retaining component registration order within each group.
Unbounded sides are omitted. pycma uses its native augmented Lagrangian; there
is no fixed penalty weight. Equal lower and upper bounds define a tolerance band
and require positive tolerance, rather than an exact equality with zero measure.

Initial values must be finite and within ordered variable bounds. Fixed floating
variables, nonfinite population function values, invalid constraint bounds or
tolerances, and unsupported representations raise contextual exceptions.

Callbacks, Results, And Side Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One ``iteration`` callback is emitted after each completed ``tell``. Its arrays
contain the evaluated population in iwopy ordering, not the strategy mean or
only the champion. The evaluation count excludes selected-solution finalization.
Callbacks use the standard ordered ``initialize``/``notify``/``finalize``
lifecycle and cause no extra problem evaluations.

Every solve starts a new strategy from the problem's initial values. Final
``SingleObjOptResults`` contain the finalized best feasible candidate and its
application-specific ``problem_results``. If no feasible candidate was observed,
the backend's favorite point is finalized and reported as unsuccessful. Success
requires a convergence stop, finite final values, and feasibility; reaching only
``maxiter``, ``maxfevals``, or ``timeout`` is unsuccessful. Inspect
``solver.stop_conditions`` for native termination reasons.

iwopy verbosity controls displayed progress while pycma log writing stays
disabled. The backend constraint logger may create an empty ``outcmaes/``
directory. If pycma's optional ``moarchiving`` package is absent, it may warn that
auxiliary constraint convergence tracking is disabled; the augmented-Lagrangian
optimization and iwopy feasibility checks still operate.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/cma/index
