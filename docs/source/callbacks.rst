Optimizer callbacks
===================

Callback-enabled iwopy optimizers accept an ordered list of callbacks through
``solve(callbacks=...)``. Callbacks are independent of solver verbosity and
receive normalized, immutable intermediate data without causing additional
problem evaluations. PyGMO IPOPT does not accept callbacks because pygmo does
not expose exact live IPOPT iterations. When callbacks are omitted or the list
is empty, iwopy does not install backend callback adapters or construct
intermediate callback data.

A callback derives from ``iwopy.OptimizerCallback`` and implements ``notify``:

.. code-block:: python

    import iwopy


    class ObjectiveOutput(iwopy.OptimizerCallback):
        def notify(self, data):
            step = data.iteration or data.n_evaluations
            if step is not None and step % 10 == 0 and data.objs is not None:
                print(step, data.objs[:, 0].min())


    history = iwopy.OptimizationHistory()
    results = solver.solve(
        verbosity=0,
        callbacks=[ObjectiveOutput(), history],
    )
    history.plot_objective()

Lifecycle
---------

For each successful solve, iwopy calls ``initialize(optimizer)`` once,
``notify(data)`` for every available intermediate event, and
``finalize(results)`` once after the iwopy result object has been created.
Reusing a callback in a later solve starts a new lifecycle;
``OptimizationHistory`` clears its recorded states during initialization.

Callbacks run in list order. Exceptions raised by callbacks propagate to the
caller and abort the solve. If backend execution or a callback raises,
``finalize`` is not called because no completed iwopy result exists. Callbacks
that acquire resources must therefore release them within their own exception
handling. There is no implicit cancellation return value.

Backend-native callback parameters are managed by iwopy and rejected during
optimizer initialization. Register callbacks only through
``solve(callbacks=...)``.

Intermediate data
-----------------

``OptimizerCallbackData`` contains the following fields:

``event``
    Either ``"iteration"`` or ``"evaluation"``.
``iteration``
    Solver iteration or generation when the backend exposes one.
``n_evaluations``
    Cumulative evaluated individuals when the backend exposes this count.
``vars_int`` and ``vars_float``
    Two-dimensional population arrays. A single-point optimizer uses one row;
    an absent variable group has zero columns.
``objs`` and ``cons``
    Two-dimensional arrays in original iwopy conventions. These are ``None``
    when an external optimizer reports coordinates that are not present in the
    iwopy evaluation cache. No evaluation is performed merely to fill a
    callback state.

The arrays are defensive, read-only copies. A callback that deliberately calls
problem evaluation or output methods through ``self.optimizer.problem`` owns
the cost and side effects of those calls.

Backend events
--------------

.. list-table::
    :header-rows: 1

    * - Optimizer
      - Event
      - Population
    * - ``GG``
      - Completed GG iteration
      - Retained current point
    * - ``SLSQP``
      - Accepted SciPy SLSQP iterate
      - Accepted point
    * - ``Optimizer_scipy``
      - Native SciPy iteration callback
      - Current point
    * - ``Optimizer_pymoo``
      - Completed pymoo generation
      - Full current population
    * - ``Optimizer_pygmo``
      - Live fitness evaluation, except for IPOPT
      - One point or one evaluated batch

PyGMO exposes optimization as one opaque ``evolve()`` call and has no common
live generation callback across its algorithms. iwopy therefore reports exact
fitness evaluations instead of simulating generations or relying on
algorithm-specific post-run logs. PyGMO's IPOPT progress log reports objective
evaluations without decision vectors, while gradient requests do not identify
IPOPT iterations reliably. Supplying callbacks to PyGMO IPOPT therefore raises
``NotImplementedError``.
