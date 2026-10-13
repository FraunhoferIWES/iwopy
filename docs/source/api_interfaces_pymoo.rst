iwopy.interfaces.pymoo
----------------------
Interface to the `pymoo` package

Select algorithms with ``optimizer`` or use the ``pymoo`` family of
``create_optimizer``. Explicit controls include ``pop_size``, ``seed``,
``vectorize``, ``store_prob_res``, and ``save_history``. Additional native
algorithm options are flat keywords, not a parameter dictionary.

``termination="iwopy"`` uses iwopy's convergence criteria;
``termination="default"`` uses native defaults. ``"iterations"`` and
``"evaluations"`` select pure limits with ``max_iterations`` and
``max_evaluations`` respectively. Both limits may be supplied together.
Native termination objects can also be supplied; explicit limits compose with
them using OR semantics. ``ftol``, ``period``, ``xtol``,
and ``cvtol`` tune applicable convergence criteria; omitted controls keep
backend defaults. Evaluation caps are checked at generation boundaries and
may overshoot. See :ref:`optimizer-termination` and :doc:`optimizer_catalog`.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/pymoo/index
