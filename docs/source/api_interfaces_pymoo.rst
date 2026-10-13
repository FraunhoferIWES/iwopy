iwopy.interfaces.pymoo
----------------------
Interface to the `pymoo` package

Select algorithms with ``optimizer`` or use the ``pymoo`` family of
``create_optimizer``. Explicit controls include ``pop_size``, ``seed``,
``vectorize``, ``store_prob_res``, and ``save_history``. Additional native
algorithm options are flat keywords, not a parameter dictionary.

``termination="iwopy"`` uses iwopy's convergence criteria;
``termination="default"`` uses native defaults. ``n_gen`` and ``n_eval``
select pure limits with ``n_max_gen`` and ``n_max_evals`` respectively.
Native termination objects can also be supplied. ``ftol``, ``period``, ``xtol``,
and ``cvtol`` tune applicable convergence criteria; omitted controls keep
backend defaults. See :doc:`api_core` and :doc:`optimizer_catalog`.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/pymoo/index
