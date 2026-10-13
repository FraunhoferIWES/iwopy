Examples
--------
Start with the :doc:`optimizer_catalog` for a short, runnable setup for each
optimizer. Install only the extra required by your chosen backend; SciPy and
the iwopy optimizers are included in the base package.

Pareto flows
^^^^^^^^^^^^

The catalog contains explicit unconstrained and constrained NSGA2/NSGA3 flows,
including selection of a weighted compromise from iwopy Pareto results:

.. code-block:: console

    uv run --extra pymoo python examples/optimizers/pymoo_nsga2.py
    uv run --extra pymoo python examples/optimizers/pymoo_nsga2_constrained.py

See :ref:`case-pymoo-nsga3` and :ref:`case-pymoo-nsga3-constrained` for the
reference-direction alternative.

Callback scripts
^^^^^^^^^^^^^^^^

The ``examples/callbacks`` folder contains standalone SLSQP and pymoo scripts
that print intermediate optimization results with a custom callback and record
the same states with ``OptimizationHistory``. Run them from the repository root:

.. code-block:: bash

    uv run python examples/callbacks/run_slsqp.py
    uv run --extra pymoo python examples/callbacks/run_pymoo.py

Both scripts save objective-history plots to the Git-ignored
``examples/callbacks/output`` directory.

Notebook workflows
^^^^^^^^^^^^^^^^^^

The notebooks explore gradients, multi-objective results, and mixed variables.
Some require pymoo or PyGMO; see their imports and :doc:`installation`.

.. toctree::

    notebooks/simple_function
    notebooks/electrostatics
    notebooks/multi_obj_chain
    notebooks/mixed
    wind_farm_layout
