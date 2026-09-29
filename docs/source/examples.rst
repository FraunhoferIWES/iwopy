Examples
--------
To run these examples, you'll need to have `pygmo` and `pymoo` installed.
You can run `pip install pygmo pymoo` in order to do so.

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

    .. toctree::
        notebooks/simple_function
        notebooks/electrostatics
        notebooks/multi_obj_chain
        notebooks/mixed
        wind_farm_layout
