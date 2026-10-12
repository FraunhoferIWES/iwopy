
IWOPY
=====

.. image:: ../../Logo_IWOPY.svg
    :alt: IWOPY logo
    :width: 100%
    :align: center
    :class: iwopy-logo

Fraunhofer IWES optimization tools in Python.

Define an optimization problem once and choose a native or external solver
through a shared Python interface. External integrations include:

* `pymoo <https://pymoo.org/index.html>`_
* `pygmo <https://esa.github.io/pygmo2/index.html>`_
* `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
* `pycma <https://github.com/CMA-ES/pycma>`_
* `Nevergrad <https://facebookresearch.github.io/nevergrad/>`_
* `Optuna <https://optuna.org/>`_
* `DEAP <https://deap.readthedocs.io/>`_

Native ``GG`` and ``SLSQP`` optimizers are included. ``LocalFD`` supplies finite
differences; ``DiscretizeRegGrid`` wraps a problem on a regular grid. Population
evaluation batches candidates or derivative points when the backend supports it.

Choose a solver in the :doc:`optimizer_catalog`: every selectable optimizer
has a short description and an explicit, runnable Python setup. SciPy
differential evolution, Nevergrad, Optuna, and DEAP support batched search.

For Users
---------

Start with :doc:`installation`, choose a solver in :doc:`optimizer_catalog`,
and explore :doc:`examples`. :doc:`callbacks` covers intermediate results;
:doc:`api` provides the complete reference.

For Developers
--------------

Use :doc:`testing` for quality checks and the repository's
`development guide <https://github.com/FraunhoferIWES/iwopy/blob/dev/docs/development.md>`_
for test ownership and contribution requirements. Submit
`issues <https://github.com/FraunhoferIWES/iwopy/issues>`_ or
`pull requests <https://github.com/FraunhoferIWES/iwopy/pulls>`_ on GitHub.
iwopy is released under the
`MIT license <https://github.com/FraunhoferIWES/iwopy/blob/main/LICENSE>`_.

Contents
--------
.. toctree::
    :maxdepth: 1

    installation
    optimizer_catalog
    examples
    callbacks
    api
    testing
    citation
    CHANGELOG
