
.. image:: ../../Logo_IWOPY.svg
    :align: center

Welcome to IWOPY
================

*Fraunhofer IWES optimization tools in Python*

The `iwopy` package is in fact a meta package that provides interfaces to other open-source Python optimization packages out there. Currently this includes

* https://pymoo.org/index.html
* https://esa.github.io/pygmo2/index.html

The basic idea of `iwopy` is to provide abstract base classes, that can be concretized for any kind of problem by the users, and the corresponding solver interfaces.

`iwopy` can thus be understood as an attempt to provide *the best of all worlds* when it comes to solving optimization problems with Python. Obviously all the credit for implementing the invoked optimizers goes to the original package providers.

**Quick Start**::

    pip install iwopy

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/iwopy

Contact (please report code issues under the github link above):
    :email:`Jonas Schmidt <jonas.schmidt@iwes.fraunhofer.de>`
    
License:
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/iwopy/blob/main/LICENSE

Contents:
    .. toctree::
        :maxdepth: 2
    
        installation

    .. toctree::
        :maxdepth: 2

        notebooks/examples
        
    .. toctree::
        :maxdepth: 1

        api

    .. toctree::
        :maxdepth: 2

        history
