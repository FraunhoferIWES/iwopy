
.. image:: ../../Logo_IWOPY_white.svg
    :align: center

Welcome to IWOPY
================

*Fraunhofer IWES optimization tools in Python*

The `iwopy` package is in fact mainly a meta package that provides interfaces to 
other open-source Python optimization packages out there. Currently this includes

* `pymoo <https://pymoo.org/index.html>`_
* `pygmo <https://esa.github.io/pygmo2/index.html>`_
* `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
* (more to come with future versions)

`iwopy` can thus be understood as an attempt to provide *the best of all worlds* 
when it comes to solving optimization problems with Python. This has not yet been 
achieved, since above list of accessable optimization packages is obviously incomplete, but it's a start. All the credit for implementing the invoked optimizers goes to the original package providers.

The basic idea of `iwopy` is to provide abstract base classes, that can be 
concretized for any kind of problem by the users, and the corresponding solver 
interfaces. However, also some helpful problem wrappers and an original optimizer 
are provided in addition:

* Problem wrapper *LocalFD*: Calculates derivatives by finite differences
* Problem wrapper *RegularDiscretizationGrid*: Puts the problem on a Grid 
* Optimizer *GG*: *Greedy Gradient* optimization with constraints

All calculations support vectorized evaluation of a complete population of 
parameters. This is useful for heuristic approaches like genetic algorithms, 
but also for evaluating gradients. It can lead to a vast speed-up and should be 
invoked whenever possible. Check the examples (or the API) for details.

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/iwopy

Please report code issues under the github link above.
    
License:
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/iwopy/blob/main/LICENSE

Contributing
------------

Please feel invited to contribute to `iwopy`! Here is how:

#. Fork *iwopy* on *github*.
#. Create a branch (`git checkout -b new_branch`)
#. Commit your changes (`git commit -am "your awesome message"`)
#. Push to the branch (`git push origin new_branch`)
#. Create a pull request `here <https://github.com/FraunhoferIWES/iwopy/pulls>`_

Support
-------
For trouble shooting and support, please 

* raise an issue `here <https://github.com/FraunhoferIWES/iwopy/issues>`_,
* or start a discussion `here <https://github.com/FraunhoferIWES/iwopy/discussions>`_,
* or contact the contributers.

Thanks for your help with improving *iwopy*!

Contents
--------
    .. toctree::
        :maxdepth: 2
    
        citation
        
    .. toctree::
        :maxdepth: 2
    
        installation

    .. toctree::
        :maxdepth: 2

        examples
        
    .. toctree::
        :maxdepth: 1

        api

    .. toctree::
        :maxdepth: 2
    
        testing

    .. toctree::
        :maxdepth: 2

        CHANGELOG
