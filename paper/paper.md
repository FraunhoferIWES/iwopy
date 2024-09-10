---
title: 'IWOPY: Fraunhofer IWES optimization tools in Python'
tags:
  - Python
  - Optimization
authors:
  - name: Jonas Schulte
    orcid: 0000-0002-8191-8141
    #equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
affiliations:
 - name: Fraunhofer IWES, Küpkersweg 70, 26129 Oldenburg, Germany
   index: 1
date: 06 February 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Optimization problems are described by optimization variables, which are scalars that are modified during the process of optimization; objective functions, which depend on the variables and whose values define the metric for rating the choices of the latter; and constraints, which are also functions of the optimization variables and define validity of
the solution. The variables can be discrete or continuous, bounded or unbounded; the number of objectives can be one or many; and constraints can require equality or inequality. Many Python packages formulate a framework for the description of such problems, accompanied by a library of optimizers. Hence, switching from one to another optimization package can often 
be tedious and a meta-solution is required that can serve as a single interface to multiple
optimization packages. The package `iwopy` solves this problem by providing a convenient and flexible interface which is general enough to be applicable to a wide range of optimization problems.

# Statement of need

The Python package `iwopy` provides a general object-oriented formulation for optimization 
problems, objective functions, and optimization constraints. The optimization problem class
defines the optimization variables, their bounds and their types. Objectives and constraints
can then be added in an independent step to the problem, such that they can easily be
exchanged and modified by the user. The framework is general enough for supporting complex
science and engineering problems, and it supports single, multi and many objective 
optimization problems.

The core functionality of `iwopy` is to provide interfaces to other existing Python
optimization packages, like `pymoo` [@pymoo], `pygmo` [@pygmo], or `scipy` [@scipy]. Once the 
problem is formulated within the framework sketched above, all individual optimizers from the 
supported linked packages can be selected and switched easily. 

Note that more optimization packages are available which are not yet supported,
like `pyomo` [@pyomo], `Platypus` [@platypus], `DEAP` [@deap] and others.
Each package is well suited for solving a wide range of optimization problems, and they 
all come with extensive user interfaces. However, `iwopy`
addresses a unification of those interfaces, enabling the user to benefit from all supported
optimizers without the need of extensive changes of the code base.

The design of `iwopy` has a focus on vectorized evaluation approaches, as for example often
provided by heuristic algorithms that rely on the concept of populations. If the vectorized
evaluation of a complete population of individual choices of optimization variables is
implemented by the user, this enables a vast speed-up of optimizations compared to the
one-by-one evaluation through a loop.

# Acknowledgements

The development of `iwopy` and has been supported through multiple publicly funded research projects. We acknowledge in particular the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects Smart Wind Farms (grant no. 0325851B) and GW-Wakes (0325397B) as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework of the project H2Digital (03SF0635).

# References