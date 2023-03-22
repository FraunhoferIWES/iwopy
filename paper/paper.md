---
title: 'IWOPY: Fraunhofer IWES optimization tools in Python'
tags:
  - Python
  - Optimization
authors:
  - name: Jonas Schmidt
    orcid: 0000-0002-8191-8141
    #equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Bernhard Stoevesandt
    orcid: 0000-0001-6626-1084
    affiliation: 1
affiliations:
 - name: Fraunhofer IWES, Küpkersweg 70, 26129 Oldenburg, Germany
   index: 1
date: 24 March 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Optimization problems are described by optimization variables, which are scalars that are modified during the process of optimization, objective functions, which depend on the variables and whose values define the metric for rating the choices of the latter, and constraints, which are also functions of the optimization variables and define validity of
the solution. The variables can be discrete or continuous, bounded or unbounded; the number of objectives can be one or many; and constraints can require equality or inequality. Many Python packages formulate a framework for the description of such problems, accompanied by
a library of optimizers. Hence, switching from one to another optimization package can often 
be tedious and a meta-solution is required that can serve as a single interface to multiple
optimization packages.

# Statement of need

The Python package `iwopy` provides a general object-oriented formulation for optimization 
problems, objective functions, and optimization constraints. The optimization problem class
defines the optimization variables, their bounds and their types. Objectives and constraints
can then be added in an independent step to the problem, such that they can easily be
exchanged and modified by the user. The framework is general enough for supporting complex
science and engineering problems, and it supports single, multi and many objective 
optimization problems.

The core functionality of `iwopy` is to provide interfaces to other existing Python
optimization packages, like `pymoo` [@pymoo] or `pygmo` [@pygmo]. Once the problem is
formulated within the framework sketched above, the optimizers from the supported
liked packages can be selected and switched easily. 

The design of `iwopy` has a focus on vectorized evaluation approaches, as for example often
provided by heuristic algorithms that rely on the concept of populations. If the vectorized
evaluation of a complete population of individual choices of optimization variables is
implemented by the user, this enables a vast speed-up of optimizations compared to the
one-by-one evaluation through a loop.

# Acknowledgements

The development of `foxes` and its predecessors flapFOAM and flappy (internal - non public) has been supported through multiple publicly funded research projects. We acknowledge in particular the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects Smart Wind Farms (grant no. 0325851B), GW-Wakes (0325397B) and X-Wakes (03EE3008A) as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework of the project H2Digital (03SF0635).

# References