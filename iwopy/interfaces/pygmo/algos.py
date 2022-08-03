from .imports import pygmo, check_import


class AlgoFactory:
    """
    Creates a pygmo algorithm from parameters
    """

    @staticmethod
    def new(type, **kwargs):
        """
        Create a pygmo algo.

        Optimizer parameters are extracted from kwargs.

        Parameters
        ----------
        type : str
            Name of the driver type
        kwargs : dict, optional
            Additional parameters, type dependent

        Returns
        -------
        pygmo.algo :
            The pygmo algorithm object

        """

        check_import()

        # nlopt:
        if type == "nlopt":

            uda = pygmo.nlopt(kwargs["optimizer"])
            algo = pygmo.algorithm(uda)

            if "ftol_rel" in kwargs:
                algo.extract(pygmo.nlopt).ftol_rel = kwargs["ftol_rel"]
            if "ftol_abs" in kwargs:
                algo.extract(pygmo.nlopt).ftol_abs = kwargs["ftol_abs"]
            if "xtol_rel" in kwargs:
                algo.extract(pygmo.nlopt).xtol_rel = kwargs["xtol_rel"]
            if "xtol_abs" in kwargs:
                algo.extract(pygmo.nlopt).xtol_abs = kwargs["xtol_abs"]
            if "maxeval" in kwargs:
                algo.extract(pygmo.nlopt).maxeval = kwargs["maxeval"]
            if "maxtime" in kwargs:
                algo.extract(pygmo.nlopt).maxtime = kwargs["maxtime"]

        # ipopt:
        elif type == "ipopt":

            uda = pygmo.ipopt()

            for k, a in kwargs.items():

                if isinstance(a, int):
                    uda.set_integer_option(k, a)

                elif isinstance(a, float):
                    uda.set_numeric_option(k, a)

                else:
                    uda.set_string_option(k, a)

            algo = pygmo.algorithm(uda)

        # sga:
        elif type == "sga":

            """

            Simple Genetic Algorithm

            Args:
                gen (``int``): number of generations.
                cr (``float``): crossover probability.
                eta_c (``float``): distribution index for ``sbx`` crossover. This parameter is inactive if other types of crossover are selected.
                m (``float``): mutation probability.
                param_m (``float``): distribution index (``polynomial`` mutation), gaussian width (``gaussian`` mutation) or inactive (``uniform`` mutation)
                param_s (``float``): the number of best individuals to use in "truncated" selection or the size of the tournament in ``tournament`` selection.
                crossover (``str``): the crossover strategy. One of ``exponential``, ``binomial``, ``single`` or ``sbx``
                mutation (``str``): the mutation strategy. One of ``gaussian``, ``polynomial`` or ``uniform``.
                selection (``str``): the selection strategy. One of ``tournament``, "truncated".
                seed (``int``): seed used by the internal random number generator

                The various blocks of pygmo genetic algorithm are listed below:

                *Selection*: two selection methods are provided: ``tournament`` and ``truncated``. ``Tournament`` selection works by
                selecting each offspring as the one having the minimal fitness in a random group of size *param_s*. The ``truncated``
                selection, instead, works selecting the best *param_s* chromosomes in the entire population over and over.
                We have deliberately not implemented the popular roulette wheel selection as we are of the opinion that such
                a system does not generalize much being highly sensitive to the fitness scaling.

                *Crossover*: four different crossover schemes are provided:``single``, ``exponential``, ``binomial``, ``sbx``. The
                ``single`` point crossover, works selecting a random point in the parent chromosome and, with probability *cr*, inserting the
                partner chromosome thereafter. The ``exponential`` crossover is taken from the algorithm differential evolution,
                implemented, in pygmo, as :class:`~pygmo.de`. It essentially selects a random point in the parent chromosome and inserts,
                in each successive gene, the partner values with probability  *cr* up to when it stops. The binomial crossover
                inserts each gene from the partner with probability *cr*. The simulated binary crossover (called ``sbx``), is taken
                from the NSGA-II algorithm, implemented in pygmo as :class:`~pygmo.nsga2`, and makes use of an additional parameter called
                distribution index *eta_c*.

                *Mutation*: three different mutations schemes are provided: ``uniform``, ``gaussian`` and ``polynomial``. Uniform mutation
                simply randomly samples from the bounds. Gaussian muattion samples around each gene using a normal distribution
                with standard deviation proportional to the *param_m* and the bounds width. The last scheme is the ``polynomial``
                mutation from Deb.

                *Reinsertion*: the only reinsertion strategy provided is what we call pure elitism. After each generation
                all parents and children are put in the same pool and only the best are passed to the next generation.

            """

            uda = pygmo.sga(**kwargs)

            algo = pygmo.algorithm(uda)

        # pso:
        elif type == "pso":

            """
            Args:
                gen (``int``): number of generations
                omega (``float``): inertia weight (or constriction factor)
                eta1 (``float``): social component
                eta2 (``float``): cognitive component
                max_vel (``float``): maximum allowed particle velocities (normalized with respect to the bounds width)
                variant (``int``): algorithmic variant
                neighb_type (``int``): swarm topology (defining each particle's neighbours)
                neighb_param (``int``): topology parameter (defines how many neighbours to consider)
                memory (``bool``): when true the velocities are not reset between successive calls to the evolve method
                seed (``int``): seed used by the internal random number generator (default is random)

            Raises:
                OverflowError: if *gen* or *seed* is negative or greater than an implementation-defined value
                ValueError: if *omega* is not in the [0,1] interval, if *eta1*, *eta2* are not in the [0,4] interval, if *max_vel* is not in ]0,1]
                ValueError: *variant* is not one of 1 .. 6, if *neighb_type* is not one of 1 .. 4 or if *neighb_param* is zero

            The following variants can be selected via the *variant* parameter:

            +-----------------------------------------+-----------------------------------------+
            | 1 - Canonical (with inertia weight)     | 2 - Same social and cognitive rand.     |
            +-----------------------------------------+-----------------------------------------+
            | 3 - Same rand. for all components       | 4 - Only one rand.                      |
            +-----------------------------------------+-----------------------------------------+
            | 5 - Canonical (with constriction fact.) | 6 - Fully Informed (FIPS)               |
            +-----------------------------------------+-----------------------------------------+

            The following topologies are selected by *neighb_type*:

            +--------------------------------------+--------------------------------------+
            | 1 - gbest                            | 2 - lbest                            |
            +--------------------------------------+--------------------------------------+
            | 3 - Von Neumann                      | 4 - Adaptive random                  |
            +--------------------------------------+--------------------------------------+

            """

            uda = pygmo.pso(**kwargs)

            algo = pygmo.algorithm(uda)

        # bee_colony:
        elif type == "bee_colony":

            """
            Artificial Bee Colony.

            Args:
                gen (``int``): number of generations
                limit (``int``): maximum number of trials for abandoning a source
                seed (``int``): seed used by the internal random number generator (default is random)

            Raises:
                OverflowError: if *gen*, *limit* or *seed* is negative or greater than an implementation-defined value
                ValueError: if *limit* is not greater than 0

            """

            uda = pygmo.bee_colony(**kwargs)

            algo = pygmo.algorithm(uda)

        # nsga2:
        elif type == "nsga2":

            """
            Non dominated Sorting Genetic Algorithm (NSGA-II).

            Args:
                gen (``int``): number of generations
                cr (``float``): crossover probability
                eta_c (``float``): distribution index for crossover
                m (``float``): mutation probability
                eta_m (``float``): distribution index for mutation
                seed (``int``): seed used by the internal random number generator (default is random)

            Raises:
                OverflowError: if *gen* or *seed* are negative or greater than an implementation-defined value
                ValueError: if either *cr* is not in [0,1[, *eta_c* is not in [0,100[, *m* is not in [0,1], or
                *eta_m* is not in [0,100[

            """

            uda = pygmo.nsga2(**kwargs)

            algo = pygmo.algorithm(uda)

        # unknown driver:
        else:
            estr = f"Unknown uda type '{type}'.\nKnown solvers:"
            estr += "\n  nlopt"
            estr += "\n  ipopt"
            estr += "\n  sga"
            estr += "\n  pso"
            estr += "\n  bee_colony"
            estr += "\n  nsga2"
            raise KeyError(estr)

        algo.set_verbosity(1)

        return algo
