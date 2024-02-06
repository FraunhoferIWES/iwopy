from . import imports


class Factory:
    """
    A factory for pymoo components
    """

    def __init__(self, pymoo_problem, verbosity):
        self.pymoo_problem = pymoo_problem
        self.verbosity = verbosity

        imports.load(verbosity)

    def print(self, *args, **kwargs):
        if self.verbosity:
            print(*args, **kwargs)

    def get_sampling(self, samp_name, **kwargs):
        """
        Sampling factory function
        """
        if samp_name is None:
            if self.pymoo_problem.is_intprob:
                samp_name = "int_random"
            else:
                samp_name = "float_random"

        if samp_name == "int_random":
            out = imports.IntegerRandomSampling(**kwargs)
        elif samp_name == "float_random":
            out = imports.FloatRandomSampling(**kwargs)
        elif samp_name == "binary_random":
            out = imports.BinaryRandomSampling(**kwargs)
        elif samp_name == "permutation_random":
            out = imports.PermutationRandomSampling(**kwargs)
        elif samp_name == "lhs":
            out = imports.LatinHypercubeSampling(**kwargs)
        else:
            raise KeyError(
                f"Unknown sampling '{samp_name}', please choose: int_random, float_random, binary_random, permutation_random, lhs"
            )

        self.print(f"Selecting sampling: {samp_name} ({type(out).__name__})")

        return out

    def get_crossover(self, cross, **pars):
        """
        Crossover factory function
        """
        if cross == "sbx":
            out = imports.SBX(**pars)
        else:
            raise KeyError(f"Unknown crossover '{cross}', please choose: sbx")

        self.print(f"Selecting crossover: {cross} ({type(out).__name__})")

        return out

    def get_mutation(self, mut, **pars):
        """
        Mutation factory function
        """
        if mut == "pm":
            out = imports.PM(**pars)
        else:
            raise KeyError(f"Unknown mutation '{mut}', please choose: pm")

        self.print(f"Selecting mutations: {mut} ({type(out).__name__})")

        return out

    def get_algorithm(self, pars):
        """
        Algorithm factory function
        """
        typ = pars["type"]

        # Genetic Algorithm:
        if typ == "GA":

            samp_name = pars.get("sampling", None)
            samp_pars = pars.get("sampling_pars", {})
            if "sampling_pars" in pars:
                del pars["sampling_pars"]
            pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            cross_pars = pars.get("crossover_pars", {})
            if "crossover_pars" in pars:
                del pars["crossover_pars"]
            if "crossover" in pars and isinstance(pars["crossover"], str):
                cross = pars["crossover"]
                pars["crossover"] = self.get_crossover(cross, **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)

            out = imports.GA(**pars)

        # Particle Swarm:
        elif typ == "PSO":

            if "samplig" in pars:
                samp_name = pars.get("sampling", None)
                samp_pars = pars.get("sampling_pars", {})
                if "sampling_pars" in pars:
                    del pars["sampling_pars"]
                pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            cross_pars = pars.get("crossover_pars", {})
            if "crossover_pars" in pars:
                del pars["crossover_pars"]
            if "crossover" in pars and isinstance(pars["crossover"], str):
                cross = pars["crossover"]
                pars["crossover"] = self.get_crossover(cross, **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)

            out = imports.PSO(**pars)

        # NSGA2:
        elif typ == "NSGA2":

            samp_name = pars.get("sampling", None)
            samp_pars = pars.get("sampling_pars", {})
            if "sampling_pars" in pars:
                del pars["sampling_pars"]
            pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            cross_pars = pars.get("crossover_pars", {})
            if "crossover_pars" in pars:
                del pars["crossover_pars"]
            if "crossover" in pars and isinstance(pars["crossover"], str):
                cross = pars["crossover"]
                pars["crossover"] = self.get_crossover(cross, **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)

            out = imports.NSGA2(**pars)

        # MixedVariableGA:
        elif typ == "MixedVariableGA":

            cross_pars = pars.get("crossover_pars", {})
            if "crossover_pars" in pars:
                del pars["crossover_pars"]
            if "crossover" in pars and isinstance(pars["crossover"], str):
                cross = pars["crossover"]
                pars["crossover"] = self.get_crossover(cross, **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)

            out = imports.MixedVariableGA(**pars)

        else:
            raise KeyError(
                f"Unknown algorithm '{typ}', please choose: GA, PSO, NSGA2, MixedVariableGA"
            )

        self.print(f"Selecting algorithm: {typ} ({type(out).__name__})")

        return out

    def get_termination(self, term_pars):

        if isinstance(term_pars, tuple):
            return term_pars

        typ = term_pars.pop("type", None)
        if typ is None:
            return None
        elif not isinstance(typ, str):
            self.print(f"Selecting termination: {type(typ).__name__}")
            return typ
        elif typ == "default":
            if self.pymoo_problem.problem.n_objectives > 1:
                out = imports.DefaultMultiObjectiveTermination(**term_pars)
            else:
                out = imports.DefaultSingleObjectiveTermination(**term_pars)
        else:
            raise KeyError(f"Unknown termination '{type}', please choose: default")

        self.print(f"Selecting termination: {typ} ({type(out).__name__})")

        return out
