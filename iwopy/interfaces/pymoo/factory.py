import numpy as np

from . import imports


class Factory:
    """
    A factory for pymoo components

    :group: interfaces.pymoo

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
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())
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
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())
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
            elif "crossover" not in pars and self.pymoo_problem.is_intprob:
                pars["crossover"] = self.get_crossover("sbx", **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)
            elif "mutation" not in pars and self.pymoo_problem.is_intprob:
                pars["mutation"] = self.get_mutation("pm", **mut_pars)

            out = imports.GA(**pars)

        # Differential Evolution:
        elif typ == "DE":
            if "sampling" in pars:
                samp_name = pars.get("sampling", None)
                samp_pars = pars.get("sampling_pars", {})
                if "sampling_pars" in pars:
                    del pars["sampling_pars"]
                if isinstance(samp_name, str):
                    pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            out = imports.DE(**pars)

        # Particle Swarm:
        elif typ == "PSO":
            if "sampling" in pars:
                samp_name = pars.get("sampling", None)
                samp_pars = pars.get("sampling_pars", {})
                if "sampling_pars" in pars:
                    del pars["sampling_pars"]
                if isinstance(samp_name, str):
                    pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            pars.setdefault("output", imports.SingleObjectiveOutput())

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
            elif "crossover" not in pars and self.pymoo_problem.is_intprob:
                pars["crossover"] = self.get_crossover("sbx", **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)
            elif "mutation" not in pars and self.pymoo_problem.is_intprob:
                pars["mutation"] = self.get_mutation("pm", **mut_pars)

            out = imports.NSGA2(**pars)

        # NSGA3:
        elif typ == "NSGA3":
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
            elif "crossover" not in pars and self.pymoo_problem.is_intprob:
                pars["crossover"] = self.get_crossover("sbx", **cross_pars)

            mut_pars = pars.get("mutation_pars", {})
            if "mutation_pars" in pars:
                del pars["mutation_pars"]
            if "mutation" in pars and isinstance(pars["mutation"], str):
                mut = pars["mutation"]
                pars["mutation"] = self.get_mutation(mut, **mut_pars)
            elif "mutation" not in pars and self.pymoo_problem.is_intprob:
                pars["mutation"] = self.get_mutation("pm", **mut_pars)

            if "ref_dirs" not in pars:
                n_obj = self.pymoo_problem.problem.n_objectives
                n_partitions = pars.pop("n_partitions", 12)
                pars["ref_dirs"] = imports.get_reference_directions(
                    "das-dennis", n_obj, n_partitions=n_partitions
                )

            out = imports.NSGA3(**pars)

        # Covariance Matrix Adaptation Evolution Strategy:
        elif typ == "CMAES":
            del pars["type"]
            out = imports.CMAES(**pars)

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
                f"Unknown algorithm '{typ}', please choose: GA, DE, PSO, NSGA2, NSGA3, CMAES, MixedVariableGA"
            )

        self.print(f"Selecting algorithm: {typ} ({type(out).__name__})")

        return out

    def get_termination(self, term_pars):
        """
        Termination factory function
        """

        if isinstance(term_pars, tuple):
            return term_pars
        elif isinstance(term_pars, list):
            return tuple(term_pars)

        typ = term_pars.pop("type", None)
        if typ is None:
            return None
        elif not isinstance(typ, str):
            self.print(f"Selecting termination: {type(typ).__name__}")
            return typ
        elif typ == "default":
            term_pars.setdefault("n_max_evals", np.inf)
            if self.pymoo_problem.problem.n_objectives > 1:
                out = imports.DefaultMultiObjectiveTermination(**term_pars)
            else:
                out = imports.DefaultSingleObjectiveTermination(**term_pars)
        else:
            raise KeyError(f"Unknown termination '{type}', please choose: default")

        self.print(f"Selecting termination: {typ} ({type(out).__name__})")

        return out
