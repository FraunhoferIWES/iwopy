from typing import Any, Protocol

import numpy as np

from iwopy.core import Problem

from . import imports


class _PymooProblem(Protocol):
    is_intprob: bool
    problem: Problem


class Factory:
    """A factory for pymoo components"""

    def __init__(self, pymoo_problem: _PymooProblem, verbosity: int) -> None:
        """
        Parameters
        ----------
        pymoo_problem
            The pymoo problem wrapper
        verbosity
            The verbosity level, 0 = silent
        """
        self.pymoo_problem = pymoo_problem
        self.verbosity = verbosity

        imports.load(verbosity)

    def print(self, *args: object, **kwargs: Any) -> None:
        """
        Print a message when verbose output is enabled.

        Parameters
        ----------
        args
            Positional arguments for the print function
        kwargs
            Keyword arguments for the print function
        """
        if self.verbosity:
            print(*args, **kwargs)

    def get_sampling(self, samp_name: str | None, **kwargs: Any) -> Any:
        """Sampling factory function"""
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

    def get_crossover(self, cross: str, **pars: Any) -> Any:
        """Crossover factory function"""
        if cross == "sbx":
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())
            out = imports.SBX(**pars)
        else:
            raise KeyError(f"Unknown crossover '{cross}', please choose: sbx")

        self.print(f"Selecting crossover: {cross} ({type(out).__name__})")

        return out

    def get_mutation(self, mut: str, **pars: Any) -> Any:
        """Mutation factory function"""
        if mut == "pm":
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())
            out = imports.PM(**pars)
        else:
            raise KeyError(f"Unknown mutation '{mut}', please choose: pm")

        self.print(f"Selecting mutations: {mut} ({type(out).__name__})")

        return out

    def get_algorithm(self, pars: dict[str, Any]) -> Any:
        """Algorithm factory function"""
        pars = pars.copy()
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
            samp_name = pars.get("sampling", None)
            samp_pars = pars.get("sampling_pars", {})
            if "sampling_pars" in pars:
                del pars["sampling_pars"]
            if samp_name is None and self.pymoo_problem.is_intprob:
                samp_name = "int_random"
            if isinstance(samp_name, str):
                pars["sampling"] = self.get_sampling(samp_name, **samp_pars)
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())

            out = imports.DE(**pars)

        # Particle Swarm:
        elif typ == "PSO":
            samp_name = pars.get("sampling", None)
            samp_pars = pars.get("sampling_pars", {})
            if "sampling_pars" in pars:
                del pars["sampling_pars"]
            if samp_name is None and self.pymoo_problem.is_intprob:
                samp_name = "int_random"
            if isinstance(samp_name, str):
                pars["sampling"] = self.get_sampling(samp_name, **samp_pars)

            pars.setdefault("output", imports.SingleObjectiveOutput())
            if self.pymoo_problem.is_intprob:
                pars.setdefault("repair", imports.RoundingRepair())

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
            if self.pymoo_problem.is_intprob:
                raise ValueError("CMAES does not support pure integer problems")
            del pars["type"]
            out = imports.CMAES(**pars)

        # MixedVariableGA:
        elif typ == "MixedVariableGA":
            if self.pymoo_problem.is_intprob:
                raise ValueError(
                    "MixedVariableGA requires a mixed-variable problem representation"
                )
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

    def get_termination(
        self,
        term_pars: dict[str, Any] | tuple[Any, ...] | list[Any],
    ) -> Any:
        """Termination factory function"""

        if isinstance(term_pars, tuple):
            return term_pars
        elif isinstance(term_pars, list):
            return tuple(term_pars)

        term_pars = term_pars.copy()
        typ = term_pars.pop("type", "iwopy")
        if not isinstance(typ, str):
            self.print(f"Selecting termination: {type(typ).__name__}")
            return typ
        elif typ == "iwopy":
            term_pars.setdefault("n_max_evals", np.inf)
            if self.pymoo_problem.problem.n_objectives > 1:
                out = imports.DefaultMultiObjectiveTermination(**term_pars)
            else:
                ftol = term_pars.pop("ftol", 1e-6)
                period = term_pars.pop("period", 30)
                n_max_gen = term_pars.pop("n_max_gen", 1000)
                n_max_evals = term_pars.pop("n_max_evals", np.inf)
                term_pars.pop("xtol", None)
                term_pars.pop("cvtol", None)
                if term_pars:
                    raise KeyError(
                        f"Unknown single-objective default termination parameter(s): {sorted(term_pars)}"
                    )
                out = imports.TerminationCollection(
                    imports.RobustTermination(
                        imports.SingleObjectiveSpaceTermination(ftol, only_feas=True),
                        period=period,
                    ),
                    imports.MaximumGenerationTermination(n_max_gen),
                    imports.MaximumFunctionCallTermination(n_max_evals),
                )
        elif typ == "default":
            term_pars.setdefault("n_max_evals", np.inf)
            if self.pymoo_problem.problem.n_objectives > 1:
                out = imports.DefaultMultiObjectiveTermination(**term_pars)
            else:
                out = imports.DefaultSingleObjectiveTermination(**term_pars)
        else:
            raise KeyError(
                f"Unknown termination '{type}', please choose: iwopy, default"
            )

        self.print(f"Selecting termination: {typ} ({type(out).__name__})")

        return out
