from typing import Any

from iwopy.utils import import_module

Callback: Any = None
Problem: Any = None
Real: Any = None
Integer: Any = None

IntegerRandomSampling: Any = None
FloatRandomSampling: Any = None
BinaryRandomSampling: Any = None
PermutationRandomSampling: Any = None

LatinHypercubeSampling: Any = None
SBX: Any = None
PM: Any = None
RoundingRepair: Any = None
GA: Any = None
DE: Any = None
NSGA2: Any = None
NSGA3: Any = None
PSO: Any = None
CMAES: Any = None
MixedVariableGA: Any = None
get_reference_directions: Any = None

DefaultSingleObjectiveTermination: Any = None
DefaultMultiObjectiveTermination: Any = None
SingleObjectiveSpaceTermination: Any = None
TerminationCollection: Any = None
MaximumGenerationTermination: Any = None
MaximumFunctionCallTermination: Any = None
RobustTermination: Any = None
SingleObjectiveOutput: Any = None

minimize: Any = None

loaded: bool = False


def load(verbosity: int = 1) -> None:
    """
    Loads the pymoo package dynamically

    Parameters
    ----------
    verbosity
        The verbosity level, 0 = silent
    """

    global Callback, Problem, Real, Integer, IntegerRandomSampling, FloatRandomSampling
    global BinaryRandomSampling, PermutationRandomSampling, LatinHypercubeSampling, SBX
    global PM, RoundingRepair, GA, DE, NSGA2, NSGA3, PSO, CMAES, MixedVariableGA
    global get_reference_directions
    global DefaultSingleObjectiveTermination, DefaultMultiObjectiveTermination
    global SingleObjectiveSpaceTermination, TerminationCollection
    global MaximumGenerationTermination, MaximumFunctionCallTermination
    global RobustTermination
    global SingleObjectiveOutput
    global minimize, loaded

    if not loaded:
        if verbosity:
            print("Loading pymoo")

        Callback = import_module(
            "pymoo.core.callback", hint="pip install pymoo"
        ).Callback
        Problem = import_module("pymoo.core.problem", hint="pip install pymoo").Problem
        Real = import_module("pymoo.core.variable", hint="pip install pymoo").Real
        Integer = import_module("pymoo.core.variable", hint="pip install pymoo").Integer

        rnd = import_module("pymoo.operators.sampling.rnd", hint="pip install pymoo")
        IntegerRandomSampling = rnd.IntegerRandomSampling
        FloatRandomSampling = rnd.FloatRandomSampling
        BinaryRandomSampling = rnd.BinaryRandomSampling
        PermutationRandomSampling = rnd.PermutationRandomSampling

        LatinHypercubeSampling = import_module(
            "pymoo.operators.sampling.lhs", hint="pip install pymoo"
        ).LatinHypercubeSampling
        SBX = import_module(
            "pymoo.operators.crossover.sbx", hint="pip install pymoo"
        ).SBX
        PM = import_module("pymoo.operators.mutation.pm", hint="pip install pymoo").PM
        RoundingRepair = import_module(
            "pymoo.operators.repair.rounding", hint="pip install pymoo"
        ).RoundingRepair
        GA = import_module(
            "pymoo.algorithms.soo.nonconvex.ga", hint="pip install pymoo"
        ).GA
        DE = import_module(
            "pymoo.algorithms.soo.nonconvex.de", hint="pip install pymoo"
        ).DE
        NSGA2 = import_module(
            "pymoo.algorithms.moo.nsga2", hint="pip install pymoo"
        ).NSGA2
        NSGA3 = import_module(
            "pymoo.algorithms.moo.nsga3", hint="pip install pymoo"
        ).NSGA3
        PSO = import_module(
            "pymoo.algorithms.soo.nonconvex.pso", hint="pip install pymoo"
        ).PSO
        CMAES = import_module(
            "pymoo.algorithms.soo.nonconvex.cmaes", hint="pip install pymoo"
        ).CMAES
        MixedVariableGA = import_module(
            "pymoo.core.mixed", hint="pip install pymoo"
        ).MixedVariableGA
        get_reference_directions = import_module(
            "pymoo.util.ref_dirs", hint="pip install pymoo"
        ).get_reference_directions

        ter = import_module("pymoo.termination.default", hint="pip install pymoo")
        DefaultSingleObjectiveTermination = ter.DefaultSingleObjectiveTermination
        DefaultMultiObjectiveTermination = ter.DefaultMultiObjectiveTermination
        SingleObjectiveSpaceTermination = import_module(
            "pymoo.termination.ftol", hint="pip install pymoo"
        ).SingleObjectiveSpaceTermination
        TerminationCollection = import_module(
            "pymoo.termination.collection", hint="pip install pymoo"
        ).TerminationCollection
        MaximumGenerationTermination = import_module(
            "pymoo.termination.max_gen", hint="pip install pymoo"
        ).MaximumGenerationTermination
        MaximumFunctionCallTermination = import_module(
            "pymoo.termination.max_eval", hint="pip install pymoo"
        ).MaximumFunctionCallTermination
        RobustTermination = import_module(
            "pymoo.termination.robust", hint="pip install pymoo"
        ).RobustTermination

        SingleObjectiveOutput = import_module(
            "pymoo.util.display.single", hint="pip install pymoo"
        ).SingleObjectiveOutput

        minimize = import_module("pymoo.optimize", hint="pip install pymoo").minimize

        loaded = True

        if verbosity:
            print("pymoo successfully loaded")
