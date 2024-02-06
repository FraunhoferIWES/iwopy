from iwopy.utils import import_module

Callback = None
Problem = None
Real = None
Integer = None

IntegerRandomSampling = None
FloatRandomSampling = None
BinaryRandomSampling = None
PermutationRandomSampling = None

LatinHypercubeSampling = None
SBX = None
PM = None
GA = None
NSGA2 = None
PSO = None
MixedVariableGA = None

DefaultSingleObjectiveTermination = None
DefaultMultiObjectiveTermination = None

minimize = None

loaded = False


def load(verbosity=1):

    global Callback, Problem, Real, Integer, IntegerRandomSampling, FloatRandomSampling
    global BinaryRandomSampling, PermutationRandomSampling, LatinHypercubeSampling, SBX
    global PM, GA, NSGA2, PSO, MixedVariableGA, DefaultSingleObjectiveTermination
    global DefaultMultiObjectiveTermination, minimize, loaded

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
        GA = import_module(
            "pymoo.algorithms.soo.nonconvex.ga", hint="pip install pymoo"
        ).GA
        NSGA2 = import_module(
            "pymoo.algorithms.moo.nsga2", hint="pip install pymoo"
        ).NSGA2
        PSO = import_module(
            "pymoo.algorithms.soo.nonconvex.pso", hint="pip install pymoo"
        ).PSO
        MixedVariableGA = import_module(
            "pymoo.core.mixed", hint="pip install pymoo"
        ).MixedVariableGA

        ter = import_module("pymoo.termination.default", hint="pip install pymoo")
        DefaultSingleObjectiveTermination = ter.DefaultSingleObjectiveTermination
        DefaultMultiObjectiveTermination = ter.DefaultMultiObjectiveTermination

        minimize = import_module("pymoo.optimize", hint="pip install pymoo").minimize

        loaded = True

        if verbosity:
            print("pymoo successfully loaded")
