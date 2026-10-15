import subprocess
import sys

import iwopy


def test_root_factory_reexports_the_core_function_without_initializing():
    from iwopy.core import create_optimizer

    assert iwopy.create_optimizer is create_optimizer
    solver = iwopy.create_optimizer(
        "scipy", "SLSQP", iwopy.benchmarks.QuadraticProblem()
    )
    assert not solver.initialized


def test_optional_interfaces_import_does_not_load_backends():
    code = """
import sys
from importlib.abc import MetaPathFinder

class BlockBackends(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cma", "pymoo", "pygmo", "nevergrad", "optuna", "deap"}:
            raise AssertionError(f"{fullname} was imported eagerly")

sys.meta_path.insert(0, BlockBackends())
import iwopy
from iwopy.interfaces.cma import Optimizer_cma
from iwopy.interfaces.nevergrad import Optimizer_nevergrad
from iwopy.interfaces.optuna import Optimizer_optuna
from iwopy.interfaces.deap import Optimizer_deap
assert not {"cma", "pymoo", "pygmo", "nevergrad", "optuna", "deap"}.intersection(sys.modules)
assert callable(iwopy.create_optimizer)
assert Optimizer_cma.__name__ == "Optimizer_cma"
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
