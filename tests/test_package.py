import subprocess
import sys

import iwopy


def test_package_version_is_nonempty_string():
    assert isinstance(iwopy.__version__, str)
    assert iwopy.__version__


def test_cma_interface_import_does_not_load_optional_backend():
    code = """
import sys
from importlib.abc import MetaPathFinder

class BlockCma(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cma" or fullname.startswith("cma."):
            raise AssertionError("pycma was imported eagerly")

sys.meta_path.insert(0, BlockCma())
import iwopy
from iwopy.interfaces.cma import Optimizer_cma
assert "cma" not in sys.modules
assert Optimizer_cma.__name__ == "Optimizer_cma"
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
