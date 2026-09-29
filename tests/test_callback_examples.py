from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "callbacks"


def _load_example(name: str) -> ModuleType:
    """Load a callback example without invoking its command-line entry point."""
    spec = spec_from_file_location(name, EXAMPLE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slsqp_callback_example_generates_history_plot(tmp_path, capsys):
    module = _load_example("run_slsqp")

    plot_path = module.main(output_dir=tmp_path)

    assert "iteration" in capsys.readouterr().out
    assert plot_path == tmp_path / "slsqp_objective_history.png"
    assert plot_path.read_bytes().startswith(b"\x89PNG")


def test_pymoo_callback_example_generates_history_plot(tmp_path, capsys):
    module = _load_example("run_pymoo")

    plot_path = module.main(output_dir=tmp_path, n_generations=2)

    assert "generation" in capsys.readouterr().out
    assert plot_path == tmp_path / "pymoo_objective_history.png"
    assert plot_path.read_bytes().startswith(b"\x89PNG")
