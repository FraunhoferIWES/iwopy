from pathlib import Path

import matplotlib.pyplot as plt

import iwopy
from iwopy.benchmarks.branin import BraninProblem
from iwopy.optimizers import SLSQP


class IterationOutput(iwopy.OptimizerCallback):
    """Print each accepted SLSQP iterate."""

    def notify(self, data: iwopy.OptimizerCallbackData) -> None:
        """Print the current variables and objective value."""
        variables = ", ".join(f"{value:.6f}" for value in data.vars_float[0])
        if data.objs is None:
            objective = "unavailable"
        else:
            objective = f"{data.objs[0, 0]:.8f}"
        print(
            f"iteration {data.iteration:>2}: x = [{variables}], objective = {objective}"
        )


def main(output_dir: Path | None = None) -> Path:
    """Run SLSQP with live output and objective history callbacks.

    Parameters
    ----------
    output_dir
        Directory receiving the generated objective-history plot.

    Returns
    -------
    pathlib.Path
        Path of the generated plot.

    """
    problem = BraninProblem(initial_values=(1.0, 1.0), ana_deriv=True)
    problem.initialize(verbosity=0)

    solver = SLSQP(problem, scipy_pars={"tol": 1e-9})
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()
    results = solver.solve(
        verbosity=0,
        callbacks=[IterationOutput(), history],
    )

    output_dir = output_dir or Path(__file__).with_name("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "slsqp_objective_history.png"
    figure = history.plot_objective()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(results)
    print(f"Saved objective history to {plot_path}")
    return plot_path


if __name__ == "__main__":
    main()
