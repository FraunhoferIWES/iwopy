from pathlib import Path

import matplotlib.pyplot as plt

import iwopy
from iwopy.benchmarks.branin import BraninProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo


class GenerationOutput(iwopy.OptimizerCallback):
    """Print the best objective value in each pymoo generation."""

    def notify(self, data: iwopy.OptimizerCallbackData) -> None:
        """Print the generation size and best objective value."""
        best_objective = data.objs[:, 0].min()
        print(
            f"generation {data.iteration:>2}: "
            f"population = {len(data.objs):>2}, "
            f"best objective = {best_objective:.8f}"
        )


def main(output_dir: Path | None = None, n_generations: int = 20) -> Path:
    """Run pymoo with live output and objective history callbacks.

    Parameters
    ----------
    output_dir
        Directory receiving the generated objective-history plot.
    n_generations
        Number of pymoo generations.

    Returns
    -------
    pathlib.Path
        Path of the generated plot.

    """
    problem = BraninProblem(initial_values=(1.0, 1.0))
    problem.initialize(verbosity=0)

    solver = Optimizer_pymoo(
        problem,
        problem_pars={"vectorize": True},
        algo_pars={"type": "GA", "pop_size": 30, "seed": 42},
        term_pars=("n_gen", n_generations),
    )
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()
    results = solver.solve(
        verbosity=0,
        callbacks=[GenerationOutput(), history],
    )

    output_dir = output_dir or Path(__file__).with_name("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "pymoo_objective_history.png"
    figure = history.plot_objective()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(results)
    print(f"Saved objective history to {plot_path}")
    return plot_path


if __name__ == "__main__":
    main()
