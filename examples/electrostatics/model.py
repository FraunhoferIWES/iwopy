import numpy as np
import matplotlib.pyplot as plt

from iwopy import Problem, Constraint, Objective


class MinPotential(Objective):
    def __init__(self, problem, n_charges):
        super().__init__(problem, "potential", vnames_float=problem.var_names_float())
        self.n_charges = n_charges

    def n_components(self):
        return 1

    def maximize(self):
        return [False]

    def calc_individual(self, vars_int, vars_float, problem_results):
        xy = problem_results
        value = 0.0
        for i in range(1, self.n_charges):
            value += 2 * np.sum(1 / np.linalg.norm(xy[i - 1, None] - xy[i:], axis=-1))
        return value

    def calc_population(self, vars_int, vars_float, problem_results):
        xy = problem_results
        n_pop = len(xy)
        value = np.zeros((n_pop, 1))
        for i in range(1, self.n_charges):
            value[:, 0] += 2 * np.sum(
                1 / np.linalg.norm(xy[:, i - 1, None] - xy[:, i:], axis=-1), axis=1
            )
        return value


class MaxRadius(Constraint):
    def __init__(self, problem, n_charges, radius, tol=1e-4):
        super().__init__(
            problem, "radius", vnames_float=problem.var_names_float(), tol=tol
        )
        self.n_charges = n_charges
        self.radius = radius

    def n_components(self):
        return self.n_charges

    def vardeps_float(self):
        deps = np.zeros((self.n_components(), self.n_charges, 2), dtype=bool)
        np.fill_diagonal(deps[..., 0], True)
        np.fill_diagonal(deps[..., 1], True)
        return deps.reshape(self.n_components(), 2 * self.n_charges)

    def calc_individual(self, vars_int, vars_float, problem_results):
        xy = problem_results
        r = np.linalg.norm(xy, axis=-1)
        return r - self.radius

    def calc_population(self, vars_int, vars_float, problem_results):
        xy = problem_results
        r = np.linalg.norm(xy, axis=-1)
        return r - self.radius


class ChargesProblem(Problem):
    def __init__(self, xy_init, radius):
        super().__init__(name="charges_problem")

        self.xy_init = xy_init
        self.n_charges = len(xy_init)
        self.radius = radius

        self.add_objective(MinPotential(self, self.n_charges))
        self.add_constraint(MaxRadius(self, self.n_charges, radius))

    def var_names_float(self):
        vnames = []
        for i in range(self.n_charges):
            vnames += [f"x{i}", f"y{i}"]
        return vnames

    def initial_values_float(self):
        return self.xy_init.reshape(2 * self.n_charges)

    def min_values_float(self):
        return np.full(2 * self.n_charges, -self.radius)

    def max_values_float(self):
        return np.full(2 * self.n_charges, self.radius)

    def apply_individual(self, vars_int, vars_float):
        return vars_float.reshape(self.n_charges, 2)

    def apply_population(self, vars_int, vars_float):
        n_pop = len(vars_float)
        return vars_float.reshape(n_pop, self.n_charges, 2)

    def get_fig(self, xy):
        fig, ax = plt.subplots()
        ax.scatter(xy[:, 0], xy[:, 1], color="orange")
        ax.add_patch(plt.Circle((0, 0), self.radius, color="darkred", fill=False))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"N = {self.n_charges}")
        return fig
