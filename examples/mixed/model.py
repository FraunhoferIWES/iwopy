import numpy as np
import matplotlib.pyplot as plt

from iwopy import Problem, Constraint, Objective


class MaxN(Objective):
    def __init__(self, problem):
        super().__init__(
            problem,
            "maxN",
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )

    def n_components(self):
        return 1

    def maximize(self):
        return [True]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        xy, valid = problem_results
        return np.sum(valid)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        xy, valid = problem_results
        return np.sum(valid, axis=(1, 2))[:, None]


class GridProblem(Problem):
    def __init__(self, n_row_max, radius, min_dist, ctol=1e-3):
        super().__init__(name="grid_problem")

        self.n_row_max = n_row_max
        self.radius = float(radius)
        self.min_dist = float(min_dist)
        self.max_dist = 2 * radius

        self.xy = None
        self.valid = None

    def initialize(self, verbosity=1):
        super().initialize(verbosity)
        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_int(self):
        return ["nx", "ny"]

    def initial_values_int(self):
        return [2, 2]

    def min_values_int(self):
        return [1, 1]

    def max_values_int(self):
        return [self.n_row_max, self.n_row_max]

    def var_names_float(self):
        return ["x0", "y0", "dx", "dy", "alpha"]

    def initial_values_float(self):
        return [0.0, 0.0, self.min_dist, self.min_dist, 0.0]

    def min_values_float(self):
        return [-2 * self.radius, -2 * self.radius, self.min_dist, self.min_dist, 0.0]

    def max_values_float(self):
        return [self.radius, self.radius, self.max_dist, self.max_dist, 90.0]

    def apply_individual(self, vars_int, vars_float):

        nx, ny = vars_int
        x0, y0, dx, dy, alpha = vars_float

        a = np.deg2rad(alpha)
        nax = np.array([np.cos(a), np.sin(a), 0.0])
        naz = np.array([0.0, 0.0, 1.0])
        nay = np.cross(naz, nax)

        self.xy = np.zeros((nx, ny, 2))
        self.xy[:] = np.array([x0, y0])[None, None, :]
        self.xy[:] += np.arange(nx)[:, None, None] * dx * nax[None, None, :2]
        self.xy[:] += np.arange(ny)[None, :, None] * dy * nay[None, None, :2]

        self.valid = np.linalg.norm(self.xy, axis=-1) <= self.radius

        return self.xy, self.valid

    def apply_population(self, vars_int, vars_float):

        n_pop = vars_int.shape[0]
        nx = vars_int[:, 0]
        ny = vars_int[:, 1]
        x0 = vars_float[:, 0]
        y0 = vars_float[:, 1]
        dx = vars_float[:, 2]
        dy = vars_float[:, 3]
        alpha = vars_float[:, 4]

        a = np.deg2rad(alpha)
        nax = np.stack([np.cos(a), np.sin(a), np.zeros(a.shape)], axis=-1)
        naz = np.zeros_like(nax)
        naz[:, 2] = 1
        nay = np.cross(naz, nax)

        mx = np.max(nx)
        my = np.max(ny)
        self.xy = np.full((n_pop, mx, my, 2), -2 * self.radius)
        for i in range(n_pop):
            self.xy[i, : nx[i], : ny[i]] = np.array([x0[i], y0[i]])[None, None, :]
            self.xy[i, : nx[i], : ny[i]] += (
                np.arange(nx[i])[:, None, None]
                * dx[i, None, None, None]
                * nax[i, None, None, :2]
            )
            self.xy[i, : nx[i], : ny[i]] += (
                np.arange(ny[i])[None, :, None]
                * dy[i, None, None, None]
                * nay[i, None, None, :2]
            )

        self.valid = np.linalg.norm(self.xy, axis=-1) <= self.radius

        return self.xy, self.valid

    def get_fig(self, xy=None, valid=None):

        if xy is None:
            xy = self.xy
        if valid is None:
            valid = self.valid

        nx, ny = xy.shape[:2]
        xy = xy.reshape(nx * ny, 2)[valid.reshape(nx * ny)]

        fig, ax = plt.subplots()
        ax.scatter(xy[:, 0], xy[:, 1], color="orange")
        ax.add_patch(plt.Circle((0, 0), self.radius, color="darkred", fill=False))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"N = {len(xy)}, min_dist = {self.min_dist}")

        return fig
