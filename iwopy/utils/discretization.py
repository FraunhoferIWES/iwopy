from collections.abc import Sequence
from typing import ClassVar, Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


class RegularDiscretizationGrid:
    """
    A lightweight regular grid in n dimensions,
    without points storage.
    """

    INT_INF: ClassVar[int] = -999999

    def __init__(
        self,
        origin: ArrayLike,
        deltas: ArrayLike,
        n_steps: ArrayLike,
        interpolation: Literal["nearest", "linear"] | None = None,
        tol: ArrayLike | None = None,
        digits: int = 12,
    ) -> None:
        """
        Parameters
        ----------
        origin
            The origin point, len: n_dims
        deltas
            The step sizes, len: n_dims.
        n_steps
            The number of steps, len: n_dims. Use
            INT_INF for infinite.
        interpolation
            The interpolation method: None, nearest, linear
        tol
            The tolerances for grid bounds, default is 0,
            shape: (n_dims,)
        digits
            The grid point precision
        """

        self.origin = np.array(origin, dtype=np.float64)
        self.n_steps = np.array(n_steps, dtype=np.int32)
        self.deltas = np.array(deltas, dtype=np.float64)
        self.digits = digits
        self.interpolation = interpolation

        self.tol: np.ndarray | None
        if tol is not None:
            self.tol = np.zeros(self.n_dims, dtype=np.float64)
            self.tol[:] = tol
        else:
            self.tol = None

        self._opts: np.ndarray | None = None

    @property
    def n_points(self) -> np.ndarray:
        """
        The number of points in each dimension

        Returns
        -------
        n_points
            The number of points in each dimension,
            shape: (n_dims,)
        """
        n = self.n_steps + 1
        n[n == self.INT_INF + 1] = self.INT_INF
        return n

    @property
    def n_dims(self) -> int:
        """
        The number of dimensions

        Returns
        -------
        n_dims
            The number of dimensions
        """
        return len(self.origin)

    @property
    def p_min(self) -> np.ndarray:
        """
        The minimal grid point values

        Returns
        -------
        p_min
            The minimal grid point values,
            shape: (n_dims,)
        """
        m = self.origin.copy()
        m[(self.n_steps == self.INT_INF) & (self.deltas < 0)] = -np.inf
        return m

    @property
    def p_max(self) -> np.ndarray:
        """
        The maximal grid point values

        Returns
        -------
        p_max
            The maximal grid point values,
            shape: (n_dims,)
        """
        m = self.origin + self.n_steps * self.deltas
        m[(self.n_steps == self.INT_INF) & (self.deltas > 0)] = np.inf
        return np.round(m, self.digits)

    def print_info(self, spaces: int = 0) -> None:
        """
        Prints basic information

        Parameters
        ----------
        spaces
            The prepending spaces
        """
        s = "" if spaces == 0 else " " * spaces
        print(f"{s}n_dims  :", self.n_dims)
        print(f"{s}deltas  :", self.deltas.tolist())
        print(f"{s}n_steps :", self.n_steps)
        print(f"{s}n_points:", self.n_points)
        print(f"{s}p_min   :", self.p_min)
        print(f"{s}p_max   :", self.p_max)

    def _error_info(self, p: np.ndarray, for_ocell: bool = False) -> None:
        """Helper for printing information at interpolation error"""
        print("GDIM:", self.n_points.tolist())
        print("GMIN:", self.p_min.tolist())
        print("GMAX:", self.p_max.tolist())
        print("GTOL:", None if self.tol is None else self.tol.tolist())
        if for_ocell:
            cmin = np.zeros(self.n_dims, dtype=np.float64)
            cmax = np.ones(self.n_dims, dtype=np.float64)
            print("CMIN:", cmin.tolist())
            print("CMAX:", cmax.tolist())
            print("Q   :", p)
        else:
            print("P   :", p)

    def _error_infos(self, pts: np.ndarray, for_ocell: bool = False) -> None:
        """Helper for printing information at interpolation error"""
        print("GDIM:", self.n_points.tolist())
        print("GMIN:", self.p_min.tolist())
        print("GMAX:", self.p_max.tolist())
        print("GTOL:", None if self.tol is None else self.tol.tolist())
        if for_ocell:
            cmin = np.zeros(self.n_dims, dtype=np.float64)
            cmax = np.ones(self.n_dims, dtype=np.float64)
            print("CMIN:", cmin.tolist())
            print("CMAX:", cmax.tolist())
            print("VMIN:", np.min(pts, axis=0).tolist())
            print("VMAX:", np.max(pts, axis=0).tolist())
            sel = np.any(pts < cmin[None, :], axis=1)
            if np.any(sel):
                s = np.argwhere(sel)[0][0]
                print(
                    f"Found {np.sum(sel)} coords blow lower bounds, e.g. coord {s}: q = {pts[s]}"
                )
            sel = np.any(pts > cmax[None, :], axis=1)
            if np.any(sel):
                s = np.argwhere(sel)[0][0]
                print(
                    f"Found {np.sum(sel)} coords above higher bounds, e.g. coord {s}: q = {pts[s]}"
                )
        else:
            print("VMIN:", np.min(pts, axis=0))
            print("VMAX:", np.max(pts, axis=0))
            sel = np.any(pts < self.p_min[None, :], axis=1)
            if np.any(sel):
                s = np.argwhere(sel)[0][0]
                print(
                    f"Found {np.sum(sel)} points blow lower bounds, e.g. point {s}: p = {pts[s]}"
                )
            sel = np.any(pts > self.p_max[None, :], axis=1)
            if np.any(sel):
                s = np.argwhere(sel)[0][0]
                print(
                    f"Found {np.sum(sel)} points above higher bounds, e.g. point {s}: p = {pts[s]}"
                )

    def is_gridi(self, inds: np.ndarray) -> bool:
        """
        Checks if grid indices are valid

        Parameters
        ----------
        inds
            The grid point indices, shape: (n_dims,)

        Returns
        -------
        on_grid
            True if on grid
        """
        sel0 = ~(self.n_steps == self.INT_INF)
        sel = (inds < 0) | (sel0 & (inds >= self.n_points))
        return not bool(np.any(sel))

    @overload
    def i2gp(self, inds: np.ndarray, error: Literal[True] = True) -> np.ndarray: ...

    @overload
    def i2gp(self, inds: np.ndarray, error: Literal[False]) -> np.ndarray | None: ...

    @overload
    def i2gp(self, inds: np.ndarray, error: bool) -> np.ndarray | None: ...

    def i2gp(self, inds: np.ndarray, error: bool = True) -> np.ndarray | None:
        """
        Translates grid point indices to grid point.

        Parameters
        ----------
        inds
            The grid point indices, shape: (n_dims,)
        error
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        gp
            The grid point, shape: (n_dims,)
        """
        if not self.is_gridi(inds):
            if error:
                self.print_info()
                raise ValueError(f"Grid indices {inds} are not on grid")
            return None

        return np.round(self.origin + inds * self.deltas, self.digits)

    def find_grid_inds(self, inds: np.ndarray) -> np.ndarray:
        """
        Finds indices that are on grid

        Parameters
        ----------
        inds
            The grid point index candidates,
            shape: (n_inds, n_dims)

        Returns
        -------
        sel_grid
            Subset selection of on-grid indices,
            shape: (n_inds, n_dims)
        """
        sel0 = ~(self.n_steps == self.INT_INF)
        sel = (inds < 0) | (sel0[None, :] & (inds >= self.n_points[None, :]))
        return ~sel

    def inds2gpts(self, inds: np.ndarray) -> np.ndarray:
        """
        Translates grid point indices to grid points.

        Parameters
        ----------
        inds
            The integer grid point indices, shape:
            (n_gpts, dims)

        Returns
        -------
        gpts
            The grid points, shape: (n_gpts, n_dims)
        """
        selg = np.all(self.find_grid_inds(inds), axis=1)
        if not np.all(selg):
            bad = np.where(~selg)[0]
            raise ValueError(
                f"Found {len(bad)} indices outside grid, e.g. index {bad[0]}: {inds[bad[0]]}"
            )

        o = self.origin[None, :]
        d = self.deltas[None, :]
        return np.round(o + inds * d, self.digits)

    def _gp2i(
        self,
        gp: np.ndarray,
        allow_outer: bool = True,
        lower_left: bool = False,
    ) -> np.ndarray:
        """Helper function for indices calculation"""
        if lower_left:
            inds = np.round((gp - self.origin) / self.deltas, self.digits).astype(
                np.int32
            )
        else:
            inds = np.round((gp - self.origin) / self.deltas).astype(np.int32)

        if not allow_outer:
            sel0 = ~(self.n_steps == self.INT_INF)
            sel = sel0 & (inds == self.n_points - 1)
            inds[sel] -= 1

        return inds

    def _gpts2inds(
        self,
        gpts: np.ndarray,
        allow_outer: bool = True,
        lower_left: bool = False,
    ) -> np.ndarray:
        """Helper function for index calculation"""
        o = self.origin[None, :]
        d = self.deltas[None, :]

        if lower_left:
            inds = np.round((gpts - o) / d, self.digits).astype(np.int32)
        else:
            inds = np.round((gpts - o) / d).astype(np.int32)

        sel0 = ~(self.n_steps == self.INT_INF)
        if not allow_outer:
            sel = sel0[None, :] & (inds == self.n_points[None, :] - 1)
            inds[sel] -= 1

        return inds

    def apply_tol(self, p: np.ndarray) -> np.ndarray:
        """
        Get tolerance corrected point

        Parameters
        ----------
        p
            The point, shape: (n_dims,)

        Returns
        -------
        q
            The corrected point, shape: (n_dims,)
        """
        if self.tol is not None:
            q = p.copy()

            dlo = q - self.p_min
            sel = (dlo < 0.0) & (dlo >= -self.tol)
            q[sel] = self.p_min[sel]

            dhi = q - self.p_max
            sel = (dhi > 0.0) & (dhi <= self.tol)
            q[sel] = self.p_max[sel]

            return q
        return p

    def apply_tols(self, pts: np.ndarray) -> np.ndarray:
        """
        Get tolerance corrected points

        Parameters
        ----------
        pts
            The points, shape: (n_pts, n_dims)

        Returns
        -------
        q
            The corrected points, shape: (n_pts, n_dims)
        """
        if self.tol is not None:
            qts = pts.copy()

            dlo = qts - self.p_min[None, :]
            sel = (dlo < 0.0) & (dlo >= -self.tol[None, :])
            if np.any(sel):
                pmin = np.zeros_like(qts)
                pmin[:] = self.p_min[None, :]
                qts[sel] = pmin[sel]

            dhi = qts - self.p_max[None, :]
            sel = (dhi > 0.0) & (dhi <= self.tol[None, :])
            if np.any(sel):
                pmax = np.zeros_like(qts)
                pmax[:] = self.p_max[None, :]
                qts[sel] = pmax[sel]

            return qts
        return pts

    @overload
    def is_gridpoint(
        self,
        p: np.ndarray,
        allow_outer: bool = True,
        ret_inds: Literal[False] = False,
    ) -> bool: ...

    @overload
    def is_gridpoint(
        self, p: np.ndarray, allow_outer: bool, ret_inds: Literal[True]
    ) -> tuple[bool, np.ndarray]: ...

    @overload
    def is_gridpoint(
        self, p: np.ndarray, allow_outer: bool, ret_inds: bool
    ) -> bool | tuple[bool, np.ndarray]: ...

    def is_gridpoint(
        self, p: np.ndarray, allow_outer: bool = True, ret_inds: bool = False
    ) -> bool | tuple[bool, np.ndarray]:
        """
        Checks if a point is on grid.

        Parameters
        ----------
        p
            The point, shape: (n_dims,)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        ret_inds
            Additionally return indices

        Returns
        -------
        on_grid
            True if on grid
        inds
            The grid point indices, shape: (n_dims,)
        """
        p = self.apply_tol(p)

        inds = self._gp2i(p, allow_outer)
        if not self.is_gridi(inds):
            if ret_inds:
                return False, inds
            return False

        p0 = np.round(self.origin + inds * self.deltas, self.digits)
        if ret_inds:
            return bool(np.all(p0 == p)), inds
        return bool(np.all(p0 == p))

    @overload
    def find_gridpoints(
        self,
        pts: np.ndarray,
        allow_outer: bool = True,
        ret_inds: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def find_gridpoints(
        self, pts: np.ndarray, allow_outer: bool, ret_inds: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def find_gridpoints(
        self, pts: np.ndarray, allow_outer: bool, ret_inds: bool
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

    def find_gridpoints(
        self, pts: np.ndarray, allow_outer: bool = True, ret_inds: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Finds points that are on grid.

        Parameters
        ----------
        pts
            The points, shape: (n_pts, n_dims)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        ret_inds
            Additionally return indices

        Returns
        -------
        sel_grid
            Subset selection of points that are on grid,
            shape: (n_pts, n_dims)
        inds
            The grid point indices, shape: (n_gpts, n_dims)
        """
        pts = self.apply_tols(pts)

        inds = self._gpts2inds(pts, allow_outer)
        sel = self.find_grid_inds(inds)

        if np.any(sel):
            o = self.origin[None, :]
            d = self.deltas[None, :]
            p0 = np.round(o + inds * d, self.digits)
            sel = sel & (p0 == pts)

        if ret_inds:
            return sel, inds[np.all(sel, axis=1)]

        return sel

    def all_gridpoints(self, pts: np.ndarray, allow_outer: bool = True) -> bool:
        """
        Checks if all points are on grid.

        Parameters
        ----------
        pts
            The points space, shape: (n_pts, n_dims)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        all_on_grid
            True if all points on grid
        """
        selg = self.find_gridpoints(pts, allow_outer)
        return bool(np.all(selg))

    def in_grid(self, p: np.ndarray) -> bool:
        """
        Checks if a point is located within the grid.

        Parameters
        ----------
        p
            The point, shape: (n_dims,)

        Returns
        -------
        in_grid
            True if within grid
        """
        p = self.apply_tol(p)
        return bool(np.all((p >= self.p_min) & (p <= self.p_max)))

    def find_ingrid(self, pts: np.ndarray) -> np.ndarray:
        """
        Finds points that are on grid.

        Parameters
        ----------
        pts
            The points, shape: (n_pts, n_dims)

        Returns
        -------
        sel_grid
            Subset selection of points that are in grid,
            shape: (n_pts, n_dims)
        """
        pts = self.apply_tols(pts)
        return (pts >= self.p_min[None, :]) & (pts <= self.p_max[None, :])

    @overload
    def gp2i(
        self,
        gp: np.ndarray,
        allow_outer: bool = True,
        error: Literal[True] = True,
    ) -> np.ndarray: ...

    @overload
    def gp2i(
        self, gp: np.ndarray, allow_outer: bool, error: Literal[False]
    ) -> np.ndarray | None: ...

    @overload
    def gp2i(
        self, gp: np.ndarray, allow_outer: bool, error: bool
    ) -> np.ndarray | None: ...

    def gp2i(
        self, gp: np.ndarray, allow_outer: bool = True, error: bool = True
    ) -> np.ndarray | None:
        """
        Get grid index of a grid point

        Parameters
        ----------
        gp
            The point, shape: (n_dims,)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        error
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        inds
            The lower-left grid corner point indices, shape: (n_dims,)
        """
        isgp, inds = self.is_gridpoint(gp, allow_outer, ret_inds=True)

        if not self.is_gridi(inds):
            if error:
                self._error_info(gp)
                raise ValueError(f"Point {gp} out of grid")
            return None

        if not isgp:
            if error:
                self._error_info(gp)
                raise KeyError(f"Point gp = {gp} is not on grid")
            return None

        return inds

    @overload
    def gpts2inds(
        self,
        gpts: np.ndarray,
        allow_outer: bool = True,
        error: Literal[True] = True,
    ) -> np.ndarray: ...

    @overload
    def gpts2inds(
        self, gpts: np.ndarray, allow_outer: bool, error: Literal[False]
    ) -> np.ndarray | None: ...

    @overload
    def gpts2inds(
        self, gpts: np.ndarray, allow_outer: bool, error: bool
    ) -> np.ndarray | None: ...

    def gpts2inds(
        self, gpts: np.ndarray, allow_outer: bool = True, error: bool = True
    ) -> np.ndarray | None:
        """
        Get grid indices of grid points.

        Parameters
        ----------
        gpts
            The grid points, shape: (n_gpts, n_dims)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        error
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        inds
            The lower-left grid corner indices,
            shape: (n_gpts, n_dims)
        """
        selg, inds = self.find_gridpoints(gpts, allow_outer, ret_inds=True)

        if not np.all(selg):
            if error:
                self._error_infos(gpts)
                sel = np.where(np.any(~selg, axis=1))[0]
                raise KeyError(
                    f"Found {len(sel)} points not on grid, e.g. point {sel[0]}: {gpts[sel[0]]}"
                )
            return None

        return inds

    def get_corner(self, p: np.ndarray, allow_outer: bool = True) -> np.ndarray:
        """
        Get the lower-left grid corner of a point.

        Parameters
        ----------
        p
            The point, shape: (n_dims,)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        p0
            The lower-left grid corner point, shape: (n_dims,)
        """
        p = self.apply_tol(p)

        if not self.in_grid(p):
            self.print_info()
            raise ValueError(f"Point {p} not in grid")

        inds = self._gp2i(p, allow_outer, lower_left=True)
        if not self.is_gridi(inds):
            self.print_info()
            raise KeyError(f"Grind indices {inds} are not on grid")

        return np.round(self.origin + inds * self.deltas, self.digits)

    def get_corners(self, pts: np.ndarray, allow_outer: bool = True) -> np.ndarray:
        """
        Get the lower-left grid corners of points.

        Parameters
        ----------
        pts
            The points space, shape: (n_pts, n_dims)
        allow_outer
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        p0
            The lower-left grid corner points, shape: (n_pts, n_dims)
        """
        pts = self.apply_tols(pts)

        selg = self.find_ingrid(pts)
        if not np.all(selg):
            self._error_infos(pts)
            selg = np.where(np.any(~selg, axis=1))[0]
            raise ValueError(
                f"Found {len(selg)} points out of grid, e.g. point {selg[0]}: {pts[selg[0]]}"
            )

        o = self.origin[None, :]
        d = self.deltas[None, :]
        inds = self._gpts2inds(pts, allow_outer, lower_left=True)

        selg = self.find_grid_inds(inds)
        if not np.all(selg):
            self._error_infos(pts)
            selg = np.where(np.any(~selg, axis=1))[0]
            raise ValueError(
                f"Found {len(selg)} indices not on grid, e.g. indices {selg[0]}: {inds[selg[0]]}"
            )

        return np.round(o + inds * d, self.digits)

    def get_cell(self, p: np.ndarray) -> np.ndarray:
        """
        Get the grid cell that contains a point.

        Parameters
        ----------
        p
            The point, shape: (n_dims,)

        Returns
        -------
        cell
            The min and max values of each dimension. Shape:
            (n_dims, 2)
        """
        cell = np.zeros((self.n_dims, 2), dtype=np.float64)
        cell[:] = self.get_corner(p, allow_outer=False)[:, None]
        cell[:, 1] += self.deltas
        return np.round(cell, self.digits)

    def get_cells(self, pts: np.ndarray) -> np.ndarray:
        """
        Get the grid cells that contain the given points,
        one cell per point.

        Parameters
        ----------
        pts
            The points, shape: (n_pts, n_dims)

        Returns
        -------
        cells
            The min and max values of each dimension. Shape:
            (n_pts, n_dims, 2)
        """
        n_pts = pts.shape[0]
        cells = np.zeros((n_pts, self.n_dims, 2), dtype=np.float64)
        cells[:] = self.get_corners(pts, allow_outer=False)[:, :, None]
        cells[:, :, 1] += self.deltas[None, :]
        return np.round(cells, self.digits)

    def _get_opts(self) -> np.ndarray:
        """Helper function that returns unit origin cell points"""
        if self._opts is None:
            ocell = np.zeros((self.n_dims, 2), dtype=np.int8)
            ocell[:, 1] += 1
            self._opts = np.stack(np.meshgrid(*ocell, indexing="ij"), axis=-1)
            self._opts = self._opts.reshape(2**self.n_dims, self.n_dims)
            del ocell
        assert self._opts is not None
        return self._opts

    def _interpolate_ocell(self, qts: np.ndarray) -> np.ndarray:
        """
        Helper function for interpolation weights in
        unit hypercube.

        Classic volume weighting, see e.g. Fig. 2 and Eq. (4) in
        http://dx.doi.org/10.1088/0004-6256/139/2/342
        """
        assert (qts >= 0.0).all(), (
            f"Found coordinates below 0: {qts[np.any(qts < 0.0, axis=-1)].tolist()}"
        )
        assert (qts <= 1.0).all(), (
            f"Found coordinates above 1: {qts[np.any(qts > 1.0, axis=-1)].tolist()}"
        )

        opts = self._get_opts()
        return np.prod(1 - np.abs(qts[:, None] - opts[None, :]), axis=-1)

    def interpolation_coeffs_point(
        self, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolation coefficients for
        a point.

        Examples
        --------
        >>> import numpy as np
        >>> from iwopy.utils import RegularDiscretizationGrid
        >>> grid = RegularDiscretizationGrid(
        ...     [0.0], [1.0], [2], interpolation="linear"
        ... )
        >>> points, coefficients = grid.interpolation_coeffs_point(
        ...     np.array([0.5])
        ... )
        >>> points.tolist()
        [[0.0], [1.0]]
        >>> coefficients.tolist()
        [0.5, 0.5]

        Parameters
        ----------
        p
            The point, shape: (n_dims,)

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The interpolation coefficients, shape: (n_gpts,)
        """
        p = self.apply_tol(p)

        cell = self.get_cell(p)
        p0 = cell[:, 0]

        if self.interpolation is None:
            return p0[None, :], np.ones(1, dtype=np.float64)

        elif self.interpolation == "nearest":
            gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
            gpts = np.round(gpts, self.digits).reshape(2**self.n_dims, self.n_dims)
            dist = np.linalg.norm(gpts - p[None, :], axis=-1)
            return gpts[None, np.argmin(dist)], np.ones(1, dtype=np.float64)

        elif self.interpolation == "linear":
            qts = np.round(
                (p[None, :] - p0[None, :]) / self.deltas[None, :], self.digits
            )
            try:
                coeffs = self._interpolate_ocell(qts)[0]
            except AssertionError:
                self._error_info(p, for_ocell=True)
                raise
            gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
            gpts = np.round(gpts, self.digits).reshape(2**self.n_dims, self.n_dims)

        else:
            raise ValueError(
                f"Unknown interpolation '{self.interpolation}'. Please choose: snap, nearest, linear"
            )

        sel = np.abs(coeffs) < 1.0e-14
        if np.any(sel):
            gpts = gpts[~sel]
            coeffs = coeffs[~sel]

        return gpts, coeffs

    @overload
    def interpolation_coeffs_points(
        self, pts: np.ndarray, ret_pmap: Literal[False] = False
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def interpolation_coeffs_points(
        self, pts: np.ndarray, ret_pmap: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @overload
    def interpolation_coeffs_points(
        self, pts: np.ndarray, ret_pmap: bool
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def interpolation_coeffs_points(
        self, pts: np.ndarray, ret_pmap: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the interpolation coefficients for a set of points.

        Examples
        --------
        >>> import numpy as np
        >>> from iwopy.utils import RegularDiscretizationGrid
        >>> grid = RegularDiscretizationGrid(
        ...     [0.0], [1.0], [2], interpolation="linear"
        ... )
        >>> points, coefficients, point_map = grid.interpolation_coeffs_points(
        ...     np.array([[0.5], [1.5]]), ret_pmap=True
        ... )
        >>> points.tolist()
        [[0.0], [1.0], [2.0]]
        >>> coefficients.tolist()
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        >>> point_map.tolist()
        [[0, 1], [1, 2]]

        Parameters
        ----------
        pts
            The points, shape: (n_pts, n_dims)
        ret_pmap
            Additionally return the map from pts to
            gpts

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The interpolation coefficients, shape:
            (n_pts, n_gpts)
        pmap
            The map from pts to gpts, shape: (n_pts, n_gp)
        """
        pts = self.apply_tols(pts)
        n_pts = len(pts)

        if self.interpolation is None:
            p0 = self.get_corners(pts, allow_outer=True)
            coeffs = np.ones((n_pts, 1), dtype=np.float64)
            gpts = p0[:, None, :]

        elif self.interpolation == "nearest":
            p0 = self.get_corners(pts, allow_outer=True)
            qts = np.round((pts - p0) / self.deltas[None, :], self.digits)
            opts = self._get_opts()
            inds = np.argmin(cdist(qts, opts), axis=-1)
            gpts = np.round(
                p0
                + np.take_along_axis(opts, inds[:, None], axis=0)
                * self.deltas[None, :],
                self.digits,
            )[:, None, :]
            coeffs = np.ones((n_pts, 1), dtype=np.float64)
            del qts, opts, inds

        elif self.interpolation == "linear":
            p0 = self.get_corners(pts, allow_outer=False)
            qts = np.round((pts - p0) / self.deltas[None, :], self.digits)
            try:
                coeffs = self._interpolate_ocell(qts)  # shape: (n_pts, n_gp)
            except AssertionError:
                self._error_infos(qts, for_ocell=True)
                raise
            opts = self._get_opts()
            gpts = np.round(
                p0[:, None] + opts[None, :] * self.deltas[None, :], self.digits
            )  # shape: (n_pts, n_gp, n_dims)
            del p0, qts, opts

        else:
            raise ValueError(
                f"Unknown interpolation '{self.interpolation}'. "
                "Please choose: nearest, linear"
            )

        # remove points with zero weights:
        sel = np.all(np.abs(coeffs) < 1.0e-14, axis=0)
        if np.any(sel):
            coeffs = coeffs[:, ~sel]
            gpts = gpts[:, ~sel]

        # reorganize data to single grid point array:
        n_pts, n_gp = coeffs.shape
        n_apts = n_pts * n_gp
        gpts, amap = np.unique(
            gpts.reshape(n_apts, self.n_dims), axis=0, return_inverse=True
        )
        n_gpts = len(gpts)
        amap = amap.reshape(n_pts, n_gp)
        temp = coeffs
        coeffs = np.zeros((n_pts, n_gpts), dtype=np.float64)
        np.put_along_axis(coeffs, amap, temp, axis=1)

        if ret_pmap:
            return gpts, coeffs, amap

        return gpts, coeffs

    def deriv_coeffs_gridpoints(
        self,
        inds: np.ndarray,
        var: int,
        order: int = 2,
        orderb: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the derivative coefficients at grid points.

        Parameters
        ----------
        inds
            The integer grid point indices, shape:
            (n_inds, n_dims)
        var
            The dimension representing the variable
            wrt which to differentiate
        order
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb
            The finite difference order at boundary points

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The gradient coefficients, shape:
            (n_inds, n_gpts)
        """
        # check indices:
        if var < 0 or var >= self.n_dims:
            raise ValueError(
                f"Variable choice '{var}' exceeds dimensions, n_dims = {self.n_dims}"
            )
        ipts = self.inds2gpts(inds)
        n_inds = len(inds)

        # find number of finite difference points n_gpts:
        sel_bleft = inds[:, var] == 0
        sel_bright = inds[:, var] == self.n_points[var] - 1
        n_gpts = 2
        if np.any(sel_bleft | sel_bright):
            if orderb == 2:
                n_gpts = 3
            s_centre = ~(sel_bleft | sel_bright)
        else:
            s_centre = np.s_[:]

        # initialize output:
        gpts = np.zeros((n_inds, n_gpts, self.n_dims), dtype=np.float64)
        coeffs = np.zeros((n_inds, n_gpts), dtype=np.float64)
        gpts[:] = self.origin[None, None, :]

        # coeffs for left boundary points:
        if np.any(sel_bleft):
            if orderb == 1:
                gpts[:, 0][sel_bleft] = ipts[sel_bleft]
                coeffs[:, 0][sel_bleft] = -1.0

                gpts[:, 1][sel_bleft] = ipts[sel_bleft]
                gpts[:, 1, var][sel_bleft] += self.deltas[var]
                coeffs[:, 1][sel_bleft] = 1.0

            elif orderb == 2:
                gpts[:, 0][sel_bleft] = ipts[sel_bleft]
                coeffs[:, 0][sel_bleft] = -1.5

                gpts[:, 1][sel_bleft] = ipts[sel_bleft]
                gpts[:, 1, var][sel_bleft] += self.deltas[var]
                coeffs[:, 1][sel_bleft] = 2.0

                gpts[:, 2][sel_bleft] = ipts[sel_bleft]
                gpts[:, 2, var][sel_bleft] += 2 * self.deltas[var]
                coeffs[:, 2][sel_bleft] = -0.5

            else:
                raise NotImplementedError(
                    f"Choice 'orderb = {orderb}' is not supported, please choose: 1 or 2"
                )

        # coeffs for right boundary points:
        if np.any(sel_bright):
            if orderb == 1:
                gpts[:, 0][sel_bright] = ipts[sel_bright]
                coeffs[:, 0][sel_bright] = 1.0

                gpts[:, 1][sel_bright] = ipts[sel_bright]
                gpts[:, 1, var][sel_bright] -= self.deltas[var]
                coeffs[:, 1][sel_bright] = -1.0

            elif orderb == 2:
                gpts[:, 0][sel_bright] = ipts[sel_bright]
                coeffs[:, 0][sel_bright] = 1.5

                gpts[:, 1][sel_bright] = ipts[sel_bright]
                gpts[:, 1, var][sel_bright] -= self.deltas[var]
                coeffs[:, 1][sel_bright] = -2.0

                gpts[:, 2][sel_bright] = ipts[sel_bright]
                gpts[:, 2, var][sel_bright] -= 2 * self.deltas[var]
                coeffs[:, 2][sel_bright] = 0.5

            else:
                raise NotImplementedError(
                    f"Choice 'orderb = {orderb}' is not supported, please choose: 1 or 2"
                )

        # coeffs for central points:
        if not np.all(sel_bleft | sel_bright):
            if order == 1:
                gpts[:, 0][s_centre] = ipts[s_centre]
                gpts[:, 0, var][s_centre] += self.deltas[var]
                coeffs[:, 0][s_centre] = 1.0

                gpts[:, 1][s_centre] = ipts[s_centre]
                coeffs[:, 1][s_centre] = -1.0

            elif order == -1:
                gpts[:, 0][s_centre] = ipts[s_centre]
                coeffs[:, 0][s_centre] = 1.0

                gpts[:, 1][s_centre] = ipts[s_centre]
                gpts[:, 1, var][s_centre] -= self.deltas[var]
                coeffs[:, 1][s_centre] = -1.0

            elif order == 2:
                gpts[:, 0][s_centre] = ipts[s_centre]
                gpts[:, 0, var][s_centre] += self.deltas[var]
                coeffs[:, 0][s_centre] = 0.5

                gpts[:, 1][s_centre] = ipts[s_centre]
                gpts[:, 1, var][s_centre] -= self.deltas[var]
                coeffs[:, 1][s_centre] = -0.5

            else:
                raise NotImplementedError(
                    f"Choice 'order = {order}' is not supported, please choose: -1 (backward), 1 (forward), 2 (centre)"
                )

        # reorganize data to single grid point array:
        n_pts, n_gp = coeffs.shape
        n_apts = n_pts * n_gp
        gpts = np.round(gpts, self.digits)
        gpts, amap = np.unique(
            gpts.reshape(n_apts, self.n_dims), axis=0, return_inverse=True
        )
        n_gpts = len(gpts)
        amap = amap.reshape(n_pts, n_gp)
        temp = coeffs
        coeffs = np.zeros((n_pts, n_gpts), dtype=np.float64)
        np.put_along_axis(coeffs, amap, temp, axis=1)

        return gpts, coeffs / self.deltas[var]

    def deriv_coeffs(
        self,
        pts: np.ndarray,
        var: int,
        order: int = 2,
        orderb: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the derivative coefficients at points.

        Parameters
        ----------
        pts
            The evaluation points, shape: (n_pts, n_dims)
        var
            The dimension representing the variable
            wrt which to differentiate
        order
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb
            The finite difference order at boundary points

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The gradient coefficients, shape:
            (n_pts, n_gpts)
        """
        gpts0, coeffs0, pmap = self.interpolation_coeffs_points(pts, ret_pmap=True)

        inds0 = self.gpts2inds(gpts0, allow_outer=True)
        gpts, coeffs1 = self.deriv_coeffs_gridpoints(inds0, var, order, orderb)

        n_pts = len(pts)
        n_inds0 = len(inds0)
        pmat = np.zeros((n_pts, n_inds0), dtype=np.int8)
        np.put_along_axis(pmat, pmap, 1.0, axis=1)

        coeffs = np.einsum("pi,pi,ig->pg", coeffs0, pmat, coeffs1)

        return gpts, coeffs

    def grad_coeffs_gridpoints(
        self,
        inds: np.ndarray,
        vars: Sequence[int] | np.ndarray | None,
        order: int | Sequence[int] | np.ndarray = 2,
        orderb: int | Sequence[int] | np.ndarray = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the gradient coefficients at grid points.

        Parameters
        ----------
        inds
            The integer grid point indices, shape:
            (n_inds, n_dims)
        vars
            The dimensions representing the variables
            wrt which to differentiate, shape: (n_vars,).
            Default is all dimensions
        order
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb
            The finite difference order at boundary points

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The gradient coefficients,
            shape: (n_inds, n_vars, n_gpts)
        """
        if vars is None:
            vars = np.arange(self.n_dims)
        n_vars = len(vars)
        n_inds = len(inds)
        if not n_vars:
            return (
                np.empty((0, self.n_dims), dtype=np.float64),
                np.empty((n_inds, 0, 0), dtype=np.float64),
            )

        gpts: np.ndarray | None = None
        cfs: np.ndarray | None = None
        sizes: list[int] = []
        for vi, v in enumerate(vars):
            o = order if isinstance(order, int) else order[vi]
            ob = orderb if isinstance(orderb, int) else orderb[vi]
            hg, hc = self.deriv_coeffs_gridpoints(inds, int(v), int(o), int(ob))

            if gpts is None:
                gpts = hg
                cfs = hc
            else:
                gpts = np.append(gpts, hg, axis=0)
                cfs = np.append(cfs, hc, axis=1)
            sizes.append(len(hg))

            del hg, hc

        assert gpts is not None
        assert cfs is not None
        gpts, gmap = np.unique(gpts, axis=0, return_inverse=True)
        n_gpts = len(gpts)

        coeffs = np.zeros((n_inds, n_vars, n_gpts), dtype=np.float64)
        i0 = 0
        for vi, s in enumerate(sizes):
            i1 = i0 + s
            np.put_along_axis(coeffs[:, vi], gmap[None, i0:i1], cfs[:, i0:i1], axis=1)
            i0 = i1

        return gpts, coeffs

    def grad_coeffs(
        self,
        pts: np.ndarray,
        vars: Sequence[int] | np.ndarray | None,
        order: int | Sequence[int] | np.ndarray = 2,
        orderb: int | Sequence[int] | np.ndarray = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the gradient coefficients at grid points.

        Parameters
        ----------
        pts
            The evaluation points, shape: (n_pts, n_dims)
        vars
            The dimensions representing the variables
            wrt which to differentiate, shape: (n_vars,).
            Default is all dimensions
        order
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb
            The finite difference order at boundary points

        Returns
        -------
        gpts
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs
            The gradient coefficients,
            shape: (n_pts, n_vars, n_gpts)
        """
        gpts0, coeffs0, pmap = self.interpolation_coeffs_points(pts, ret_pmap=True)

        inds0 = self.gpts2inds(gpts0, allow_outer=True)
        gpts, coeffs1 = self.grad_coeffs_gridpoints(inds0, vars, order, orderb)

        n_pts = len(pts)
        n_inds0 = len(inds0)
        pmat = np.zeros((n_pts, n_inds0), dtype=np.int8)
        np.put_along_axis(pmat, pmap, 1.0, axis=1)

        coeffs = np.einsum("pi,pi,ivg->pvg", coeffs0, pmat, coeffs1)

        return gpts, coeffs
