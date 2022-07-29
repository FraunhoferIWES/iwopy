import numpy as np
from scipy.interpolate import RegularGridInterpolator


class RegularDiscretizationGrid:
    """
    A lightweight regular grid in n dimensions,
    without points storage.

    Parameters
    ----------
    origin : array-like
        The origin point, len: n_dims
    deltas : array-like
        The step sizes, len: n_dims.
    n_steps : array-like
        The number of steps, len: n_dims. Use
        INT_INF for infinite.
    **kwargs : dict, optional
        Additional parameters for `RegularGridInterpolator`

    Attributes
    ----------
    origin : numpy.ndarray
        The origin point, shape: (n_dims,)
    deltas : numpy.ndarray
        The step sizes, shape: (n_dims,)
    n_steps : numpy.ndarray
        The number of steps, shape: (n_dims,)

    """

    INT_INF = -999999
    DIGITS = 12

    def __init__(self, origin, deltas, n_steps, **kwargs):

        self.origin = np.array(origin, dtype=np.float64)
        self.n_steps = np.array(n_steps, dtype=np.int32)
        self.deltas = np.array(deltas, dtype=np.float64)

        self._ocell = np.round(self.get_cell(self.origin) - self.origin[:, None], self.DIGITS)
        self._interp = self._get_interp(self._ocell, **kwargs)

    def _get_interp(self, cell0, **kwargs):
        """
        Helper function for interpolation object creation
        """

        n_dims = len(cell0)

        cdata = np.zeros_like(cell0)
        cdata[:, 1] = 1.0
        cdata = np.meshgrid(*cdata, indexing="ij")

        cshp = list(cdata[0].shape)
        idata = np.zeros(cshp + [2**n_dims], dtype=np.float64)
        isel = np.zeros_like(cdata[0], dtype=np.int32)
        for di in range(n_dims):
            dj = n_dims - 1 - di
            isel[:] += (2**dj * cdata[di]).astype(np.int32)
        np.put_along_axis(idata, isel[..., None], 1, axis=-1)

        return RegularGridInterpolator(cell0, idata, **kwargs)

    @property
    def n_points(self):
        """
        The number of points in each dimension

        Returns
        -------
        numpy.ndarray :
            The number of points in each dimension,
            shape: (n_dims,)

        """
        n = self.n_steps + 1
        n[n == self.INT_INF + 1] = self.INT_INF
        return n

    @property
    def n_dims(self):
        """
        The number of dimensions

        Returns
        -------
        int :
            The number of dimensions

        """
        return len(self.origin)

    @property
    def p_min(self):
        """
        The minimal grid point values

        Returns
        -------
        numpy.ndarray:
            The minimal grid point values,
            shape: (n_dims,)

        """
        m = self.origin.copy()
        m[(self.n_steps == self.INT_INF) & (self.deltas < 0)] = -np.inf
        return m

    @property
    def p_max(self):
        """
        The maximal grid point values

        Returns
        -------
        numpy.ndarray:
            The maximal grid point values,
            shape: (n_dims,)

        """
        m = self.origin + self.n_steps * self.deltas
        m[(self.n_steps == self.INT_INF) & (self.deltas > 0)] = np.inf
        return m

    def print_info(self, spaces=0):
        """
        Prints basic information

        Parameters
        ----------
        spaces : int
            The prepending spaces

        """
        s = "" if spaces == 0 else " " * spaces
        print(f"{s}n_dims  :", self.n_dims)
        print(f"{s}deltas  :", self.deltas.tolist())
        print(f"{s}n_steps :", self.n_steps)
        print(f"{s}n_points:", self.n_points)
        print(f"{s}p_min   :", self.p_min)
        print(f"{s}p_max   :", self.p_max)

    def _error_info(self, p, for_ocell=False):
        """
        Helper for printing information at interpolation error
        """
        print("GDIM:", self.n_points.tolist())
        print("GMIN:", self.p_min.tolist())
        print("GMAX:", self.p_max.tolist())
        if for_ocell:
            cmin = self._ocell[:, 0]
            cmax = self._ocell[:, 1]
            print("CMIN:", cmin.tolist())
            print("CMAX:", cmax.tolist())
            print("Q   :", p)
        else:
            print("P   :", p)

    def _error_infos(self, pts, for_ocell=False):
        """
        Helper for printing information at interpolation error
        """
        print("GDIM:", self.n_points.tolist())
        print("GMIN:", self.p_min.tolist())
        print("GMAX:", self.p_max.tolist())
        if for_ocell:
            cmin = self._ocell[:, 0]
            cmax = self._ocell[:, 1]
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

    def is_gridi(self, inds):
        """
        Checks if grid indices are valid

        Parameters
        ----------
        inds : int
            The grid point indices, shape: (n_dims,)

        Returns
        -------
        bool :
            True if on grid

        """
        sel0 = ~(self.n_steps == self.INT_INF)
        sel = (inds < 0) | (sel0 & (inds >= self.n_points))
        return not np.any(sel)

    def i2gp(self, inds, error=True):
        """
        Translates grid point indices to grid point.

        Parameters
        ----------
        inds : int
            The grid point indices, shape: (n_dims,)
        error : bool
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        gp : numpy.ndarray
            The grid point, shape: (n_dims,)

        """
        if not self.is_gridi(inds):
            if error:
                self.print_info()
                raise ValueError(f"Grind indices {inds} are not on grid")

        return np.round(self.origin + inds * self.deltas, self.DIGITS)

    def find_grid_inds(self, inds):
        """
        Finds indices that are on grid

        Parameters
        ----------
        inds : numpy.ndarray
            The grid point index candidates,
            shape: (n_inds, n_dims)

        Returns
        -------
        sel_grid : numpy.ndarray of bool
            Subset selection of on-grid indices,
            shape: (n_inds, n_dims)

        """
        sel0 = ~(self.n_steps == self.INT_INF)
        sel = (inds < 0) | (sel0[None, :] & (inds >= self.n_points[None, :]))
        return ~sel

    def inds2gpts(self, inds):
        """
        Translates grid point indices to grid points.

        Parameters
        ----------
        inds : array-like
            The integer grid point indices, shape:
            (n_gpts, dims)

        Returns
        -------
        gpts : numpy.ndarray
            The grid points, shape: (n_gpts, n_dims)

        """
        selg = np.all(self.find_grid_inds(inds), axis=1)
        if not np.all(selg):
            selg = np.where(selg)[0]
            raise ValueError(f"Found {len(selg)} indices outside grid, e.g. index {selg[0]}: {inds[selg[0]]}")

        o = self.origin[None, :]
        d = self.deltas[None, :]
        return np.round(o + inds * d, self.DIGITS)

    def _gp2i(self, gp, allow_outer=True, lower_left=False):
        """
        Helper function for indices calculation
        """
        if lower_left:
            inds = np.round((gp - self.origin) / self.deltas, self.DIGITS).astype(np.int32)
        else:
            inds = np.round((gp - self.origin) / self.deltas).astype(np.int32)

        if not allow_outer:
            sel0 = ~(self.n_steps == self.INT_INF)
            sel = sel0 & (inds == self.n_points - 1)
            inds[sel] -= 1

        return inds

    def _gpts2inds(self, gpts, allow_outer=True, lower_left=False):
        """
        Helper function for index calculation
        """
        o = self.origin[None, :]
        d = self.deltas[None, :]

        if lower_left:
            inds = np.round((gpts - o) / d, self.DIGITS).astype(np.int32)
        else:
            inds = np.round((gpts - o) / d).astype(np.int32)

        sel0 = ~(self.n_steps == self.INT_INF)
        if not allow_outer:
            sel = sel0[None, :] & (inds == self.n_points[None, :] - 1)
            inds[sel] -= 1
        
        return inds

    def is_gridpoint(self, p, ret_inds=False):
        """
        Checks if a point is on grid.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)
        ret_inds : bool
            Additionally return indices

        Returns
        -------
        bool :
            True if on grid
        inds : numpy.ndarray, optional
            The grid point indices, shape: (n_dims,)

        """
        inds = self._gp2i(p)
        if not self.is_gridi(inds):
            if ret_inds:
                return False, inds
            return False
        
        p0 = np.round(self.origin + inds * self.deltas, self.DIGITS)
        if ret_inds:
            return np.all(p0 == p), inds
        return np.all(p0 == p)

    def find_gridpoints(self, pts, allow_outer=True, ret_inds=False):
        """
        Finds points that are on grid.

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_dims)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        ret_inds : bool
            Additionally return indices

        Returns
        -------
        sel_grid : numpy.ndarray of bool
            Subset selection of points that are on grid,
            shape: (n_pts, n_dims)
        inds : numpy.ndarray, optional
            The grid point indices, shape: (n_gpts, n_dims)

        """
        inds = self._gpts2inds(pts, allow_outer)
        sel = self.find_grid_inds(inds)

        if np.any(sel):
            o = self.origin[None, :]
            d = self.deltas[None, :]
            p0 = np.round(o + inds * d, self.DIGITS)
            sel = sel & (p0 == pts)

        if ret_inds:
            return sel, inds[np.all(sel, axis=1)]

        return sel

    def all_gridpoints(self, pts, allow_outer=True):
        """
        Checks if all points are on grid.

        Parameters
        ----------
        pts : numpy.ndarray
            The points space, shape: (n_pts, n_dims)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        bool :
            True if all points on grid

        """
        selg = self.find_gridpoints(pts, allow_outer)
        return np.all(selg)

    def in_grid(self, p):
        """
        Checks if a point is located within the grid.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)

        Returns
        -------
        bool :
            True if within grid

        """
        return np.all((p >= self.p_min) & (p <= self.p_max))

    def find_ingrid(self, pts):
        """
        Finds points that are on grid.

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_dims)

        Returns
        -------
        sel_grid : numpy.ndarray of bool
            Subset selection of points that are in grid,
            shape: (n_pts, n_dims)

        """
        return (pts >= self.p_min[None, :]) & (pts <= self.p_max[None, :])

    def gp2i(self, gp, allow_outer=True, error=True):
        """
        Get grid index of a grid point

        Parameters
        ----------
        gp : numpy.ndarray
            The point, shape: (n_dims,)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        error : bool
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner point indices, shape: (n_dims,)

        """
        isgp, inds = self.is_gridpoint(gp, ret_inds=True)

        if isgp:
            if error:
                self._error_info(gp)
                raise KeyError(f"Point gp = {gp} is not on grid")
            return None

        if not self.is_gridi(inds):
            if error:
                self._error_info(gp)
                raise ValueError(f"Point {gp} out of grid")
            return None

        return inds

    def gpts2inds(self, gpts, allow_outer=True, error=True):
        """
        Get grid indices of grid points.

        Parameters
        ----------
        gpts : numpy.ndarray
            The grid points, shape: (n_gpts, n_dims)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner
        error : bool
            Flag for throwing error if off-grid, else
            return None in that case

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner indices,
            shape: (n_gpts, n_dims)

        """
        selg, inds = self.find_gridpoints(gpts, allow_outer, ret_inds=True)

        if not np.all(selg):
            if error:
                self._error_infos(gpts)
                sel = np.where(np.any(~selg, axis=1))[0]
                raise KeyError(f"Found {len(sel)} points not on grid, e.g. point {sel[0]}: {gpts[0]}")
            return None

        return inds

    def get_corner(self, p, allow_outer=True):
        """
        Get the lower-left grid corner of a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner point, shape: (n_dims,)

        """
        if not self.in_grid(p):
            self.print_info()
            raise ValueError(f"Point {p} not in grid")

        inds = self._gp2i(p, allow_outer, lower_left=True)
        if not self.is_gridi(inds):
            self.print_info()
            raise KeyError(f"Grind indices {inds} are not on grid")

        return np.round(self.origin + inds * self.deltas, self.DIGITS)

    def get_corners(self, pts, allow_outer=True):
        """
        Get the lower-left grid corners of points.

        Parameters
        ----------
        pts : numpy.ndarray
            The points space, shape: (n_pts, n_dims)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner points, shape: (n_pts, n_dims)

        """
        selg = self.find_ingrid(pts)
        if not np.all(selg):
            self._error_infos(pts)
            selg = np.where(np.any(~selg), axis=1)[0]
            raise ValueError(f"Found {len(selg)} points out of grid, e.g. point {selg[0]}: {pts[selg[0]]}")

        o = self.origin[None, :]
        d = self.deltas[None, :]
        inds = self._gpts2inds(pts, allow_outer, lower_left=True)

        selg = self.find_grid_inds(inds)
        if not np.all(selg):
            self._error_infos(pts)
            selg = np.where(np.any(~selg), axis=1)[0]
            raise ValueError(f"Found {len(selg)} indices not on grid, e.g. indices {selg[0]}: {inds[selg[0]]}")

        return np.round(o + inds * d, self.DIGITS)

    def get_cell(self, p):
        """
        Get the grid cell that contains a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)

        Returns
        -------
        cell : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_dims, 2)

        """
        cell = np.zeros((self.n_dims, 2), dtype=np.float64)
        cell[:] = self.get_corner(p, allow_outer=False)[:, None]
        cell[:, 1] += self.deltas
        return np.round(cell, self.DIGITS)

    def get_cells(self, pts):
        """
        Get the grid cells that contain the given points,
        one cell per point.

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_dims)

        Returns
        -------
        cells : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_pts, n_dims, 2)

        """
        n_pts = pts.shape[0]
        cells = np.zeros((n_pts, self.n_dims, 2), dtype=np.float64)
        cells[:] = self.get_corners(pts, allow_outer=False)[:, :, None]
        cells[:, :, 1] += self.deltas[None, :]
        return np.round(cells, self.DIGITS)

    def interpolation_coeffs_point(self, p):
        """
        Get the interpolation coefficients for
        a point.

        Example
        -------
            >>> g = RegularDiscretizationGrid(...)
            >>> p = ...
            >>> gpts, c = g.interpolation_coeffs_point(p)
            >>> ratg = ... calc results at gpts, shape (n_gpts, x) ...
            >>> ires = np.einsum('gx,g->x', ratg, c)

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
            The interpolation coefficients, shape: (n_gpts,)

        """
        cell = self.get_cell(p)
        p0 = cell[:, 0]
        n_dims = len(p0)
        pts = np.round(p[None, :] - p0[None, :], self.DIGITS)

        try:
            coeffs = self._interp(pts)[0]
        except ValueError as e:
            self._error_info(p, for_ocell=True)
            raise e

        gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
        gpts = np.round(gpts, self.DIGITS).reshape(2**n_dims, n_dims)

        sel = np.abs(coeffs) < 1.0e-14
        if np.any(sel):
            gpts = gpts[~sel]
            coeffs = coeffs[~sel]

        return gpts, coeffs

    def interpolation_coeffs_points(self, pts, ret_pmap=False):
        """
        Get the interpolation coefficients for a set of points.

        Example
        -------
            >>> g = RegularDiscretizationGrid(...)
            >>> pts = ...
            >>> gpts, c = g.interpolation_coeffs_points(pts)
            >>> ratg = ... calc results at gpts, shape (n_gpts, x) ...
            >>> ires = np.einsum('gx,pg->px', ratg, c)

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_dims)
        ret_pmap : bool
            Additionally return the map from pts to
            gpts

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
            The interpolation coefficients, shape:
            (n_pts, n_gpts)
        pmap : numpy.ndarray, optional
            The map from pts to gpts, shape: (n_pts, n_gp)

        """
        cells = self.get_cells(pts)
        ocell = np.round(cells[0] - cells[0, :, 0, None], self.DIGITS)
        p0 = cells[:, :, 0]

        opts = np.round(pts - p0, self.DIGITS)
        try:
            coeffs = self._interp(opts)  # shape: (n_pts, n_gp)
        except ValueError as e:
            self._error_infos(opts, for_ocell=True)
            raise e

        ipts = np.stack(np.meshgrid(*ocell, indexing="ij"), axis=-1)
        ipts = ipts.reshape(2**self.n_dims, self.n_dims)
        gpts = np.round(p0[:, None] + ipts[None, :], self.DIGITS)  # shape: (n_pts, n_gp, n_dims)

        # remove points with zero weights:
        sel = np.all(np.abs(coeffs) < 1.0e-14, axis=0)
        if np.any(sel):
            ipts = ipts[~sel]
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

    def deriv_coeffs_gridpoints(self, inds, var, order=2, orderb=1):
        """
        Calculates the derivative coefficients at grid points.

        Parameters
        ----------
        inds : numpy.ndarray
            The integer grid point indices, shape:
            (n_inds, n_dims)
        var : int
            The dimension representing the variable
            wrt which to differentiate
        order : int
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb : int
            The finite difference order at boundary points

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
            The gradient coefficients, shape:
            (n_inds, n_gpts)

        """
        # check indices:
        if var < 0 or var > self.n_dims:
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
        gpts = np.round(gpts, self.DIGITS)
        gpts, amap = np.unique(
            gpts.reshape(n_apts, self.n_dims), axis=0, return_inverse=True
        )
        n_gpts = len(gpts)
        amap = amap.reshape(n_pts, n_gp)
        temp = coeffs
        coeffs = np.zeros((n_pts, n_gpts), dtype=np.float64)
        np.put_along_axis(coeffs, amap, temp, axis=1)

        return gpts, coeffs / self.deltas[var]

    def deriv_coeffs(self, pts, var, order=2, orderb=1):
        """
        Calculates the derivative coefficients at points.

        Parameters
        ----------
        pts : numpy.ndarray
            The evaluation points, shape: (n_pts, n_dims)
        var : int
            The dimension representing the variable
            wrt which to differentiate
        order : int
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb : int
            The finite difference order at boundary points

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
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

    def grad_coeffs_gridpoints(self, inds, vars, order=2, orderb=1):
        """
        Calculates the gradient coefficients at grid points.

        Parameters
        ----------
        inds : numpy.ndarray
            The integer grid point indices, shape:
            (n_inds, n_dims)
        vars : list of int, optional
            The dimensions representing the variables
            wrt which to differentiate, shape: (n_vars,).
            Default is all dimensions
        order : int or list of int
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb : int list of int
            The finite difference order at boundary points

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
            The gradient coefficients,
            shape: (n_inds, n_vars, n_gpts)

        """
        if vars is None:
            vars = np.arange(self.n_dims)
        n_vars = len(vars)
        n_inds = len(inds)

        gpts = None
        cfs = None
        sizes = []
        for vi, v in enumerate(vars):

            o = order if isinstance(order, int) else order[vi]
            ob = orderb if isinstance(orderb, int) else orderb[vi]
            hg, hc = self.deriv_coeffs_gridpoints(inds, v, o, ob)

            if gpts is None:
                gpts = hg
                cfs = hc
            else:
                gpts = np.append(gpts, hg, axis=0)
                cfs = np.append(cfs, hc, axis=1)
            sizes.append(len(hg))

            del hg, hc

        gpts, gmap = np.unique(gpts, axis=0, return_inverse=True)
        n_gpts = len(gpts)

        coeffs = np.zeros((n_inds, n_vars, n_gpts), dtype=np.float64)
        i0 = 0
        for vi, s in enumerate(sizes):
            i1 = i0 + s
            np.put_along_axis(coeffs[:, vi], gmap[None, i0:i1], cfs[:, i0:i1], axis=1)
            i0 = i1

        return gpts, coeffs

    def grad_coeffs(self, pts, vars, order=2, orderb=1):
        """
        Calculates the gradient coefficients at grid points.

        Parameters
        ----------
        pts : numpy.ndarray
            The evaluation points, shape: (n_pts, n_dims)
        vars : list of int, optional
            The dimensions representing the variables
            wrt which to differentiate, shape: (n_vars,).
            Default is all dimensions
        order : int list of int
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb : int list of int
            The finite difference order at boundary points

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_gpts, n_dims)
        coeffs : numpy.ndarray
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
