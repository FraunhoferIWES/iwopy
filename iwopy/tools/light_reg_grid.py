import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LightRegGrid:
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

    INT_INF = -99999

    def __init__(self, origin, deltas, n_steps, **kwargs):

        self.origin = np.array(origin, dtype=np.float64)
        self.n_steps = np.array(n_steps, dtype=np.int32)
        self.deltas = np.array(deltas, dtype=np.float64)

        ocell = self.get_cell(self.origin) - self.origin[:, None]
        self._interp = self._get_interp(ocell, **kwargs)

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

    def gp2i(self, gp):
        """
        Get the lower-left grid corner indices of a point.

        Parameters
        ----------
        gp : numpy.ndarray
            The point, shape: (n_dims,)

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner point indices, shape: (n_dims,)

        """
        return ((gp - self.origin) // self.deltas).astype(np.int32)

    def i2gp(self, i):
        """
        Translates grid point index to grid point.

        Parameters
        ----------
        i : int 
            The grid point index
        
        Returns
        -------
        gp : numpy.ndarray
            The grid point, shape: (n_dims,)

        """
        return self.origin + i * self.deltas

    def gpts2inds(self, gpts):
        """
        Get the lower-left grid corner indices of points.

        Parameters
        ----------
        gpts : numpy.ndarray
            The grid points, shape: (n_gpts, n_dims)

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner indices, 
            shape: (n_gpts, n_dims)

        """
        o = self.origin[None, :]
        d = self.deltas[None, :]
        return ((gpts - o) // d).astype(np.int32)
    
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
        o = self.origin[None, :]
        d = self.deltas[None, :]
        return o + inds * d

    def get_corner(self, p):
        """
        Get the lower-left grid corner of a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_dims,)

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner point, shape: (n_dims,)

        """
        return self.origin + ((p - self.origin) // self.deltas) * self.deltas

    def get_corners(self, pts):
        """
        Get the lower-left grid corners of points.

        Parameters
        ----------
        pts : numpy.ndarray
            The points space, shape: (n_pts, n_dims)

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner points, shape: (n_pts, n_dims)

        """
        o = self.origin[None, :]
        d = self.deltas[None, :]
        return o + ((pts - o) // d) * d

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
        cell[:] = self.get_corner(p)[:, None]
        cell[:, 1] += self.deltas
        return cell

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
        cells[:] = self.get_corners(pts)[:, :, None]
        cells[:, :, 1] += self.deltas[None, :]
        return cells

    def _error_info(self, p):
        """
        Helper for printing information at interpolation error
        """
        print("GDIM:", list(range(self.n_dims)))
        m = ", ".join(self.origin.astype(str).tolist())
        print("GMIN:", f"[{m}]")
        vmax = self.origin + self.n_steps * self.deltas
        m = ", ".join(vmax.astype(str).tolist())
        print("GMAX:", f"[{m}]")
        print("P   :", p)

    def interpolation_coeffs_point(self, p):
        """
        Get the interpolation coefficients for
        a point.

        Example
        -------
            >>> g = LightRegGrid(...)
            >>> p = ...
            >>> gpts, c = g.interpolation_coeffs_point(p)
            >>> rpts = ... calc results at gpts, shape (n_gpts, x) ...
            >>> ires = np.einsum('gx,g->x', rpts, c)

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
        pts = p[None, :] - p0[None, :]

        try:
            coeffs = self._interp(pts)[0]
        except ValueError as e:
            self._error_info(p)
            raise e

        gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
        gpts = gpts.reshape(2**n_dims, n_dims)

        sel = np.abs(coeffs) < 1.0e-14
        if np.any(sel):
            gpts = gpts[~sel]
            coeffs = coeffs[~sel]

        return gpts, coeffs

    def _error_infos(self, pts):
        """
        Helper for printing information at interpolation error
        """
        print("GDIM:", list(range(self.n_dims)))
        m = ", ".join(self.origin.astype(str).tolist())
        print("GMIN:", f"[{m}]")
        vmax = self.origin + self.n_steps * self.deltas
        m = ", ".join(vmax.astype(str).tolist())
        print("GMAX:", f"[{m}]")
        print("VMIN:", np.min(pts, axis=0))
        print("VMAX:", np.max(pts, axis=0))
        sel = np.any(pts < self.origin[None, :], axis=1)
        if np.any(sel):
            s = np.argwhere(sel)[0][0]
            print(
                f"Found {np.sum(sel)} points blow lower bounds, e.g. point {s}: q = {pts[s]}"
            )
        sel = np.any(pts > vmax[None, :], axis=1)
        if np.any(sel):
            s = np.argwhere(sel)[0][0]
            print(
                f"Found {np.sum(sel)} points above higher bounds, e.g. point {s}: q = {pts[s]}"
            )

    def interpolation_coeffs_points(self, pts):
        """
        Get the interpolation coefficients for a set of points.

        Example
        -------
            >>> g = LightRegGrid(...)
            >>> pts = ...
            >>> gpts, c = g.interpolation_coeffs_points(pts)
            >>> rpts = ... calc results at gpts, shape (n_pts, n_gpts, x) ...
            >>> ires = np.einsum('pgx,pg->px', rpts, c)

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_dims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_dims,)
            or None for all

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_pts, n_gpts, n_dims)
        coeffs : numpy.ndarray
            The interpolation coefficients, shape: 
            (n_pts, n_gpts)

        """
        cells = self.get_cells(pts)
        ocell = cells[0] - cells[0, :, 0, None]
        p0 = cells[:, :, 0]
        opts = pts - p0

        try:
            coeffs = self._interp(opts)
        except ValueError as e:
            self._error_infos(opts)
            raise e

        ipts = np.stack(np.meshgrid(*ocell, indexing="ij"), axis=-1)
        ipts = ipts.reshape(2**self.n_dims, self.n_dims)

        sel = np.all(np.abs(coeffs) < 1.0e-14, axis=0)
        if np.any(sel):
            ipts = ipts[~sel]
            coeffs = coeffs[:, ~sel]
        
        gpts = p0[:, None] + ipts[None, :]

        return gpts, coeffs

    def grad_coeffs_gridpoints(self, inds, vars=None, order=2, orderb=1):
        """
        Calculates the gradient coefficients at grid points.

        Parameters
        ----------
        inds : numpy.ndarray
            The integer grid point indices, shape: 
            (n_inds, n_dims)
        vars : array-like of int, optional
            The variables wrt which to differentiate,
            default is all, shape: (n_vars,)
        order : int
            The finite difference order,
            1 = forward, -1 = backward, 2 = centre
        orderb : int
            The finite difference order at boundary points
        
        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for coeffs,
            shape: (n_inds, n_vars, n_dpts, n_dims)
        coeffs : numpy.ndarray
            The gradient coefficients, shape: 
            (n_inds, n_vars, n_dpts)
        
        """
        # prepare:
        ipts = self.inds2gpts(inds)
        vars = np.arange(self.n_dims, dtype=np.int32) if vars is None else np.array(vars, dtype=np.int32)
        n_vars = len(vars)
        n_inds = len(inds)

        # check indices:
        chk = (inds < 0) | (inds > self.n_points[None, :])
        if np.any(chk):
            chk = np.any(chk, axis=1)
            print("GSIZE:", self.n_steps.tolist())
            raise ValueError(f"Found {np.sum(chk)} indices out of grid bounds: {inds[chk]}")
        
        # find number of finite difference points n_dpts:
        if order not in [-1, 1, 2]:
            raise NotImplementedError(f"Choice 'order = {order}' is not supported, please choose: -1 (backward), 1 (forward), 2 (centre)")
        if orderb not in [1, 2]:
            raise NotImplementedError(f"Choice 'orderb = {orderb}' is not supported, please choose: 1 or 2")
        sel_bleft = inds == 0
        sel_bright = inds == self.n_points[None, :]
        n_dpts = 2
        if np.any(sel_bleft | sel_bright):
            if orderb == 2:
                n_dpts = 3
            s_centre = ~(sel_bleft | sel_bright)
        else:
            s_centre = np.s_[:]

        # initialize output:
        gpts = np.full((n_inds, self.n_dims, n_dpts, self.n_dims), np.nan, dtype=np.float64)
        coeffs = np.zeros((n_inds, self.n_dims, n_dpts), dtype=np.float64)

        # coeffs for left boundary points:
        if np.any(sel_bleft):

            seli = np.any(sel_bleft, axis=1)
            vrs = np.where(sel_bleft)[1]

            if orderb == 1:
                gpts[:, :, 0][sel_bleft] = ipts[seli]


        # resize:
        if n_vars < self.n_dims:
            gpts = gpts[:, vars]
            coeffs = coeffs[:, vars]

        return gpts, coeffs
        
