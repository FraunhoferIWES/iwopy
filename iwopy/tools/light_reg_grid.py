from re import sub
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

    def __init__(self, origin, deltas, n_steps):

        self.origin = np.array(origin, dtype=np.float64)
        self.n_steps = np.array(n_steps, dtype=np.int32)
        self.deltas = np.array(deltas, dtype=np.float64)

        self._imem = {}

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

    def get_corner(self, p, subgrid=None):
        """
        Get the lower-left grid corner of a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_sdims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_sdims,)
            or None for all

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner point, shape: (n_sdims,)

        """
        if subgrid is None:
            return self.origin + ((p - self.origin) // self.deltas) * self.deltas
        else:
            o = self.origin[subgrid]
            d = self.deltas[subgrid]
            if len(p) != len(subgrid):
                p = p[subgrid]
            return o + ((p - o) // d) * d

    def get_corners(self, pts, subgrid=None):
        """
        Get the lower-left grid corners of points.

        Parameters
        ----------
        pts : numpy.ndarray
            The points space, shape: (n_pts, n_sdims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_sdims,)
            or None for all

        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner points, shape: (n_pts, n_sdims)

        """
        if subgrid is None:
            o = self.origin[None, :]
            d = self.deltas[None, :]
        else:
            o = self.origin[subgrid][None, :]
            d = self.deltas[subgrid][None, :]
            if pts.shape[1] != len(subgrid):
                pts = pts[:, subgrid]
        return o + ((pts - o) // d) * d

    def get_cell(self, p, subgrid=None):
        """
        Get the grid cell that contains a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_sdims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_sdims,)
            or None for all

        Returns
        -------
        cell : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_sdims, 2)

        """
        n_dims = self.n_dims if subgrid is None else len(subgrid)
        d = self.deltas if subgrid is None else self.deltas[subgrid]

        cell = np.zeros((n_dims, 2), dtype=np.float64)
        cell[:] = self.get_corner(p, subgrid)[:, None]
        cell[:, 1] += d

        return cell

    def get_cells(self, pts, subgrid=None):
        """
        Get the grid cells that contain the given points,
        one cell per point.

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_sdims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_dims,)
            or None for all

        Returns
        -------
        cells : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_pts, n_sdims, 2)

        """
        n_pts = pts.shape[0]
        n_dims = self.n_dims if subgrid is None else len(subgrid)
        d = self.deltas if subgrid is None else self.deltas[subgrid]

        cells = np.zeros((n_pts, n_dims, 2), dtype=np.float64)
        cells[:] = self.get_corners(pts, subgrid)[:, :, None]
        cells[:, :, 1] += d[None, :]

        return cells

    def _get_interp(self, cell0, subgrid, **kwargs):
        """
        Helper function for interpolation object lookup
        """
        mname = "all" if subgrid is None else tuple(subgrid)

        if mname not in self._imem:

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

            self._imem[mname] = RegularGridInterpolator(cell0, idata, **kwargs)

        return self._imem[mname]

    def _error_info(self, p, subgrid):
        """
        Helper for printing information at interpolation error
        """
        n_dims = self.n_dims if subgrid is None else len(subgrid)
        print("GDIM:", list(range(n_dims)) if subgrid is None else subgrid)
        m = ", ".join(self.origin.astype(str).tolist())
        print("GMIN:", f"[{m}]")
        vmax = self.origin + self.n_steps * self.deltas
        m = ", ".join(vmax.astype(str).tolist())
        print("GMAX:", f"[{m}]")
        print("P   :", p)

    def interpolation_coeffs_point(self, p, subgrid=None, **kwargs):
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
            The point, shape: (n_sdims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_sdims,)
            or None for all
        kwargs : dict, optional
            Additional parameters for `RegularGridInterpolator`

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for interpolation,
            shape: (n_gpts, n_sdims)
        coeffs : numpy.ndarray
            The interpolation coefficients, shape: (n_gpts,)

        """
        cell = self.get_cell(p, subgrid)
        p0 = cell[:, 0]
        ocell = cell - p0[:, None]
        n_dims = len(p0)
        pts = p[None, :] - p0[None, :]

        interp = self._get_interp(ocell, subgrid, **kwargs)

        try:
            coeffs = interp(pts)[0]
        except ValueError as e:
            self._error_info(p, subgrid)
            raise e

        gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
        gpts = gpts.reshape(2**n_dims, n_dims)

        sel = np.abs(coeffs) < 1.0e-14
        if np.any(sel):
            gpts = gpts[~sel]
            coeffs = coeffs[~sel]

        return gpts, coeffs

    def _error_infos(self, pts, subgrid):
        """
        Helper for printing information at interpolation error
        """
        n_dims = self.n_dims if subgrid is None else len(subgrid)
        print("GDIM:", list(range(n_dims)) if subgrid is None else subgrid)
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

    def interpolation_coeffs_points(self, pts, subgrid=None, **kwargs):
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
            The points, shape: (n_pts, n_sdims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_sdims,)
            or None for all
        kwargs : dict, optional
            Additional parameters for `RegularGridInterpolator`

        Returns
        -------
        gpts : numpy.ndarray
            The grid points relevant for interpolation,
            shape: (n_pts, n_gpts, n_sdims)
        coeffs : numpy.ndarray
            The interpolation coefficients, shape: (n_pts, n_gpts)

        """
        if subgrid is not None and pts.shape[1] != len(subgrid):
            pts = pts[:, subgrid]
        n_dims = pts.shape[1]

        cells = self.get_cells(pts, subgrid)
        ocell = cells[0] - cells[0, :, 0, None]
        p0 = cells[:, :, 0]
        opts = pts - p0

        interp = self._get_interp(ocell, subgrid, **kwargs)

        try:
            coeffs = interp(opts)
        except ValueError as e:
            self._error_infos(opts, subgrid)
            raise e

        ipts = np.stack(np.meshgrid(*ocell, indexing="ij"), axis=-1)
        ipts = ipts.reshape(2**n_dims, n_dims)

        sel = np.all(np.abs(coeffs) < 1.0e-14, axis=0)
        if np.any(sel):
            ipts = ipts[~sel]
            coeffs = coeffs[:, ~sel]
        
        gpts = p0[:, None] + ipts[None, :]

        return gpts, coeffs
