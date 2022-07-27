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

        self._ocell = np.round(self.get_cell(self.origin) - self.origin[:, None], 14)
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
        s = "" if spaces is 0 else " "*spaces
        print(f"{s}n_dims  :", self.n_dims)
        print(f"{s}n_steps :", self.n_steps)
        print(f"{s}n_points:", self.n_points)
        print(f"{s}p_min   :", self.p_min)
        print(f"{s}p_max   :", self.p_max)

    def gp2i(self, gp, allow_outer=True):
        """
        Get grid index of a grid point

        Parameters
        ----------
        gp : numpy.ndarray
            The point, shape: (n_dims,)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner point indices, shape: (n_dims,)

        """
        inds = ((gp - self.origin) / self.deltas).astype(np.int32)

        sel0 = ~(self.n_steps == self.INT_INF)
        if not allow_outer:
            sel = sel0 & (inds == self.n_points - 1)
            inds[sel] -= 1

        sel = (inds < 0) | (sel0 & (inds >= self.n_points))
        if np.any(sel):
            self._error_info(gp)
            raise ValueError(f"Point {gp} out of grid")

        return inds

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

    def gpts2inds(self, gpts, allow_outer=True):
        """
        Get grid indices of grid points.

        Parameters
        ----------
        gpts : numpy.ndarray
            The grid points, shape: (n_gpts, n_dims)
        allow_outer : bool
            Allow outermost point indices, else
            reduce those to lower-left cell corner

        Returns
        -------
        inds : numpy.ndarray
            The lower-left grid corner indices, 
            shape: (n_gpts, n_dims)

        """
        o = self.origin[None, :]
        d = self.deltas[None, :]
        inds = ((gpts - o) / d).astype(np.int32)

        sel0 = ~(self.n_steps == self.INT_INF)
        if not allow_outer:
            sel = sel0[None, :] & (inds == self.n_points[None, :] - 1)
            inds[sel] -= 1

        sel = (inds < 0) | (sel0[None, :] & (inds >= self.n_points[None, :]))
        if np.any(sel):
            self._error_infos(gpts)
            raise ValueError(f"Found {np.sum(np.any(sel, axis=1))} points out of grid")

        return inds

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
        i = self.gp2i(p, allow_outer)
        return self.origin + i * self.deltas

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
        o = self.origin[None, :]
        d = self.deltas[None, :]
        i = self.gpts2inds(pts, allow_outer)
        return o + i * d

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
        cells[:] = self.get_corners(pts, allow_outer=False)[:, :, None]
        cells[:, :, 1] += self.deltas[None, :]
        return cells

    def _error_info(self, p, for_ocell=False):
        """
        Helper for printing information at interpolation error
        """
        print("GDIM:", self.n_points.tolist())
        print("GMIN:", self.p_min.tolist())
        print("GMAX:", self.p_max.tolist())
        if for_ocell:
            print("CMIN:", np.min(self._cell, axis=0).tolist())
            print("CMAX:", np.max(self._cell, axis=0).tolist())
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
        print("VMIN:", np.min(pts, axis=0))
        print("VMAX:", np.max(pts, axis=0))
        if for_ocell:
            cmin = np.min(self._cell, axis=0)
            cmax = np.max(self._cell, axis=0)
            print("CMIN:", cmin.tolist())
            print("CMAX:", cmax.tolist())
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

    def interpolation_coeffs_point(self, p):
        """
        Get the interpolation coefficients for
        a point.

        Example
        -------
            >>> g = LightRegGrid(...)
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
        pts = np.round(p[None, :] - p0[None, :], 14)

        try:
            coeffs = self._interp(pts)[0]
        except ValueError as e:
            self._error_info(p, for_ocell=True)
            raise e

        gpts = np.stack(np.meshgrid(*cell, indexing="ij"), axis=-1)
        gpts = gpts.reshape(2**n_dims, n_dims)

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
            >>> g = LightRegGrid(...)
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
        ocell = cells[0] - cells[0, :, 0, None]
        p0 = cells[:, :, 0]

        opts = np.round(pts - p0, 14)
        try:
            coeffs = self._interp(opts) # shape: (n_pts, n_gp)
        except ValueError as e:
            print(opts)
            self._error_infos(opts, for_ocell=True)
            raise e

        ipts = np.stack(np.meshgrid(*ocell, indexing="ij"), axis=-1)
        ipts = ipts.reshape(2**self.n_dims, self.n_dims)
        gpts = p0[:, None] + ipts[None, :] # shape: (n_pts, n_gp, n_dims)

        # remove points with zero weights:
        sel = np.all(np.abs(coeffs) < 1.0e-14, axis=0)
        if np.any(sel):
            ipts = ipts[~sel]
            coeffs = coeffs[:, ~sel]
            gpts = gpts[:, ~sel]

        # reorganize data to single grid point array:
        n_pts, n_gp = coeffs.shape
        n_apts = n_pts * n_gp
        gpts, amap = np.unique(gpts.reshape(n_apts, self.n_dims), axis=0, return_inverse=True)
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
            raise ValueError(f"Variable choice '{var}' exceeds dimensions, n_dims = {self.n_dims}")
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
                coeffs[:, 0][sel_bleft] = -1.

                gpts[:, 1][sel_bleft] = ipts[sel_bleft]
                gpts[:, 1, var][sel_bleft] += self.deltas[var]
                coeffs[:, 1][sel_bleft] = 1.

            elif orderb == 2:
                gpts[:, 0][sel_bleft] = ipts[sel_bleft]
                coeffs[:, 0][sel_bleft] = -1.5

                gpts[:, 1][sel_bleft] = ipts[sel_bleft]
                gpts[:, 1, var][sel_bleft] += self.deltas[var]
                coeffs[:, 1][sel_bleft] = 2.

                gpts[:, 2][sel_bleft] = ipts[sel_bleft]
                gpts[:, 2, var][sel_bleft] += 2 * self.deltas[var]
                coeffs[:, 2][sel_bleft] = -0.5

            else:
                raise NotImplementedError(f"Choice 'orderb = {orderb}' is not supported, please choose: 1 or 2")

        # coeffs for right boundary points:
        if np.any(sel_bright):

            if orderb == 1:
                gpts[:, 0][sel_bright] = ipts[sel_bright]
                coeffs[:, 0][sel_bright] = 1.

                gpts[:, 1][sel_bright] = ipts[sel_bright]
                gpts[:, 1, var][sel_bright] -= self.deltas[var]
                coeffs[:, 1][sel_bright] = -1.

            elif orderb == 2:
                gpts[:, 0][sel_bright] = ipts[sel_bright]
                coeffs[:, 0][sel_bright] = 1.5

                gpts[:, 1][sel_bright] = ipts[sel_bright]
                gpts[:, 1, var][sel_bright] -= self.deltas[var]
                coeffs[:, 1][sel_bright] = -2.

                gpts[:, 2][sel_bright] = ipts[sel_bright]
                gpts[:, 2, var][sel_bright] -= 2 * self.deltas[var]
                coeffs[:, 2][sel_bright] = 0.5
                
            else:
                raise NotImplementedError(f"Choice 'orderb = {orderb}' is not supported, please choose: 1 or 2")

        # coeffs for central points:
        if not np.all(sel_bleft | sel_bright):

            if order == 1:
                gpts[:, 0][s_centre] = ipts[s_centre]
                gpts[:, 0, var][s_centre] += self.deltas[var]
                coeffs[:, 0][s_centre] = 1.

                gpts[:, 1][s_centre] = ipts[s_centre]
                coeffs[:, 1][s_centre] = -1.

            elif order == -1:
                gpts[:, 0][s_centre] = ipts[s_centre]
                coeffs[:, 0][s_centre] = 1.

                gpts[:, 1][s_centre] = ipts[s_centre]
                gpts[:, 1, var][s_centre] -= self.deltas[var]
                coeffs[:, 1][s_centre] = -1.

            elif order == 2:
                gpts[:, 0][s_centre] = ipts[s_centre]
                gpts[:, 0, var][s_centre] += self.deltas[var]
                coeffs[:, 0][s_centre] = 0.5

                gpts[:, 1][s_centre] = ipts[s_centre]
                gpts[:, 1, var][s_centre] -= self.deltas[var]
                coeffs[:, 1][s_centre] = -0.5

            else:
                raise NotImplementedError(f"Choice 'order = {order}' is not supported, please choose: -1 (backward), 1 (forward), 2 (centre)")

        # reorganize data to single grid point array:
        n_pts, n_gp = coeffs.shape
        n_apts = n_pts * n_gp
        gpts, amap = np.unique(gpts.reshape(n_apts, self.n_dims), axis=0, return_inverse=True)
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
        np.put_along_axis(pmat, pmap, 1., axis=1)

        coeffs = np.einsum('pi,pi,ig->pg', coeffs0, pmat, coeffs1)

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
        for v in vars:

            hg, hc = self.deriv_coeffs_gridpoints(inds, v, order, orderb)

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
            The gradient coefficients, 
            shape: (n_pts, n_vars, n_gpts)
        
        """
        gpts0, coeffs0, pmap = self.interpolation_coeffs_points(pts, ret_pmap=True)

        inds0 = self.gpts2inds(gpts0, allow_outer=True)
        gpts, coeffs1 = self.grad_coeffs_gridpoints(inds0, vars, order, orderb)

        n_pts = len(pts)
        n_inds0 = len(inds0)
        pmat = np.zeros((n_pts, n_inds0), dtype=np.int8)
        np.put_along_axis(pmat, pmap, 1., axis=1)

        coeffs = np.einsum('pi,pi,ivg->pvg', coeffs0, pmat, coeffs1)

        return gpts, coeffs
