import numpy as np
from scipy.interpolate import RegularGridInterpolator

class GhostGrid:
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
            The point, shape: (n_p_dims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_p_dims,)
            or None for all
        
        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner point, shape: (n_p_dims,)

        """
        if subgrid is None:
            return self.origin + ( (p - self.origin) // self.deltas ) * self.deltas
        else:
            o = self.origin[subgrid]
            d = self.deltas[subgrid]
            return o + ( (p - o) // d ) * d

    def get_corners(self, pts, subgrid=None):
        """
        Get the lower-left grid corners of points.

        Parameters
        ----------
        pts : numpy.ndarray
            The points space, shape: (n_pts, n_p_dims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_p_dims,)
            or None for all
        
        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner points, shape: (n_pts, n_p_dims)

        """
        if subgrid is None:
            o = self.origin[None, :]
            d = self.deltas[None, :]
        else:
            o = self.origin[subgrid][None, :]
            d = self.deltas[subgrid][None, :]
        return o + ( (pts - o) // d ) * d

    def get_cell(self, p, subgrid=None):
        """
        Get the grid cell that contains a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_p_dims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_p_dims,)
            or None for all
        
        Returns
        -------
        cell : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_p_dims, 2)

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
            The points, shape: (n_pts, n_p_dims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_dims,)
            or None for all
        
        Returns
        -------
        cells : numpy.ndarray
            The min and max values of each dimension. Shape:
            (n_pts, n_p_dims, 2)

        """
        n_pts = pts.shape[0]
        n_dims = self.n_dims if subgrid is None else len(subgrid)
        d = self.deltas if subgrid is None else self.deltas[subgrid]

        cells = np.zeros((n_pts, n_dims, 2), dtype=np.float64)
        cells[:] = self.get_corners(pts, subgrid)[:, :, None]
        cells[:, :, 1] += d[None, :]
        
        return cells
        
    def interpolation_coeffs_point(self, p, subgrid=None, cell=None, **kwargs):
        """
        Get the interpolation coefficients for
        a point.

        Parameters
        ----------
        p : numpy.ndarray
            The point, shape: (n_p_dims,)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_p_dims,)
            or None for all
        cell : numpy.ndarray, optional
            The cell, if already known. Shape: (n_p_dims, 2)
        kwargs : dict, optional
            Additional parameters for `RegularGridInterpolator`
        
        Returns
        -------
        cell : numpy.ndarray
            The points to which the coeffs refer for interpolation, 
            for each dimension. Shape: (n_p_dims, 2)
        coeffs : numpy.ndarray
            The interpolation coefficients wrt the cell points,
            shape: (n_p_dims,)

        """
        if cell is None:
            cell = self.get_cell(p, subgrid)
        
        cdata = np.zeros_like(cell)
        cdata[:, 1] = 1.
        cdata = np.meshgrid(*cdata, indexing="ij")
        cdata = np.stack(cdata, axis=-1)

        interp = RegularGridInterpolator(cell, cdata, **kwargs)

        return cell, interp(p)

    def interpolation_coeffs_points(self, pts, subgrid=None, cells=None, **kwargs):
        """
        Get the interpolation coefficients for
        a set of points.

        Parameters
        ----------
        pts : numpy.ndarray
            The points, shape: (n_pts, n_p_dims)
        subgrid : list of int, optional
            The subgrid dimensions, shape: (n_p_dims,)
            or None for all
        cells : numpy.ndarray, optional
            The cells, if known. Shape: (n_pts, n_p_dims, 2)
        kwargs : dict, optional
            Additional parameters for `RegularGridInterpolator`
        
        Returns
        -------
        cells : numpy.ndarray
            The points to which the coeffs refer for interpolation, 
            for each dimension. Shape: (n_pts, n_p_dims, 2)
        coeffs : numpy.ndarray
            The interpolation coefficients wrt the cell points,
            shape: (n_pts, n_p_dims)

        """
        if cells is None:
            cells = self.get_cells(pts, subgrid)

        ocell = np.zeros_like(cells[0])
        ocell[:, 1] = 1.
        opts = (pts - cells[:, :, 0]) / (cells[:, :, 1] - cells[:, :, 0])

        TODO

        cdata = np.meshgrid(*ocell, indexing="ij")
        cdata = np.stack(cdata, axis=-1)

        interp = RegularGridInterpolator(ocell, cdata, **kwargs)
