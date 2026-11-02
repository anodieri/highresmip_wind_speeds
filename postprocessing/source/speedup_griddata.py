# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:36:28 2023

Downloaded from GitHub, function to speed up interpolation from curvilinear grid to rectilinear grid
https://github.com/DuTianshi/Speedup-Griddata

@author: DU
"""

def interp_weights(lon, lat, tlon, tlat, d=2):
    """
    mapping the relationship between the two axis

    Parameters
    ----------
    lon : array_like
        Origin map x. Should have the same size as values.
    lat : array_like
        Origin map y. Should have the same size as values.
    tlon : array_like
        new map x.
    tlat : array_like
        new map y.
    d : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    TYPE
        returns the middle variable of griddata,
        use combine the function: interpolate.
    """
    import numpy as np
    import scipy.spatial.qhull as qhull
    
    def flatten(data):
        if data.ndim == 2:
            data = data.flatten()
        return data
    
    lon, lat, tlon, tlat = map(flatten, [lon, lat, tlon, tlat])
    xy = (lon, lat)
    uv = (tlon, tlat)
    xy = np.array(xy).T
    uv = np.array(uv).T
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, shape=None):
    """
    using the mapping relationship on data

    Parameters
    ----------
    values : float
        the original data. Should have the same size as lon and lat.
    vtx, wts : TYPE
        middle variable of griddata, obtain from the function: interp_weights.
    shape : OPTION
        the shape of new data you need, because griddata need to flatten the 
        original data before use, thus the return data is one-dimensional, 
        this parameters is for reshape of the new data.

    Returns
    -------
    xx : float
        the interpolated data.

    """
    import numpy as np
    
    if values.ndim == 2:
        values = values.flatten()
    data = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    if shape is not None:
        xx = data.reshape(shape)
        return xx
    else:
        return data
