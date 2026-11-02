The `griddata` function in `scipy.interpolate` offers a way to interpolate data on a grid. However, its performance can be suboptimal when dealing with large datasets due to recalculating grid relationships each time. To address this issue, a separate implementation of the `griddata` function has been developed, inspired by `scipy.interpolate.interp1d` functions.

This alternative implementation divides the `griddata` process into two parts:

1. **Establishing Grid Relationships**: The first step involves establishing the relationships between the old and new grids.
   
2. **Applying Established Relationships**: The second step utilizes these established relationships to interpolate different input data efficiently.

Installation
-------------

To install speedup_griddata, simply clone this repository and run the setup file using the following command:
    
    $ python setup.py install

Dependencies
-------------

speedup_griddata requires the following dependencies to be installed:

"numpy", "scipy"

Usage
-------------

speedup_griddata do the exact same thing as scipy.interpolate.griddata, but faster when interpolating multiple data with the same grid.

Contributing
-------------

Contributions to speedup_griddata are welcome! Please feel free to fork this repository and submit pull requests with your improvements.

Citation
-------------

This software is written by Tianshi DU in 2023 in Qingdao, Shandong, China.
