# highresmip_wind_speeds
Code to work with HighResMIP climate model data from CMIP6, analysing the wind speeds in the North Sea. The code can handle vertical and horizontal interpolation, calculate error scores and create various types of plots.

This repository consists of three folders: `downloading_data`, `environments` and `postprocessing`. Below each of these is explained in detail. In general, to get the scripts to work some paths have to be adapted to your personal file system. Wherever this is needed, it is marked with `TODO`.

## Environments
The project is built in Python. The environments folder provides the yml-files detailing the packages and versions used in the code. Different parts use different environments, this is always specified at the top of the file.
For how to use these .yml-files in conda, see https://www.anaconda.com/docs/getting-started/working-with-conda/environments. This folder is mainly to allow fixes if needed, most of the time simply creating a new environment and installing the packages listed in the imports should work.

## Downloading data
As the name suggests, this folder contains the scripts to download data for ERA5, CERRA, KNMI lidar data and HighResMIP-CMIP6 climate model data. Each folder contains a README explaining how to download each type of data.
Downloading of the TNO lidar data does not have it's separate folder, as this is straightforward: see https://offshorewind-measurements.tno.nl/en/data/. 

## Postprocessing
This folder contains the source code to clip the data to the domain, to interpolate vertically and horizontally, to calculate indices or error scores and to create plots. The functions are organised inn classes such as WindDataSet, Validator, PowerCalculator, ... Instances of these classes can easily be created and manipulated in Jupyter notebooks for example.
