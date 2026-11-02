# Handling CERRA data

## Downloading
Downloading CERRA data can be done using the `01_download_cerra_wind_levels.py` and `01_download_cerra_wind_single.py` for height levels and single levels respectively. Instructions are found at the top of the script.
To adapt to different variables/levels/..., use https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels?tab=download (for height levels, single/model/pressure levels check links https://climate.copernicus.eu/copernicus-regional-reanalysis-europe-cerra). 
On this website the form allows to select the variables and such to download, after which an API request can be created. This can serve as an example to edit the scripts mentioned above.

## Processing
1. After downloading, the filenames are all random strings. To solve this, use the `02_edit_file_names.py` script, which contains instructions on how to run it at the top.
2. To actually process CERRA data, the `03_process_CERRA.py` script can be used. They provide options for the level, number of years, frequency and a test mode. Again, instructions are found at the top of the script. The scripts are quite slow, consider optimizing if this needs to be done repeatedly or for a lot of data.
