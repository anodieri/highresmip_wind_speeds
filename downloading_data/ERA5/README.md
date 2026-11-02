# Handling ERA5 data

## Downloading
Downloading ERA5 data can be done using the `01_download_era5_wind`. Instructions are found at the top of the script.
To adapt to different variables/levels/..., use e.g. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download (for single levels). 
On this website the form allows to select the variables and such to download, after which an API request can be created. This can serve as an example to edit the scripts mentioned above.

## Preprocessing
1. After downloading, the filenames are all random strings. To solve this, use the `02_edit_file_names.py` script, which contains instructions on how to run it at the top.
2. Next, use the `03_concatenate_files.py` script to concatenate the whole list of files into one file for each variable.
3. Actually processing the ERA5 data is done using the functions from  `processor.py`, typically in a notebook.