# environment: postprocess

# script to merge different ERA5 files together into one file, to be used after using edit_file_names
# use:
# conda activate postprocess
# navigate to the folder in which this file is saved
# python 03_concatenate_files.py path_to_era5_files

# import necessary packages
import sys
import os
import xarray as xr


# the path of the folder where the files are is the first command line argument that's given
# should be the absolute path!
path = sys.argv[1]

# Get a list of all files in that folder
dir_list = os.listdir(path)

# Use a test mode unless overwritten by a second command line argument
test_mode = True
if len(sys.argv)>2:
    if sys.argv[2] == 'False':
        test_mode = False

# the names of the variables, surrounded by _ cause u10 is contained in u100
variables = ['_z_'] # for geopot height on pressure levels
#['_u_', '_v_'] # for wind speed on pressure levels
#['_u10_', '_v10_', '_u100_', '_v100_'] # for wind speed single levels

# loop over the variables
for v in variables:
    prefix = 'ERA5' + v
    var_files = [file for file in os.listdir(path) if prefix in file] # get the paths of the files with this variable
    var_files.sort() # sort the filenames in place
    print ('\nFiles to be concatenated:')
    for file in var_files: print(file)

    var_datasets = [xr.open_dataarray(path + '/' + file) for file in var_files] # open the datasets
    ds_full = xr.concat(var_datasets, dim = 'valid_time') #concatenate them

    print('Saving file: ' + path + '/ERA5' + v + '.nc')
    if not test_mode: ds_full.to_netcdf(path + '/ERA5' + v + '.nc', mode="w", engine="netcdf4")

    print('Removing individual files')
    if not test_mode:
        for file in var_files:
            os.remove(path + '/' + file)

if test_mode: print('No actual saving of the files was done since a test mode was used, add False to the command line argument to actually concatenate. Manually put the files into folders to get separate files for historical and future data')

# Split files into future and historical: put them in two separate folders