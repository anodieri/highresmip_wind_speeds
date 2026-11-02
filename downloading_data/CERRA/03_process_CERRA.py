# script to postprocess CERRA data for 10 m / 100 m
# performs the regridding from a curvilinear grid to a rectilinear grid and cuts out the domain and computes daily means (option)
# takes a few days to run

# kernel: process_raw_data
# use: in terminal with the right kernel activated
# python ./03_process_CERRA.py --height 10/100 --test False --freq 3h/day --select 2025
# test: False to exit test mode
# select: optional, to select only the files with this string

# Add the location of the module to system path
import sys
sys.path.append("postprocessing/source") # TODO EDIT THIS TO PATH WHERE SOURCE FILES ARE LOCATED

# Import the processor module (works now that the location of this file was added to the system path)
import processor as proc

# Import the needed packages
import xarray as xr
import os
import argparse

# Parse the keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument('--select', default = None)
parser.add_argument('--height')
parser.add_argument('--test', default=True)
parser.add_argument('--freq')
args = parser.parse_args()
print(args)

# Set up a test mode except if the argument test is set to False
test_mode = True
if args.test == 'False':
    test_mode = False

# Set up the domain
points_new_domain =  [
    [-6, 59],
    [-4, 48],
    [15, 54],
    [11.5, 60],
    [4, 61]
]
domain = proc.create_domain(points_new_domain, plot= False)

# Create processors for the limited domain without specifying the vertical interpolation method (not needed)
p = proc.Processor(interpolation_method='', alpha = '', domain = domain)

# The height should be 10 or 100, is given as a command line argument
height = args.height

# The frequency should be day or 3h, is given as a command line argument
freq = args.freq
if freq == 'day': daily_mean = True
elif freq == '3h': daily_mean = False

# List the paths to the 10 m CERRA datasets containing wind speed and wind direction
# concatenating these files takes a loooong time (since they are pretty big), so separately processed
# and then the processed data is concatenated
data_path = 'path_to_CERRA_data'+ height + 'm/' # TODO EDIT

paths_wspd = os.listdir( data_path +'wspd')
paths_wspd = [p for p in paths_wspd if '.nc' in p] #ignore other filetypes
paths_wspd.sort() # sort them in place
if args.select is not None: paths_wspd = [p for p in paths_wspd if args.select in p] # select based on string if needed

paths_wdir = os.listdir(data_path + 'wdir')
paths_wdir = [p for p in paths_wdir if '.nc' in p] 
paths_wdir.sort()
if args.select is not None: paths_wdir = [p for p in paths_wdir if args.select in p]

# Print the filenames if the test mode is activated
if test_mode: 
    print('Wind speed files:')
    print(paths_wspd)
    print('Wind direction files')
    print(paths_wdir)
    # and limit the number of files if testing
    paths_wspd = paths_wspd[0:2]
    paths_wdir = paths_wdir[0:2]


# Create an empty list to save the datasets in
datasets = []

# Postprocess each of these datasets
print('Start postprocessing')

for i in range(len(paths_wspd)):
    print(f'Starting {paths_wspd[i]}')
    ds = p.process_cerra(data_path + 'wspd/' + paths_wspd[i], data_path + 'wdir/' + paths_wdir[i], 
        test_mode = test_mode, daily_mean = daily_mean) # calculate the daily means already depending on frequency
    print(f'{paths_wspd[i]} done') # print the progress
    datasets.append(ds)

# Concatenate the datasets in the right order
print('Concatenating files')
ds_full = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts') # drop conflicting attrs: history is different for both files 
    
# Set filename
if args.select is not None: sel = f'_{args.select}'
else: sel = ''
filename = f"path_processed_CERRA/CERRA_{args.height}m_{args.freq}{sel}.nc" # TODO EDIT
# now we'll have a full version containing hist and future, but easy to split afterwards

# Save the processed file as netcdf
print('Saving file')
ds_full.to_netcdf(filename, mode="w",engine="netcdf4") 
print(f'{filename} saved')


if test_mode:
    print("Test mode was used, so only a tiny fraction of CERRA was postprocessed. Check the filenames printed to see if everything is as expected and rerun the command with --test set to False to proceed to actually postprocess CERRA.")
