# script to clip the geopotential height files downloaded with the wget scripts
# can be used at the same time as the downloading script: clips all files, waits e.g. an hour, and clips all newly downloaded files and so on
# until no more files to be clipped are found
# important to use a suitable waiting time: too short and no new files will be fully downloaded when restarting, so the loop is ended too early
# but too long and huge amounts of data can be downloaded in the waiting time
# the only files that are not clipped are the ones that are currently being downloaded, these are renamed problem
# have to be clipped afterwards

# environment: process_raw_data
# use:
# activate the right conda environment
# run the script in the background with
# nohup python 02_clipper.py path_to_files False >output.out &
# use the False to get out of the test_mode


# import necessary packages
import sys
import os
import xarray as xr
import time
# add the location of the module to system path
sys.path.append(".../postprocessing/source") # TODO EDIT THIS TO PATH WHERE SOURCE FILES ARE LOCATED
import processor as proc

# the path of the folder where the files are is the first command line argument that's given
# should be the absolute path!
path = sys.argv[1]

# Use a test mode unless overwritten by a second command line argument
test_mode = True
if len(sys.argv)>2:
    if sys.argv[2] == 'False':
        test_mode = False

# construct the domain
points_new_domain =  [
    [-6, 59],
    [-4, 48],
    [15, 54],
    [11.5, 60],
    [4, 61]
]
domain = proc.create_domain(points_new_domain, plot= True)

# Create an empty list to save problematic filenames
problems = []

# Define a stop criterion: keep looping this script with breaks, to actively clip files as they are being downloaded
# to avoid using enormous amounts of memory
# script will stop when list of files is not updated anymore
done = False

while not done:

    # Get a list of all files in the folder
    dir_list = os.listdir(path)
    dir_list.sort()

    # Only do netcdf files that are fully downloaded and have not been clipped yet
    dir_list = [file for file in dir_list if '.nc' in file and not 'part' in file and not 'clipped' in file and not 'PROBLEM' in file]
    #print(*dir_list)

    # If the list of files is empty, we are likely done and the script is stopped (the following for loop will just do nothing)
    if len(dir_list) == 0: 
        done  = True
        print ('No more files to be clipped.')

    # Loop over all of the files in the list
    for file in dir_list:

        file_path = path+'/'+file

        # Open the dataset and get some info out of it (if it exists and is not an empty shell of a download or currently being downloaded, in which case we add it to the problem list)
        print(f"Opening file {file_path}")
        try:
            ds = xr.open_dataset(file_path, engine = 'netcdf4')
        except (KeyError,  OSError) as e:
            if not test_mode: os.rename(file_path, file_path[:-3]+'PROBLEM')
            problems.append(file_path)
            print(f'Problematic file was renamed to {file_path[:-3]} PROBLEM')
            continue # go to the next file

        # clip the dataset: select the pressure levels and clip the dataset
        if 'plev' in ds.coords:
            if 100_000 in ds.plev.data:
                ds_clipped = ds.sel(plev = [100_000, 85_000]) # two lowest pressure levels
            elif 92_500 in ds.plev.data:
                print('No 1000 hPa! Using 925 hPa.') # this is needed for CNRM
                ds_clipped = ds.sel(plev = [92_500, 85_000]) # two lowest pressure levels

        elif 'lev' in ds.coords:
            ds_clipped = ds.sel(lev = [100_000, 85_000]) # two lowest pressure levels

        else:
            ds_clipped = ds # if there are no levels, they can't be clipped

        # remove these vars if they are present, otherwise no error
        ds_clipped = ds_clipped.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds', 'time_bounds'], errors = 'ignore') 

        # Convert the longitudes above 180° to negative ones
        ds_clipped.lon.data[ds_clipped.lon.data>180] = ds_clipped.lon.data[ds_clipped.lon.data>180] -360
        ds_clipped = ds_clipped.sortby(variables='lon') # sort this
        ds_clipped = proc.clip_dataset(ds_clipped, domain.buffer(5)) # add a buffer zone, used when regridding data
        # added some extra buffer compared to processor for safety reasons

        # Construct a filename for the clipped file
        new_filename = path + '/'+ file[:-3] + '_clipped.nc'
        
        # Save the clipped version of the file
        print(f"Saving new file: {new_filename}")
        if not test_mode: 
            ds_clipped.to_netcdf(new_filename, mode="w",engine="netcdf4")

        # Remove the old version of the file
        print(f"Removing old file: {file_path}")
        if not test_mode: os.remove(file_path)

        # Close the netcdf files
        ds.close()
        ds_clipped.close()

    # Get the test_mode out of the loop, cause no files are actually altered in the test mode this would results in an endless loop
    if test_mode: done = True

    # Wait for new files to be downloaded
    if not done:
        print('Waiting ...')
        time.sleep(60*60) # 1 hour of waiting time


print('\nProblematic files:')
print(problems)

if test_mode:
    print("Test mode was used, so no files were actually altered. Check the filenames printed to see if everything is as expected and rerun the command with a second argument set to 'False' to proceed to clip the files.")
