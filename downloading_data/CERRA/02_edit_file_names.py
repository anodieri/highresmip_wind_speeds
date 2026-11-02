# environment: postprocess

# script to change the random files names for CERRA after downloading data from copernicus to something that makes sense
# namely CERRA and the dates

# use:
# conda activate postprocess
# navigate to the location of this file
# python 02_edit_file_names.py path_to_cerra_downloads

# import necessary packages
import sys
import os
import xarray as xr


# the path of the folder where the files are is the first command line argument that's given
# should be the absolute path!
path = sys.argv[1]

# Use a test mode unless overwritten by a second command line argument
test_mode = True
if len(sys.argv)>2:
    if sys.argv[2] == 'False':
        test_mode = False

# Get a list of all files in that folder
dir_list = os.listdir(path)

# Loop over all of the files in the list
for file in dir_list:
    # Only do netcdf files that are fully downloaded
    if '.nc' in file and not 'part' in file: 
        file_path = path+'/'+file

        # Open the dataset
        print(f"Opening file {file_path}")
        ds = xr.open_dataarray(path+'/'+file, engine = 'netcdf4')
        start_date = str(ds.valid_time.data[0]).split('T')[0]
        end_date = str(ds.valid_time.data[-1]).split('T')[0]

        # Construct a new, clearer filename 
        new_filename = path + '/'+ 'CERRA_' + start_date + '_' + end_date + '.nc'
        
        # if the new filename is different from the previous one, save the file under that name
        if new_filename != file_path:
            print(f"Saving new file: {new_filename}")
            if not test_mode: ds.to_netcdf(new_filename, mode="w",engine="netcdf4")

            # Remove the old version of the file
            print(f"Removing old file: {file_path}")
            if not test_mode: os.remove(file_path)

        # Close the netcdf file
        ds.close()

if test_mode:
    print("Test mode was used, so no files were actually altered. Check the filenames printed to see if everything is as expected and rerun the command with a second argument set to 'False' to proceed to actually rename the files.")
