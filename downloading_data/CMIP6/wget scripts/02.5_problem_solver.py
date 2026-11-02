# script to rename files that got 'PROBLEM' in their name from the clipper.py script
# because they were opened while being downloaded
# this script loops over the files with the word PROBLEM every 2 hours, and if the file can
# be opened then, the problem part is removed from the filename
# so that the clipper.py script will clip the non-problematic file during its next iteration


# environment: process_raw_data
# use:
# activate the right conda environment
# run the script in the background with
# nohup python 02.5_problem_solver.py path_to_files False >output.out &
# use the False to get out of the test_mode
# optional: edit waiting time below depending on type of downloads:short for small files that are downloaded quickly, long for larger files

# import necessary packages
import sys
import os
import xarray as xr
import time

# the path of the folder where the files are is the first command line argument that's given
# should be the absolute path!
path = sys.argv[1]

# Use a test mode unless overwritten by a second command line argument
test_mode = True
if len(sys.argv)>2:
    if sys.argv[2] == 'False':
        test_mode = False

# Define a stop criterion: keep looping this script with breaks, to actively clip files as they are being downloaded
# to avoid using enormous amounts of memory
# script will stop when list of files is not updated anymore
done = False
dir_list_previous = []

while not done:

    # Get a list of all files in the folder
    dir_list = os.listdir(path)
    dir_list.sort()

    # Only look at problematic files
    dir_list = [file for file in dir_list if 'PROBLEM' in file]

    # If the list of files is empty, we are likely done and the script is stopped (the following for loop will just do nothing)
    if len(dir_list) == 0: 
        done  = True
        print ('No more problematic files.')

    # Loop over all of the files in the list
    for file in dir_list:

        file_path = path+'/'+file

        # Try to open the file, if it can be opened it is not problematic so can be renamed again
        # this happens when clipper looks at files while they are being downloaded, cannot open them and marks them incorrectly as problematic
        print(f"Opening file {file_path}")
        try:
            ds = xr.open_dataset(file_path, engine = 'netcdf4')
            if not test_mode: os.rename(file_path, file_path.replace('PROBLEM', '.nc'))
            print(f'Problematic file was not problematic, renamed to {file_path.replace('PROBLEM', '.nc')}')
        except (KeyError,  OSError) as e:
            print(f'Problematic file was actually problematic still')
            continue # go to the next file

        # Close the netcdf files
        ds.close()

    # Get the test_mode out of the loop, cause no files are actually altered in the test mode this would results in an endless loop
    if test_mode: done = True

    # If the previous list is the same as the current one, then we assume the downloading has stopped and this script can also be stopped
    if dir_list_previous == dir_list: 
        done = True
        print ('No updates in file list, script stopped')

    # Wait for new files to be downloaded
    if not done:
        print('Waiting ...')
        time.sleep(2*60*60) # 2 hours of waiting time, TODO CAN BE ADAPTED

    # Set the current list of files as the previous one for the next iteration
    dir_list_previous = dir_list

if test_mode:
    print("Test mode was used, so no files were actually altered. Check the filenames printed to see if everything is as expected and rerun the command with a second argument set to 'False' to proceed to clip the files.")
