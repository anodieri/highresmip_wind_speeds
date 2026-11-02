# Script to concatenate wind speed files (ua and uv) after downloading them using wget scripts
# environment: process_raw_data

# use:
# activate the correct conda environment
# python 03_concatenate_files.py path_to_wspd_files False 
# False is used to get out of test mode
# path points to the wind speed files

# add the location of the module to system path
import sys
import os
import xarray as xr
import datetime
import cftime
import numpy as np
sys.path.append(".../postprocessing/source") # TODO EDIT THIS TO PATH WHERE SOURCE FILES ARE LOCATED
import processor as proc

# Get the path to where the files are stored as the first keyword argument
path = sys.argv[1] +'/'

# Use a test mode unless overwritten by a second command line argument
test_mode = True
if len(sys.argv)>2:
    if sys.argv[2] == 'False':
        test_mode = False

# Get a list of all files in the folder
dir_list = os.listdir(path)
dir_list.sort()

filename = path + dir_list[0]
# Using the first file, set up the experiment and resolution names
if "EC-Earth" in filename or "CMCC" in filename or "MPI" in filename:
    exps = ['hist-1950', 'highres-future']
    if "EC-Earth" in filename:
        resolutions = ['3P', '3P-HR']
    elif "CMCC" in filename:
        resolutions = ['-VHR4', '-HR4']
    elif "MPI" in filename:
        resolutions = ['HR', 'XR']
elif "FGOALS" in filename:
    exps = ['hist-1950', 'highres-future', 'highresSST-future', 'highresSST-present']
    resolutions = ['-H', '-L']
else:
    exps = ['highresSST-present', 'highresSST-future']
    if "MRI" in filename:
        resolutions = ['-H', '-S']
    elif "HiRAM" in filename:
        resolutions = ['LR', 'HR']

# Check if any are problematic
print('PROBLEMATIC FILES:')
for f in dir_list:
    if 'PROBLEM' in f: 
        print(f"{f} SHOULD BE REDOWNLOADED\n")

# Split into a list of u and v components, only keeping netcdf files
u_files = [f for f in dir_list if 'ua' in f and '.nc' and 'clipped' in f]
v_files = [f for f in dir_list if 'va' in f and '.nc' and 'clipped' in f]

# Check if the path actually points to ua and uv wind speeds
# if yes concatenate them
if len(u_files) != 0:

    # Split into experiments
    groups_u = proc.split_into_simulations(resolutions=resolutions, experiments=exps, path_list= u_files)
    groups_v = proc.split_into_simulations(resolutions=resolutions, experiments= exps, path_list= v_files)

    # Loop over the paths (alternative: just swap u for v in the path for v, might be a more certain way to do this without forgetting files or something)
    for paths_sim_u, paths_sim_v in zip(list(groups_u.values()), list(groups_v.values())):

        # Initialise empty lists to save the u and v datasets for each part
        sim_files_u = []
        sim_files_v = []

        # Loop over the paths for this simulation and add the datasets to the list
        for p_v in paths_sim_v:
            p_u = p_v.replace('va', 'ua')
            counter_u, counter_v = 0,0
            du = xr.open_dataset(path + p_u)
            dv = xr.open_dataset(path + p_v)

            # For FGOALS we have to fix some dates and replace data with NaNs
            # Replace the zero values at timestamps with date 1,1,1 with nans, other values are left in
            if 'FGOALS' in p_u:
                du_masked = du.ua.where((du.time != cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True)) | (du.ua != 0), np.nan)
                dv_masked = dv.va.where((dv.time != cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True)) | (dv.va != 0), np.nan)

                # Copy the timestamps
                time_edited_u = du.time.data
                time_edited_v = dv.time.data

                # Loop over the time dimension and check for 1,1,1 dates, and replace with the next timestamp
                for i in range(len(time_edited_u)):
                    if time_edited_u[i] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True):
                        counter_u = counter_u + 1 #+ np.sum(du.ua.isel(time = i)==0).data # alternative to check numer of nans instead of number of timesteps
                        time_edited_u[i] = time_edited_u[i-1] + datetime.timedelta(days = 1)
                    if time_edited_v[i] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True):
                        counter_v = counter_v + 1 #+ np.sum(dv.va.isel(time = i)==0).data
                        time_edited_v[i] = time_edited_v[i-1] + datetime.timedelta(days = 1)
                
                # Print how many NaNs were introduced at how many timesteps
                if counter_u > 0: print(f'{p_u}: {np.sum(np.isnan(du_masked)).data - np.sum(np.isnan(du.ua)).data} NaNs added replacing zeros at timesteps (1,1,1), {counter_u} timesteps (1,1,1) corrected')
                if counter_v > 0: print(f'{p_v}: {np.sum(np.isnan(dv_masked)).data - np.sum(np.isnan(dv.va)).data} NaNs added replacing zeros at timesteps (1,1,1), {counter_v} timesteps (1,1,1) corrected')
                
                # Replace the time and wind speed components with the corrected values
                du['time'] = time_edited_u
                dv['time'] = time_edited_v
                du['ua'] = du_masked
                dv['va'] = dv_masked
            
            # Ohterwise we somehow get duplicate values because the match is not exact exact
            if 'HiRAM' in p_u:
                du['lat'] = du['lat'].astype('float32')
                du['lon'] = du['lon'].astype('float32')
                dv['lat'] = dv['lat'].astype('float32')
                dv['lon'] = dv['lon'].astype('float32')

            # Check if there are any problems with different dates by comparing length and full equality
            if (len(dv.time.data) != len(du.time.data)) | (len(dv.time.data)!=sum(dv.time.data == du.time.data)):
                print('Corrupted files:')
                print(p_v)
                print(dv.time.data)
                print(p_u)
                print(du.time.data)
            else:
                sim_files_u.append(du)
                sim_files_v.append(dv)
                
            du.close()
            dv.close()
        
        # Concatenate the datasets
        ds_sim_u = xr.concat(sim_files_u, dim = 'time', data_vars = 'all')
        ds_sim_v = xr.concat(sim_files_v, dim = 'time', data_vars = 'all')

        # Put u and v in one file 
        ds_sim = xr.merge([ds_sim_u, ds_sim_v], compat = 'identical')

        # Save the file
        parts = p_u.split('_')
        filename = parts[2] + '_' + parts[4] + '_' + parts[3] + '_day.nc'
        print("Save " + path + filename)
        if not test_mode:
            ds_sim.to_netcdf(path + filename, mode="w",engine="netcdf4")

# in the other case we're working with surface winds
else:

    # Select the clipped netcdf files
    files = [f for f in dir_list if '.nc' in f and 'clipped' in f]

    # Split into experiments
    groups = proc.split_into_simulations(resolutions=resolutions, experiments=exps, path_list= files)
    
    # Loop over the paths (alternative: just swap u for v in the path for v, might be a more certain way to do this without forgetting files or something)
    for paths_sim in list(groups.values()):

        # Initialise empty lists to save the sfcWind datasets
        sim_files = []

        # Loop over the paths for this simulation and add the datasets to the list
        for p in paths_sim:
            ds = xr.open_dataset(path + p)
            sim_files.append(ds)
            ds.close()
            # For FGOALS we have to fix some dates and replace data with NaNs
            # Replace the zero values at timestamps with date 1,1,1 with nans, other values are left in
            counter = 0
            if 'FGOALS' in p:
                ds_masked = ds.sfcWind.where((ds.time != cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True)) | (ds.sfcWind != 0), np.nan)

                # Copy the timestamps
                time_edited = ds.time.data

                # Loop over the time dimension and check for 1,1,1 dates, and replace with the next timestamp
                for i in range(len(time_edited)):
                    if time_edited[i] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True):
                        counter = counter + 1 #+ np.sum(du.ua.isel(time = i)==0).data # alternative to check numer of nans instead of number of timesteps
                        time_edited[i] = time_edited[i-1] + datetime.timedelta(days = 1)
                
                # Print how many NaNs were introduced at how many timesteps
                if counter > 0: print(f'{p}: {np.sum(np.isnan(ds_masked)).data - np.sum(np.isnan(ds.sfcWind)).data} NaNs added replacing zeros at timesteps (1,1,1), {counter} timesteps (1,1,1) corrected')
                
                # Replace the time and wind speed components with the corrected values
                ds['time'] = time_edited
                ds['sfcWind'] = ds_masked
        
        # Concatenate the datasets
        ds_sim = xr.concat(sim_files, dim = 'time', data_vars = 'all')

        # Save the file
        parts = p.split('_')
        filename = parts[0] + '_' + parts[2] + '_' + parts[4] + '_' + parts[3] + '_day.nc'
        print("Save " + path + filename)
        if not test_mode:
            ds_sim.to_netcdf(path + filename, mode="w",engine="netcdf4")

# Print test mode warning
if test_mode:
    print('Test mode was used, so the files were not actually concatenated. Add a second keyword argument set to False to get out of the test mode. ')
else:
    print('Concatenated files were saved. Deletion of the individual files should be done manually.')


