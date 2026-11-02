# Classes to investigate extremes in the wind speed data
# namely as persistent low and high wind speeds

import post_processing as pp
import numpy as np
import xarray as xr
import itertools
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.stats as stats

class ExtremeDataHandler:
    """ Class to load and save datasets containing extremes 
    """

    def __init__(self, path: str = ''): #TODO ADAPT DEFAULT PATH
        """ Create an ExtremeDataHandler object
        path: path to the folder where to save the files
        """
        self.path = path

    def __repr__(self):
        """ How to print an ExtremeDataHandler object
        """
        return f'ExtremeDataHandler object with {self.path} as base path to save the calculated extremes'

    def construct_filename(self, data_type: str, name: str = None, persistence_time: int = None, cut_in: float = -np.inf, cut_out: float = np.inf, relative: bool = False, 
        num_days: bool = False, include_reanalysis: bool = None, spatial: bool = False):
        """ Construct the filename for which the ExtremeCalculator has saved the file based on the properties we want
        data_type: the type of dataset, should be base, barplot, cut_ins, list_ens or weighted
        etc, same as before
        """
        # Construct generic parts of the filename
        rel=''
        if relative: rel = "_relative"
        days = ''
        if num_days: days = '_number_of_days'
        spat = ''
        if spatial: spat = '_spatial'

        # Construct the actual filename depending on the type of data we want to load
        match data_type:
            case 'base':
                filename = f'persistent_{name}_{persistence_time}_days_{cut_in}-{cut_out}{rel}{days}{spat}.nc'
            case 'barplot':
                filename = f'ens_list_{name}_comparison_{persistence_time}_days_{cut_in}-{cut_out}{rel}{days}.npy'
                if name is None: filename = filename.replace('_None_', '_')
            case 'cut_ins':
                filename = f'persistent_{name}_multiple{rel}{days}.nc'
            case 'list_ens':
                filename = f'persistent_list_{name}_{persistence_time}_days_{cut_in}-{cut_out}{rel}{days}.nc'
            case 'weighted':
                reanalysis = '_with_reanalysis'
                if not include_reanalysis: reanalysis = '_without_reanalysis'
                filename = f"{name}_weighted_bias{rel}{days}{reanalysis}.nc"
        return filename
    
    def save_file(self, ds: xr.DataArray, filename: str):
        """Save an xarrray dataset as a netcdf file or a numpy array as .npy file
        ds: the xarraydataset or regular dataset
        filename: the name of the file
        """
        # Save as a netcdf file
        if '.nc' in filename:  
            ds.to_netcdf(self.path + filename, mode="w", engine="netcdf4") 
        # or as a numpy array
        elif '.npy' in filename:
            np.save(self.path + filename, ds) 

        # Print a message
        print(self.path + filename + ' saved')

    def load_data(self, filename: str):
        """ Load a dataset from a file 
        filename: the name of the file to open
        """

        # depending on the extension, this is a netcdf file to be loaded with xarray or just a numpy array
        if '.nc' in filename:
            ds =  xr.open_dataset(self.path + filename)
            if len(ds.data_vars) == 1: # if the dataset has only one variable, it's easier to open it as a dataarray
                ds = xr.open_dataarray(self.path + filename)
        elif '.npy' in filename:
            ds = np.load(self.path + filename)
        return ds
    
    def create_spatial_xr_dataset(self, winddataset: pp.WindDataSet, extreme_values: np.array, persistence_times: np.array, long_name: str):
        """Create an xarray dataset with the spatial dimensions of a winddataset
        winddataset: the winddataset to use the spatial lat and lon dimensions from
        extreme_values: the actual data to use as the variable, as np array
        persistence_times: list of the persistence times to be used
        long_name: the long name of the extreme variable
        """
        # Construct the attributes
        attrs = winddataset.data.attrs
        attrs['long_name'] = long_name

        # Construct the lat and lon dimensions for the datasets
        lat = xr.DataArray(
            data=winddataset.data.lat.data.astype('float32'),
            dims=["lat"],
            coords=dict(
                lat=(["lat"], winddataset.data.lat.data)
            ),
            attrs=dict(
            long_name="latitude",
            units="degrees_north",
            axis="Y"
            ),
        )
        lon = xr.DataArray(
            data=winddataset.data.lon.data.astype('float32'),
            dims="lon",
            coords=dict(
                lon=(["lon"], winddataset.data.lon.data)
            ),
            attrs=dict(
            long_name="longitude",
            units="degrees_east",
            axis="X"
            ),
        )

        # Construct the actual dataset with the extreme values
        ds = xr.Dataset(
            data_vars=dict(
                extreme = (
                    ["persistence_time", "lat","lon"], 
                    extreme_values,
                    attrs
                )
            ),
            coords = dict(
                lon = lon,
                lat = lat,
                persistence_time = persistence_times
                ),
        )

        # Return the xarray dataset
        return ds

    def create_point_xr_dataset(self, winddataset: pp.WindDataSet, extreme_values: np.array, persistence_times: np.array, long_name: str, attrs: dict = None):
        """Create an xarray dataset 
        winddataset: the winddataset to copy some attributes from
        extreme_values: the actual data to use as the variable, as np array
        persistence_times: the persistence_times for which these extremes values are
        long_name: the long name of the extreme variable
        """
        
        # Construct the attributes
        attrs = winddataset.data.attrs
        attrs['long_name'] = long_name

        # Construct the actual dataset with the extreme values
        ds = xr.Dataset(
            data_vars=dict(
                extreme=(
                    ["persistence_time"], 
                    extreme_values,
                    attrs
                )
            ),
            coords=dict(
                persistence_time = persistence_times
                ),
        )

        # Return the xarray dataset
        return ds

class ExtremeCalculator:
    """ Objects that given a winddataset, calculate some sort of extremes of wind speed such as persistent lows and highs, ...)
    """

    def __init__(self, path: str = ''): # TODO ADAPT DEFAULT PATH
        """ Create an ExtremeCalculator object
        path: path to the folder where to save the files
        """
        self.datahandler = ExtremeDataHandler(path)

    def __repr__(self):
        """ How to print an ExtremeCalculator object
        """
        return f'ExtremeCalculator object with {self.datahandler.path} as base path to save the calculated extremes'

    def apply_thresholds(self, winddataset: pp.WindDataSet, cut_in: float = -np.inf, cut_out: float = np.inf):
        """ Select the wind speeds between the given cut in and cut out wind speed, replace other values by nan
        winddataset: the winddataset to apply this operation to
        cut_in: the minimum wind speed needed to be selected
        cut_out: the maximum wind speed that is selected 
        """
        
        # Replace the nans that were already in the dataset by inf, to avoid nans from the dataset being indistinguishable from nans after selecting the wind speeds
        variable = list(winddataset.data.data_vars)[0] # generalise wind_speed, cf, power
        output = winddataset.data.fillna(np.inf)[variable]

        # Put nans where windspeed is outside of active range, while not touching the infs which were nans before
        filtered = output.where((output==np.inf) | ((output>cut_in) & (output<cut_out)))

        return filtered

    def percentage_inactive(self, winddataset: pp.WindDataSet, cut_in: float = -np.inf, cut_out: float = np.inf):
        """Get the percentage of time that the wind speed is outside of the active range at each location and return this as an xarray dataset
        winddataset: the winddataset to apply this to
        cut_in: the minimum wind speed
        cut_out: the maximum wind speed
        """

        # Select the times when the wind speed is in active range
        op = self.apply_thresholds(winddataset, cut_in, cut_out) 

        # Convert this to the percentage of the time that wind speed is outside of active range
        inactive = np.isnan(op).sum(dim = 'time')/len(winddataset.data.time)*100 # summed over time, so spatial dimensions are retained

        # Create a long name for the variable
        long_name = f"percentage of time with wind speeds < {cut_in} or > {cut_out}"

        # Create the xarray dataset
        ds = self.datahandler.create_spatial_xr_dataset(winddataset, inactive.data, long_name)

        return ds

    def get_number_series_nans(self, data: np.array, length: int, num_days: bool = False):
        """ Get the number of occurrences of series of nans in a row with a certain length in the given data
        data: the data (raw wind speed data with nans when there is inactivity, not spatial!)
        length: the number of nans in a row we are looking for
        num_days: should the total number of days be returned (True) or the count of the number of occurrences (False, default)
        """

        # Check first if the full series is nan, to save time
        if all(np.isinf(data)): # nans were converted to inf to distinguish nans present in data from nans by filtering for periods
            occurrences = np.nan
        
        else:
            # Initialise a counter
            occurrences = 0

            # Loop over the data grouped by the value returned by isnan using an iterator
            for key, group in itertools.groupby(data, np.isnan): 
                if key: # if the key is true, we are looking at occurrences of nans (np.isnan returned true)
                    size_group = sum(1 for _ in group)
                    if  size_group >= length: # now check if the length of this sequence of nans is longer than the length we are looking for
                    #if len(list(group)) >= length: # same as previous line, but this one takes longer
                        if num_days:
                            occurrences += size_group # add the length of this sequence to the counter if we want the actual number of days
                        else:
                            occurrences += 1 # if yes, add an occurrence to the counter
            
        # Return the counter
        return occurrences

    def get_number_series_nans_multiple(self, data: xr.DataArray, persistence_times: np.array, num_days: bool = False):
        """ Calculate persistent low/high occurrences for multiple persistence times, more efficient way than looping over the get_number_series_nans functions
        data: data array containing either point or spatial data
        persistence_times: persistence times to calculate
        num_days: if False, return number of persistent events; if True, return total number of timesteps belonging to persistent events
        adapted from own function with llm
        """

        # Convert persistence times to array
        persistence_times = np.asarray(persistence_times, dtype=int)

        # Extract values from data
        values = data.values

        # Put time on axis 0 and flatten all spatial dimensions, but remember the spatial shape to reshape later
        # (time,) -> (time, 1)
        # (time, lat, lon) -> (time, lat*lon)
        if values.ndim == 1:
            spatial_shape = ()
            values_2d = values[:, np.newaxis] # puts time values in the other direction, mainly to have correct shapes later
        else:
            spatial_shape = values.shape[1:]
            values_2d = values.reshape(values.shape[0], -1) # assumes time is the first dimension, -1: other dimension is chosen automatically

        n_times = values_2d.shape[0] # total number of timesteps
        n_locations = values_2d.shape[1] # total number of locations
        n_persistence = len(persistence_times)

        # Prepare empty np.array to store results for each persistence time and location
        results = np.zeros((n_persistence, n_locations), dtype=float)

        # Keep track of the current length of the filtered run at every location
        run_length = np.zeros(n_locations, dtype=np.int32)

        # Keep track of locations with at least one valid observation
        # inf indicates an originally missing value
        has_valid_data = np.zeros(n_locations, dtype=bool)

        # Traverse the time dimension once
        for t in range(n_times):

            # Some book keeping
            values_t = values_2d[t]  # Get the values at this timestep
            is_original_nan = np.isinf(values_t) # Original NaNs were replaced by inf
            is_filtered = np.isnan(values_t) # NaNs at this stage were introduced by thresholding

            # Keep track of locations containing valid observations
            has_valid_data |= ~is_original_nan #becomes true where has_valid_data is true or is_original_nan is false, remains true from the moment there is one valid timestep in the series

            # Extend existing filtered runs
            run_length[is_filtered] += 1 # increase length were the value was filtered out

            # Everything that is not filtered terminates a run: normal observations and originally missing data
            run_ended = ~is_filtered

            # if at any location, the run has ended, deal with it
            if np.any(run_ended):

                ended_lengths = run_length[run_ended] # store length of ended runs

                # Only process locations where a run actually existed
                had_run = ended_lengths > 0

                # if such locations exist, updates are needed
                if np.any(had_run):

                    ended_locations = np.flatnonzero(run_ended) # indices of non_zero elements in run ended
                    ended_locations = ended_locations[had_run]# select the ones that actually had a run, works because had run originates form ended_lengths which has the size of run_ended
                    ended_lengths = ended_lengths[had_run]

                    # Check all persistence times for the completed runs
                    for p, persistence_time in enumerate(persistence_times):

                        qualifying = ended_lengths >= persistence_time
                        if num_days:
                            results[p, ended_locations[qualifying]] += ended_lengths[qualifying]
                        else:
                            results[p, ended_locations[qualifying]] += 1

            # Reset locations where the run ended
            run_length[run_ended] = 0

        # Deal with runs that continue until the end of the dataset
        run_still_active = run_length > 0
        if np.any(run_still_active):

            locations = np.flatnonzero(run_still_active)
            ended_lengths = run_length[locations]

            for p, persistence_time in enumerate(persistence_times):

                qualifying = ended_lengths >= persistence_time

                if num_days:
                    results[p, locations[qualifying]] += ended_lengths[qualifying]
                else:
                    results[p, locations[qualifying]] += 1

        # Locations where all observations were originally NaN should return NaN
        results[:, ~has_valid_data] = np.nan

        # Restore the original spatial dimensions if original data was spatial
        if spatial_shape:
            return results.reshape((n_persistence,) + spatial_shape)

        return results[:, 0] # reshape multiple persistence times nicely

    def get_number_per_year(self, winddataset: pp.WindDataSet, extremes: np.array):
        """ Calculate the relative number of occurrences per year
        winddataset: the winddataset from which the extremes were calculated
        extremes: the actual extremes
        """

        # Get the name of the variable to generalise
        variable = list(winddataset.data.data_vars)[0]

        # Get number of non nan days divided by number of days in a year
        nyears = sum(~np.isnan(winddataset.data[variable].data)) / 365 
        # len(winddataset.data.time)/365 # alternative: count all days, 
        # for simplicity all years 365 days, does not matter that much

        return extremes/nyears # this works on spatial as well, if the number of non nan timesteps is 0 then occ is also nan so no problem

    def get_persistent_low_high(self, winddataset: pp.WindDataSet, persistence_times: list, cut_in: float = -np.inf, cut_out: float = np.inf, relative: bool = False, num_days: bool = False, spatial: bool = False, save: bool = False):
        """ Get the number of occurrences of persistent lows or highs or combined events for a range of persistence times, 
        winddataset: the winddataset to apply this to
        persistence_times: the number of timesteps in a row an event needs to persist
        cut_in: the minimum wind speed
        cut_out: the maximum wind speed
        relative: should the absolute or the number of occurrences per year be calculated
        num_days: should the total number of days be returned (True) or the count of the number of occurrences (False, default)
        spatial: should spatial extremes be returned in case of a spatial winddatset? both types are saved by default
        save: should the data be saved in a netcdf file?
        """

        # Select the timesteps where the wind speed is outside of the range
        ops = self.apply_thresholds(winddataset, cut_in, cut_out)

        # Get the name of the variable to generalise
        variable = list(winddataset.data.data_vars)[0]

        # Calculate all persistence times during one traversal of the time series
        occ = self.get_number_series_nans_multiple(ops, persistence_times=persistence_times,num_days=num_days)

        # Set the long name for the variable
        if num_days: unit = 'days'
        else: unit = 'occurrences'
        long_name = f"number of {unit} {variable.replace('_', ' ') } < {cut_in} or > {cut_out} of >= {persistence_times} timesteps (!not per year)"
        long_name_rel = long_name.replace('(!not per year)', 'per year')

        # If the winddataset is spatial:
        if isinstance(winddataset, pp.SpatialWindDataSet):

            # Get the number of years and calculate the relative number per year
            occ_rel = np.array([self.get_number_per_year(winddataset, occ[p]) for p in range(len(persistence_times))]) 

            # Create spatial datasets for rel and absolute number for intermediate saving
            ds_spatial = self.datahandler.create_spatial_xr_dataset(winddataset, occ, persistence_times, long_name) 
            ds_spatial_rel = self.datahandler.create_spatial_xr_dataset(winddataset, occ_rel, persistence_times, long_name_rel)
            
            # Create final dataset
            if relative:
                ds = ds_spatial_rel.mean(dim = ["lat", 'lon'], skipna = True)
            else:
                ds = ds_spatial.mean(dim = ["lat", 'lon'], skipna = True)

        # If the winddataset is not spatial but a point dataset:
        elif isinstance(winddataset, pp.PointWindDataSet):
            
            # Get the number of years and calculate the relative number per year if needed
            if relative:
                occ_rel = np.array([self.get_number_per_year(winddataset, occ[p]) for p in range(len(persistence_times))]) 

                # Create the dataset
                ds = self.datahandler.create_point_xr_dataset(winddataset, occ_rel, persistence_times, long_name_rel)
            else:
                ds = self.datahandler.create_point_xr_dataset(winddataset, occ, persistence_times, long_name)

        # If needed, save this as a netcdf file (to avoid having to wait)
        if save:
            # Construct filename
            if len(persistence_times) > 1:  per = 'multiple'
            else: per =  persistence_times.item()
            filename = self.datahandler.construct_filename(data_type = 'base', name = winddataset.name, persistence_time = per, cut_in=cut_in,
                cut_out = cut_out, relative = relative, num_days = num_days, spatial = False)
            
            # Save the file
            self.datahandler.save_file(ds, filename) 

            # Also save the spatial dataset for relative and absolute 
            if isinstance(winddataset,pp.SpatialWindDataSet):
                filename_spatial_abs = self.datahandler.construct_filename(data_type = 'base', name = winddataset.name, persistence_time = per, cut_in=cut_in,
                    cut_out = cut_out, relative = False, num_days = num_days, spatial = True)
                self.datahandler.save_file(ds_spatial, filename_spatial_abs)
                filename_spatial_rel = self.datahandler.construct_filename(data_type = 'base', name = winddataset.name, persistence_time = per, cut_in=cut_in,
                    cut_out = cut_out, relative = True, num_days = num_days, spatial = True)
                self.datahandler.save_file(ds_spatial_rel, filename_spatial_rel)
        
        # Return the spatial dataset if needed, otherwise the regular one
        if isinstance(winddataset,pp.SpatialWindDataSet) & spatial:
            return ds_spatial
        else: return ds

    def get_persistent_low_high_multiple_cut_ins(self, winddataset: pp.PointWindDataSet, persistence_times: np.array, cut_ins: np.array, relative: bool = False, num_days: bool = False, save: bool = False, save_intermediate: bool = False):
        """Get the number of occurrences of persistent lows for different persistence times and cut ins per year in one pointwinddataset and save it in one xarray dataarray 
        winddataset: the winddataset to apply this to, should be a pointwinddataset
        persistence_times: the different persistence times
        cut_in: the different minimums
        relative: should the number per year or the absolute number be saved
        num_days: should the total number of days be returned (True) or the count of the number of occurrences (False, default)
        save: should the data be saved in a netcdf file? 
        save_intermediate: should intermediate files for each duration and cutin be saved? Mainly useful for spatial datasets, to also save a spatial version
        """

        # Create an empty array to put the data in
        extremes = np.empty(shape = (len(persistence_times), len(cut_ins)))

        # Loop over the cut_ins? calculating the number of occurences
        for c in range(len(cut_ins)):
            extremes[:,c] = self.get_persistent_low_high(winddataset = winddataset, persistence_times = persistence_times, relative=relative,
                cut_in = cut_ins[c], num_days = num_days, save = save_intermediate).extreme.data

        # Create a DataArray saving the extremes
        ds = xr.DataArray(data = extremes,
            dims = ['duration', 'cut_in'], 
            coords = dict(duration=persistence_times, cut_in = cut_ins))

        # If needed, save this as a netcdf file (to avoid having to wait)
        if save:
            filename = self.datahandler.construct_filename(data_type='cut_ins', name=winddataset.name, relative=relative, num_days=num_days)
            self.datahandler.save_file(ds, filename) 

        return ds

    def get_persistent_low_high_list_ensembles(self, ens_list: list, persistence_times: list, cut_in: float = - np.inf, cut_out: float = np.inf, name: str = None, 
        num_days:bool = False, save: bool = False):
        """Get the number of occurrences of persistent lows for a list of ensembles and one cut in and a list of persistence times, both relative and absolute and save it in two xarray dataarrays 
        ens_list: list of ensembles to apply this to
        persistence_times: the different persistence times
        cut_in: the different minimums
        num_days: should the total number of days be returned (True) or the count of the number of occurrences (False, default)
        save: should the data be saved in a netcdf file? 
        """

        # Create an empty arrays to put the data in, also save the number of timesteps for the statistical tests afterwards
        n_persistence = len(persistence_times)
        n_ensembles = len(ens_list)
        n_models = len(ens_list[0].data_list)

        extremes = np.empty((n_persistence, n_ensembles, n_models), dtype=float)
        extremes_relative = np.empty_like(extremes)
        lengths = np.empty((n_ensembles, n_models), dtype=float)

        # Loop over the ensembles and ensemble members, calculating the number of occurences
        for e, ens in enumerate(ens_list):
            for w, winddataset in enumerate(ens.data_list):
                
                print(f"Processing ensemble {e + 1}/{n_ensembles}, model {w + 1}/{n_models}, cut_in={cut_in}")

                # Add the name of the ensemble to the winddatasets name so that it is clear from the name which ens (hist/fut) intermediately saved files are from
                winddataset.name = winddataset.name + '_' + ens.name
                                
                # Calculate the number of extremes, spatially for spatial datasets
                extremes_spatial_abs = self.get_persistent_low_high(winddataset = winddataset, persistence_times = persistence_times, relative=False,
                    cut_in = cut_in, cut_out = cut_out, num_days = num_days, spatial = True, save = save)
                
                # Remove ensemble name again (otherwise double in naming files for the ensemble)
                winddataset.name = winddataset.name.split('_')[0]

                # Get the mean number of occurrences across the full domain, for pointdatasets this does nothing
                if isinstance(winddataset,pp.SpatialWindDataSet):
                    extremes[:, e, w] = (extremes_spatial_abs.extreme.mean(dim=["lat", "lon"],skipna=True).values)
                else:
                    extremes[:, e, w] = (extremes_spatial_abs.extreme.values)                
                
                # Calculate the relative number per year from the already calculated occurrence counts
                for p in range(n_persistence):
                    absolute_occ = extremes_spatial_abs.isel(persistence_time=p)
                    relative_occ = self.get_number_per_year(winddataset,absolute_occ)
                    if isinstance(winddataset, pp.SpatialWindDataSet):
                        extremes_relative[p, e, w] = (relative_occ.extreme.mean(dim=["lat", "lon"], skipna=True).item())
                    else:
                        extremes_relative[p, e, w] = (relative_occ.extreme.item())

                # Save the number of timesteps that are not nan, for easy calculation of stat test, also a mean across the full domain
                variable = list(winddataset.data.data_vars)[0] # Get the name of the variable to generalise
                lengths[e,w] = sum(~np.isnan(winddataset.data[variable].data)).mean() # len(winddataset.data.time) og version with all timesteps
                
        # Create a Datasets saving the extremes, absolute and relative
        ds = xr.Dataset(data_vars= dict(number_extremes=(['persistence_time','Location', 'Model'], extremes), 
                                        length=(['Location', 'Model'], lengths)),
            coords= dict(Location=[ens.name.split('_')[0] for ens in ens_list], Model =[ds.name for ds in ens_list[0].data_list], persistence_time = persistence_times))
        ds_relative = xr.Dataset(data_vars= dict(number_extremes=(['persistence_time', 'Location', 'Model'], extremes_relative), 
                                        length=(['Location', 'Model'], lengths)),
            coords= dict(Location=[ens.name.split('_')[0] for ens in ens_list], Model =[ds.name for ds in ens_list[0].data_list], persistence_time = persistence_times))

        # If needed, save this as a netcdf file (to avoid having to wait)
        if save:
            # Construct filename
            if len(persistence_times) > 1:  per = 'multiple'
            else: per =  persistence_times.item()
            filename = self.datahandler.construct_filename(data_type='list_ens', name = name, persistence_time=per, cut_in=cut_in,
                cut_out=cut_out, relative=False, num_days=num_days)
            self.datahandler.save_file(ds, filename)
            filename_relative = self.datahandler.construct_filename(data_type='list_ens', name = name, persistence_time=per, cut_in=cut_in,
                cut_out=cut_out, relative=True, num_days=num_days)
            self.datahandler.save_file(ds_relative, filename_relative) 
        return ds

    def compare_persistent_low_high_number_lidars(self, ens_list: list, persistence_times: list, cut_in: float = -np.inf, cut_out: float = np.inf, name: str = None, 
        verbose: bool = False, relative: bool = False, num_days: bool = False, save: bool = False):
        """ Compare the number of persistent lows/highs for a list of ensembles, specifically ensembles with pointdatasets
        so e.g. ensembles adapted to one observational point as the reference
        ens_list: list of ensembles
        persistence_times, cut_in, cut_out: as usual
        name: prefix to be added to the saved files
        verbose: should the outcomes be printed?
        relative: divide by the total number of years in the dataset
        num_days: should the total number of days be returned (True) or the count of the number of occurrences (False, default)
        save: should the results be saved?
        """
        # Initialise an empty list to store the results
        output = []

        # Loop over the ensembles in the list
        for ens in ens_list:
            # Calculate the number of periods for the reference
            if ens.ref is not None:
                ref_number = self.get_persistent_low_high(ens.ref, persistence_times, cut_in = cut_in, cut_out = cut_out, num_days = num_days, relative = relative)
                data = [ref_number.extreme.data.item()] # this will be the list of numbers for this ensemble
                if verbose: print(f'Reference {ens.ref.name}: {ref_number} occurrences')
            
            # Loop over the ensemble members
            for ds in ens.data_list:
                # again computing the number of periods
                ds_number = self.get_persistent_low_high(ds, persistence_times, cut_in = cut_in, cut_out = cut_out, num_days = num_days, relative = relative)
                if verbose: print(f'Reference {ds.name}: {ds_number} occurrences')
                data.append(ds_number.extreme.data.item())
            if verbose: print('------------------------------')

            # Add the data to the list with the outputs
            output.append(data)

        # Save the output if necessary
        if save:
            filename = self.datahandler.construct_filename(data_type='barplot', name=name, persistence_time = 'multiple', cut_in=cut_in,
                cut_out=cut_out, relative=relative, num_days=num_days)
            self.datahandler.save_file(output, filename) 

        return output

    def weighted_number_low_high(self, ens_list: list, name: str, persistence_times: np.array, cut_ins: np.array, include_reanalysis: bool, relative: bool = False, num_days: bool = False, verbose: bool = True):
        """ Plot weighted biases: given a list of ensembles with the same members but different references, compute a weighted average of the bias using the number of timesteps per reference
        ens_list: the list of ensembles
        name: name of the list of ensembles (used to load the precomputed number of lows)
        include_reanalysis: should the reanalyses be included in the weighted mean?
        persistence_times, cut_ins, relative, num_days: as before
        verbose: should factors be printed?
        """

        # Get the number of non nan timesteps for each of the pointdatasets (i.e. lidar observations)
        total_timesteps = [sum(~np.isnan(ens.ref.data.wind_speed)).data for ens in ens_list if type(ens.ref) ==  pp.PointWindDataSet]
        total_timesteps = sum(np.array(total_timesteps)) # calculating sum

        # Add the number of timesteps from the spatial datasets (i.e. reanalyses)
        if include_reanalysis:
            # The spatial datasets are reanalysis datasets which don't contain nans except where the domain is clipped
            # so we just use the maximal number of non nan timesteps
            timesteps_spatial = [sum(~np.isnan(ens.ref.data.wind_speed)).max().data for ens in ens_list if type(ens.ref) ==  pp.SpatialWindDataSet]

            # Calculate the total number of timesteps (not nans) of the lidars for the normalisation
            total_timesteps = sum(np.array(timesteps_spatial)) + total_timesteps
        if verbose: print(f"Total number of timesteps: {total_timesteps}")

        # Load one test dataset to get the model names
        test = self.datahandler.load_data(self.datahandler.construct_filename(data_type = 'list_ens', name = name, cut_in = cut_ins[0], 
                        persistence_time = 'multiple', relative = relative, num_days = num_days)) 
        model_names = test.Model.data

        # Load the reference datasets (to only open them once)
        refs = []
        for ref in [ens.ref for ens in ens_list]:
            if verbose: print('opening ' + ref.name)

            # Load the reference data
            obs = xr.open_dataarray(self.datahandler.path.replace('cmip', 'refs') + 
                self.datahandler.construct_filename(data_type = 'cut_ins', name = ref.name, relative = relative, num_days = num_days))
            refs.append(obs)         

        # Initialise an empty array to save the biases
        weighted_biases = np.empty(shape = (len(cut_ins), len(persistence_times), len(model_names)))

        # Loop over the persistence times and the capacity factor thresholds
        for i, cap in enumerate(cut_ins):
            
            # Load the model data
            filename_model = self.datahandler.construct_filename(data_type = 'list_ens', name = name, cut_in = cap, 
                persistence_time = 'multiple', relative = relative, num_days = num_days)
            models = self.datahandler.load_data(filename_model)
            
            # Loop over durations
            for j, dur in enumerate(persistence_times):

                if verbose:
                    print('-----------------------------')
                    print(f'{dur} days, {cap}%')
                    print('-----------------------------')

                # Select the duration
                models_dur = models.sel(persistence_time = dur)

                # Loop over the models
                for m, model in enumerate(model_names):
                    if verbose: print(f'Starting {model}')

                    # Initialise a 0 weighted bias
                    weighted_bias = 0

                    # Loop over the reference datasets
                    for k, ref in enumerate([ens.ref for ens in ens_list]):
                        if verbose: print(ref.name)

                        # Construct the name of the location in the file of extremes for the model, to avoid doing twice the same ERA5 and CERRA are both sea (as both are calculated for the sea gridpoints)
                        if ref.name in ['ERA5', 'CERRA']: location = 'Sea'
                        else: location = ref.name

                        # Calculate the weight (#non nan timesteps for this lidar divided by total lidar timesteps) and the bias
                        if type(ref) == pp.PointWindDataSet: factor = sum(~np.isnan(ref.data.wind_speed)).data/total_timesteps
                        else: 
                            if include_reanalysis:
                                factor = (sum(~np.isnan(ref.data.wind_speed))).max().data/total_timesteps
                            else: 
                                factor = 0 # set factor to zero if the reanalysis should not be included in the weighted mean
                        if verbose: print(f"Factor: {factor}")

                        # Calculate bias with earlier loaded reference datasets
                        bias_lidar = models_dur.sel(Location = location, Model = model).number_extremes.data - refs[k].sel(duration = dur, cut_in = cap).data
                        if verbose:
                            print(f"model value: {models_dur.sel(Location = location, Model = model).number_extremes.data}")
                            print(f"ref value: {refs[k].sel(duration = dur, cut_in = cap).data}") 
                            print(f"Bias: {bias_lidar}")

                        # Add the weighted bias for this lidar to the mean
                        weighted_bias = weighted_bias + bias_lidar*factor

                    # Add the bias to the array
                    weighted_biases[i, j, m] = weighted_bias

        # Create a DataArray saving the weighted biases
        ds = xr.DataArray(data = weighted_biases,
            dims= ['cut_in', 'duration', 'model'], 
            coords= dict(duration=persistence_times, cut_in = cut_ins, model = model_names))

        # Save the weighted biases for this model in a numpy file
        filename = self.datahandler.construct_filename(data_type = 'weighted', relative = relative, 
            name  = name, num_days = num_days, include_reanalysis = include_reanalysis)
        print(f'Saving {filename}')
        self.datahandler.save_file(ds = ds, filename = filename)

class ExtremePlotter:
    """ Object to plot the extremes data created by an ExtremeCalculator
    """

    def __init__(self, path: str = ''): # TODO ADAPT DEFAULT PATH
        """ Create an ExtremePlotter object
        path: path to the folder where to find files
        """
        self.datahandler = ExtremeDataHandler(path)

    def __repr__(self):
        """ How to print an ExtremePlotter object
        """
        return f'ExtremePlotter object with {self.datahandler.path} as base path to load the calculated extremes'

    def plot_persistent_low_high_map(self, name: str, persistence_time: int, cut_in: float = -np.inf, cut_out: float = np.inf, relative: bool = False,
        num_days: bool = False, text_size: float = 10, colorbar_label: str = None, title: str = None, cmap: str = 'viridis', plot_range = None):
        """ Plot the map showing the number of persistent lows and higs
        name: the name of the winddataset 
        etc same as before for getting persistent low/high (see function get_persistent_low_high) and for plotting 
        maybe could've just been **kwargs?
        """
        
        # Load the data
        data = self.datahandler.load_data(self.datahandler.construct_filename('base', name, 'multiple', cut_in, cut_out, relative, num_days, spatial = True))
        
        # Select the preferred persistence time
        data = data.sel(persistence_time = persistence_time)

        # Construct the type of extreme for the plot title and the colorbar label
        if cut_in == -np.inf:
            extreme_type = 'highs'
        elif cut_out == np.inf:
            extreme_type = 'lows'
        else:
            extreme_type = 'inactive periods'

        # Set the title and colorbar label if not given
        rel=''
        if relative: rel = " per year"
        unit ='occurrences'
        if num_days: unit = "days in periods"
        if title is None: title = f'Occurrence {persistence_time} days {extreme_type} in {name}'
        if colorbar_label is None: colorbar_label = f'Average number of {unit} >= {persistence_time} days {extreme_type}{rel}'

        # Make the plot
        comp  = pp.Comparator() # little hack to use plot map function from postprocessing and not from extremes
        comp.plot_map(data.where(data!=0), # convert zeros to nans so they are not plotted, important for special domains
            title = title ,
            colorbar_label = colorbar_label,
            text_size = text_size,
            plot_range =  plot_range,
            cmap = cmap
            )

    def barplot_number_occurences(self, ens_list: list, persistence_time: int, cut_in: float = -np.inf, cut_out: float = np.inf , name: str = None, relative: bool = False, num_days: bool = False, text_size: float = 13, title: str = None, verbose: bool = True, save: bool = False):
        """ Plot a barplot comparing the number of occurences of specific persistent highs/lows among several point ensembles
        specifically for comparing several ensembles with a different observational dataset as the reference and containing the reanalysis datasets CERRA and ERA5
        some parts of this function are hardcoded for this!
        ens_list: the list of ensembles, should be the same as the one the occurences are computed from
        persistence_time, cut_in, cut_out: as usual
        name: name of the ensemble list, prefix in the name of the files
        text_size: the size of the text in the figure
        title: optional, the title to add to the figure, if not given this is automatically generated
        relative: should the numbers be absolute or relative per year
        verbose: set this to False to avoid printing the warning
        """

        # Print a little warning
        if verbose:
            print('Warning: this function expects a list of ensembles each containing the same members in the same order and a lidar dataset as the reference.')

        # Load the data we need 
        filename = self.datahandler.construct_filename('barplot', name= name, persistence_time='multiple', cut_in=cut_in, cut_out=cut_out,
            relative=relative, num_days=num_days)
        data = self.datahandler.load_data(filename)
        data = data.sel(persistence_time = persistence_time)

        # Create the figure, the width depending on whether the data is relative or not
        if relative: width = 5
        else: width = 6
        fig, ax = plt.subplots(figsize=(12, width))

        # Set the font size
        plt.rcParams.update({'font.size': text_size})

        # Settings for the bar plot
        x = np.arange(len(ens_list)) # positions on the x-axis, one for each of the ensembles
        width = 0.25  # the width of the bars
        multiplier = 0
        labels = [member.name for member in ens_list[0].data_list] # labels for the data ! this assumes all ensembles have the same members
        labels.insert(0, 'Lidar') # add the lidar label for the reference data

        # set the x-axis labels as the names of the lidars (supposes the first part before any space is the name of the lidar)
        if not relative:
            names = [ens.ref.name.split(' ')[0]+"\n("+ str(round(sum(~np.isnan(ens.ref.data.capacity_factor.data))/365, 1))+' years)' for ens in ens_list]
        else: names = [ens.ref.name.split(' ')[0] for ens in ens_list] # don't include the duration for the relative version

        # Create the bars for the barplot, by looping over the types of data
        for i in range(len(ens_list[0].data_list)+1): 
            offset = width * multiplier # construct where this bar needs to be
            rects = ax.bar(x+offset, np.array(data).T[i], width, label = labels[i]) # plot the bar
            multiplier+=1 # advance
            ax.bar_label(rects, np.round(np.array(data).T[i],1), padding=3, size = 10) # adding the numbers to the top of the chart
        
        # Set the label on the y-axis depending on whether is a relative plot or not
        if relative: 
            if num_days: ax.set_ylabel('Mean number of days per year')
            else: ax.set_ylabel('Mean number of occurrences per year')
        else: 
            if num_days: ax.set_ylabel('Number of days during active period lidar')
            else: ax.set_ylabel('Number of occurrences during active period lidar')

        # Set the x axis ticks and some more generalities
        ax.set_xticks(x + width, names)
        ax.spines[['right', 'top']].set_visible(False) # remove right and top axis
        ax.legend(ncols = 3, frameon = False)

        # Set the title depending on the type of high/low/combined event we're looking at and on whether the data is relative
        if title is None:
            if not relative: rel_label = ''
            else: rel_label = ' per year'
            if num_days: unit = 'Days in '
            else: unit = 'Number of '
            if cut_in == -np.inf:
                ax.set_title(f'{unit} persistent highs of at least {persistence_time} days{rel_label} >{cut_out} %')
            elif cut_out == np.inf:
                ax.set_title(f'{unit} of persistent lows of at least {persistence_time} days{rel_label} < {cut_in} %')
            else:ax.set_title(f'{unit} of inactive periods of at least {persistence_time} days{rel_label}')
        else: ax.set_title(title)

        if save:
            fig.savefig('barplot_lidars.png')

    def heatmap(self, data: np.array, row_labels: list, col_labels: list, ax=None, cbar_kw: dict = None, cbarlabel: str = "", grid_color: str = 'w', rotation: float = 0, **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels. Copied from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
        data : A 2D numpy array of shape (M, N).
        row_label: A list or array of length M with the labels for the rows.
        col_labels: A list or array of length N with the labels for the columns.
        ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current Axes or create a new one.  Optional.
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel: The label for the colorbar.  Optional.
        rotation: the rotation angle for the x axis labels
        grid_color: color of the grid in the figure, default white
        **kwargs: All other arguments are forwarded to `imshow`.
        """

        # Create ax if needed
        if ax is None:
            ax = plt.gca()

        # Set the colorbar keyword arguments
        if cbar_kw is None:
            cbar_kw = {}

        # Expand one column data
        if len(data.shape) == 1: data = np.expand_dims(data, axis = 1)

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(range(data.shape[1]), labels=col_labels,
                    rotation= rotation, ha="center", rotation_mode="anchor")
        ax.set_yticks(range(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on the bottom.
        ax.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

        # Turn spines off and create grid.
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color=grid_color, linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold: float=None, significance: xr.DataArray = None, **textkw):
        """
        A function to annotate a heatmap. Copied from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
        im: The AxesImage to be labeled.
        data: Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt: The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolor: A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold: Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        significance: dataset specifying with 1 and 0s whether the data is significant, should be the same shape as data
        **kwargs: All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Expand one column data
        if len(data.shape) == 1: data = np.expand_dims(data, axis = 1)

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                number = valfmt(data[i, j], None)
                if number == 0: number = 0 # to avoid things like -0.0
                font = 'normal'
                if significance is not None:
                    if significance[j, i] == 0: font = 'light'
                    elif significance[j, i] == 1:  font = 'bold'
                text = im.axes.text(j, i, number, weight = font, **kw)
                texts.append(text)

        return texts

    def plot_multiple_cut_ins_durations(self, name: str, relative: bool = False, num_days: bool = False, ax = None, cmap: str = 'viridis_r', plot_range: list = None, grid_color: str = 'w', remove_cbar: bool = False,**text_kwargs):
        """ Plot a heatmap of the number of extremes for different cut in speeds and durations
        name: the name of the winddataset
        relative: do we want the numbers per year or absolute
        num_days: do we want the actual number of days or the number of occurrences of such perios
        ax: optional, ax to plot this heatmap on
        cmap: the color map
        plot_range: the range for the colorbar of the plot
        grid_color: the color of the grid
        """

        # Load the data
        filename = self.datahandler.construct_filename(data_type='cut_ins', name= name, relative=relative, num_days=num_days)
        ds = self.datahandler.load_data(filename)

        # Set the plot_range if needed
        if plot_range is None:
            plot_range = [np.min(ds.data), np.max(ds.data)]

        # Create the figure
        if ax is None:
            fig, ax = plt.subplots()
        if relative: rel_label = ' per year'
        else: rel_label = ''
        if num_days: unit = 'days'
        else: unit = 'occurrences'
        im, cbar = self.heatmap(ds.data, ds.duration.data, ds.cut_in.data.astype(int), ax=ax, vmin = plot_range[0], vmax = plot_range[1],
                        cmap=cmap, cbarlabel=f"Number of {unit}{rel_label}", cbar_kw = dict(fraction=0.046, pad=0.04), grid_color = grid_color)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}",**text_kwargs)
        
        # Plot niceties
        if remove_cbar:cbar.remove()
        ax.set_ylabel('Duration (days)')
        ax.set_xlabel('Capacity factor threshold (%)')
        ax.set_title(name)
        plt.tight_layout()
        return im # for plotting in rasters

    def plot_multiple_list_ens(self, name: str, persistence_time: int, cut_in: float = -np.inf, cut_out: float = np.inf, relative: bool = False, num_days: bool = False, 
        compare_fut: bool = False, ax = None, cmap = 'viridis_r', title: str = None, plot_range: list = None, text_colors = ("black", "white"), grid_color: str = 'w', significance: xr.DataArray = None, remove_cbar: bool = False):
        """ Plot a heatmap of the number of extremes for the members of a list of ensembles, all with the same members but e.g. at the different locations
        name: the name given to the saved files
        persistence_time, cut_in, cut_out as usual
        relative: do we want the numbers per year or absolute
        num_days: do we want the actual number of days or the number of occurrences of such periods
        compare_fut: do we want to calculate the difference between future and historical
        ax: optional, ax to plot this heatmap on
        cmap: the color map
        title: optional, title for the plot
        plot_range: the range for the colorbar of the plot
        text_colors: colors for the text annotations
        grid_color: color of the grid
        significance: optional, dataset containing the 1/0 depending on whether the datapoint is significant, should be the same shape as the data that is loaded
        remove_cbar: keep or remove the colorbar
        """
        # Load the data  
        filename = self.datahandler.construct_filename(data_type='list_ens', name= name, persistence_time='multiple', cut_in=cut_in, cut_out=cut_out,
            relative=relative, num_days=num_days)
        ds = self.datahandler.load_data(filename)
        ds = ds.sel(persistence_time = persistence_time)

        # Subtract hist from future data if needed
        if compare_fut:
            ds2 = self.datahandler.load_data(filename.replace('_hist','_fut')) # load the future dataset which should nicely be named like this
            ds2 = ds2.sel(persistence_time = persistence_time)
            ds = ds2 - ds # this automatically ignores models that are there for historical but not for the future (e.g. reanalysis datasets)
            cmap = 'coolwarm' # set a divergent color scale
            text_colors = ('black', 'black') # set black text cause this color scale is quite light

        # Select the data, two options to be compatible with old version when length was not saved yet
        if type(ds) == xr.Dataset: data = ds.number_extremes.data
        else: data = ds.data

        # Set the plot range
        if plot_range is None:
            plot_range = [np.min(data), np.max(data)]
            # Symmetrize the color scale if the first value is negative
            if plot_range[0]<0:
                abs_max = np.max(np.abs(plot_range))
                plot_range = [-abs_max,abs_max]

        # Create the figure
        if ax is None:
            fig, ax = plt.subplots()
        if relative: rel_label = ' per year'
        else: rel_label = ''
        if num_days: unit = 'days'
        else: unit = 'occurrences'
        im, cbar = self.heatmap(np.transpose(data), [m.split('(')[0] for m in ds.Model.data], ds.Location.data, ax=ax, cmap=cmap, grid_color=grid_color, 
            cbarlabel=f"Number of {unit}{rel_label}", rotation = 30, vmin = plot_range[0], vmax = plot_range[1], cbar_kw = dict(fraction=0.046, pad=0.04))
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}", textcolors= text_colors, significance = significance)
        ax.set_ylabel('Model')
        ax.set_xlabel('Location')
        if remove_cbar:cbar.remove()


        # Construct the title of the plot if needed
        if compare_fut: fut_label = '\nchange in future'
        else: fut_label = ''
        if title is None:
            title = f'{name} {int(persistence_time)} days of < {int(cut_in)}% CF{fut_label}'
        ax.set_title(title)
        plt.tight_layout()

        return im # for plotting in rasters

    def plot_weighted_biases(self, model: str, name: str, include_reanalysis: bool, relative: bool = False, num_days: bool = False, ax = None, cmap = 'seismic', 
        plot_range: list = None, text_colors = ("black", "white"), grid_color: str = 'w', remove_cbar: bool = False, **kwargs):
        """ Plot tables of the biases per model, weighted by the reference datasets
        model: name of the model to plot
        name: name of the ensemble to load the data from
        include_reanalysis: should the reanalyses be taken into account in the weighted mean?
        relative, num_days as per usual
        remainder are plot arguments
        """
        # Create the plots if ax is not given
        if ax is None:
            fig, ax = plt.subplots()

        # Load the data
        weighted_biases = self.datahandler.load_data(self.datahandler.construct_filename(data_type ='weighted', name = name, relative = relative, num_days = num_days, include_reanalysis = include_reanalysis))
        
        # Select the model
        weighted_biases =  weighted_biases.sel(model = model)

        # Plot it 
        if include_reanalysis: cbarlabel = 'Weighted bias'
        else: cbarlabel = 'Weighted bias (only lidars)'
        im, cbar = self.heatmap(weighted_biases.T, row_labels = weighted_biases.duration.data, col_labels = weighted_biases.cut_in.data, grid_color = grid_color,
            cbarlabel = cbarlabel, cmap = cmap, vmin = plot_range[0], vmax = plot_range[1], ax = ax)
        self.annotate_heatmap(im, data = weighted_biases.T, textcolors= text_colors ,valfmt="{x:.1f}",  **kwargs);
        
        # Plot niceties
        ax.set_ylabel('Duration (days)')
        ax.set_xlabel('Capacity factor threshold (%)')
        ax.set_title(model)
        # Remove colorbar if needed
        if remove_cbar: cbar.remove()
        plt.tight_layout()
        return im

    def plot_map_change_number_extremes(self, names: str, persistence_time: int, cut_in: float, cut_out: float = np.inf, relative: bool = False, num_days: bool = False, 
        hatch_list = None, cmap: str = 'coolwarm', plot_mean: bool = False, offsets: list = [-2.2, -2], 
        plot_range = None, no_difference: bool = False, text_size: float = 6):
        """ For each ensemble member, plot the change in the number of persistent lows for the specified duration.
        names: name of the ensemble members including run
        persistence_time, cut_in, ...: properties extremes
        hatch_list: optional, hatching created by poisson_test_spatial
        cmap: the colormap
        relative: if True, relative bias is plotted, if False, bias is plotted
        plot_mean: optional, determines if the plot should have an annotation with the mean
        offsets: offsets [x,y] to the annotation of plot mean
        no_difference: plot numbers for both hist and fut separate instead of taking difference, level should be None in this case
        text_size: size of the text in the figure
        """
        # Check if input is consistent with the working of the function
        if (hatch_list is not None) & no_difference:
            raise ValueError('Conflicting input parameters: impossible to plot hatching for statistical test '
            'on non-difference maps of hist and future data. Alter the no_difference or the level parameter and try again.')

        # Load the data and calculate the differences
        diffs = []
        for name in names:
            filename = self.datahandler.construct_filename(data_type='base', name = name + '_Sea_hist', 
                persistence_time='multiple', cut_in=cut_in, cut_out=cut_out,
                relative=relative, num_days=num_days, spatial = True)
            hist = self.datahandler.load_data(filename)
            fut = self.datahandler.load_data(filename.replace('_hist','_fut')) # load the future dataset which should nicely be named like this

            # Select the preferred persistence time
            hist = hist.sel(persistence_time = persistence_time)
            fut = fut.sel(persistence_time = persistence_time)            

            # Add the difference between historical and future to the list
            if no_difference:
                diffs.append(hist)
                diffs.append(fut)
            else:
                diffs.append(fut-hist)
        print('Difference here is fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0 and if decrease in the future, we have fut < hist <=> fut - hist < 0')

        # Calculate the range as min, max of all diffs
        if plot_range is None:
            plot_range = [np.min(np.array([diff.min() for diff in diffs])), np.max(np.array([diff.max() for diff in diffs]))]

            # Symmetrize the range
            if not no_difference:
                abs_max = np.max(np.abs(plot_range))
                plot_range = [-abs_max,abs_max]

        # Create the plots
        nrows, ncols = pp.close_factors(len(diffs))
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (12, 10))
        
        # Plot a difference map for each ensemble member
        for j, diff in enumerate(diffs):
            # Choose the axis
            if (ncols > 1) | (nrows > 1): ax = axes.reshape(-1)[j]
            else: ax = axes
            ax.set_global()

            # Set the hatching if needed
            if hatch_list is not None:
                hatch = hatch_list[j] # select this member
                hatch = np.where(np.isnan(diff), np.nan, hatch) # mask nan values from original dataset to avoid hatching outside domain
                hatch = np.where(hatch == 1, hatch, np.nan) # put nans wherever the statistical test did not return true

            else: 
                hatch = None

            # Set the ticks depending on where the plot is positioned
            if ax.get_subplotspec().is_first_col() & ax.get_subplotspec().is_last_row(): labels = ['left', 'bottom']
            elif ax.get_subplotspec().is_last_row(): labels = ['bottom']
            elif ax.get_subplotspec().is_first_col(): labels = ['left']
            else: labels = False

            # Create the plot
            comp  = pp.Comparator() # little hack to use plot map function from postprocessing and not from extremes
            if no_difference: title = names[j//2].split(' ')[0] + f"{' historical' if j%2 == 0 else ' future'}"
            else: title =  names[j].split(' ')[0]
            im = comp.plot_map(diff, title = title, plot_range = plot_range, labels = labels, cmap = cmap, hatch = hatch, ax = ax, 
                remove_cbar = True, hatch_pattern = 'XXXXXX', text_size = text_size, plot_mean = plot_mean, offsets = offsets)

        # Remove unnecessary axes
        for i in range(1,ncols*nrows - len(diffs)+1):
            axes.reshape(-1)[-i].axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax = axes, fraction=0.03, pad=0.04)
        if no_difference: label = f'Number of occurrences persistent lows per year'
        else: label = f'Change number of occurrences persistent lows per year'
        cbar.ax.set_ylabel(label, rotation=-90, va="bottom")