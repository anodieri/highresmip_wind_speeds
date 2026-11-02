# Class and helper functions to process raw downloaded data from CMIP6, ERA5, CERRA and observations 
# into WindDataSets of a common format
# also converts geopotential height to actual heights

#import necessary packages
import numpy as np
import xarray as xr
import cftime
import datetime
import scipy.interpolate
from wrf import interplevel
import speedup_griddata as sg
from shapely import Polygon
import rioxarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import metpy.calc
from metpy.units import units
import metview
#helper functions outside the class

def calculate_wind_speed(u: np.array, v: np.array):
    """ Calculate wind speed from u and v components of wind
    u,v : the components"""

    wind_speed = np.sqrt(np.square(u) + np.square(v)) #calculate length of the wind vector with components u and v
    return(wind_speed)

def calculate_wind_direction(u: np.array, v: np.array):
    """ Calculate wind direction from u and v components of wind
    u,v: the components"""

    wind_direction = np.rad2deg(np.arctan2(u, v))+180 #convert to wind direction (from length of vectors to angle)
    return(wind_direction)

def create_dataset(wind_speed: np.array, wind_direction: np.array, lat: np.array, lon: np.array, time: np.array, attributes: dict):
    """ Given wind_speed, wind_direction, lat, lon, and time save these in an xarray dataset, for spatial dataset
    (these have to be given separately because different filetypes have different names e.g. ERA5 has valid_time instead 
    of time like in CMIP6)
    """
    
    # construct dimensions for the netcdf file
    lat = xr.DataArray(
        data=lat.astype('float32'),
        dims=["lat"],
        coords=dict(
            lat=(["lat"], lat)
        ),
        attrs=dict(
        long_name="latitude",
        units="degrees_north",
        axis="Y"
        ),
    )
    lon = xr.DataArray(
        data=lon.astype('float32'),
        dims="lon",
        coords=dict(
            lon=(["lon"], lon)
        ),
        attrs=dict(
        long_name="longitude",
        units="degrees_east",
        axis="X"
        ),
    )
    time = xr.DataArray(
        data=time,
        dims="time",
        coords=dict(
            time=(["time"], time)
        ),
        attrs=dict(
        long_name="time",
        ),
    )

    # save wind speed and wind direction variables in a netcdf file
    ds = xr.Dataset(
        data_vars=dict(
            wind_speed = (
                ["time","lat","lon"], wind_speed.astype('float32'),
                dict(long_name = "wind_speed",
                units = "m s-1",
                vert_units = "m")),
            wind_direction = (
                ["time","lat","lon"], wind_direction.astype('float32'),
                dict(long_name = "wind_direction",
                units = "degrees",
                vert_units = "m"))
            ),
        coords=dict(
            lon=lon,
            lat=lat,
            time=time
            ),
        attrs=attributes
    )

    # return the netcdf file
    return ds

def create_point_dataset(wind_speed: np.array, wind_direction: np.array, lat: float, lon: float, height: float, time: np.array, attributes: dict):
    """ Given wind_speed, wind_direction, lat, lon, height and time save these in an xarray dataset, for a point measurement dataset
    """
    
    # construct dimensions for the netcdf file
    time = xr.DataArray(
        data=time,
        dims="time",
        coords=dict(
            time=(["time"], time)
        ),
        attrs=dict(
        long_name="time",
        ),
    )

    # construct the attributes, lat lon and height are attributes now since there is only one value
    attributes_own = dict(
        lat = lat,
        lon = lon,
        height = height, 
        frequency = '6h', 
        calendar = 'Gregorian',
        )
    attributes_own.update(attributes) #add all given attributes

    # save wind speed and wind direction variables in a netcdf file
    ds = xr.Dataset(
        data_vars=dict(
            wind_speed = (
                ["time"],wind_speed.astype('float32'),
                dict(long_name = "wind_speed",
                units = "m s-1",
                vert_units = "m")),
            wind_direction = (
                ["time"], wind_direction.astype('float32'),
                dict(long_name = "wind_direction",
                units = "degrees",
                vert_units = "m"))
            ),
        coords=dict(
            time=time
            ),
        attrs=attributes_own
    )

    # return the netcdf file
    return ds

def _find_nearest(array: np.array, value: float):
    """ find the element in the array that is closest to the value
    array: array/list to find the closest value in
    value: the value the returned value will be closest to
    """

    #if there's only one element, return it
    if len(array)==1:
        return array
    #otherwise select the closest
    else:
        index  = (np.abs(array-value)).argmin()
        return array[index]

def create_domain(points: list, plot = True):
    """ Given a list of coordinates of points, create a domain and plots it
    points: a list of lists containing the lon, lat coordinates of each point
    plot: should the domain be plotted?
    """

    # Check if no longitudes are above >180
    if True in [pt[0]>180 for pt in points]:
        print("Warning: a longitude > 180 is used in the coordinates of the domain, while negative longitudes are assumed here.")

    # Create the domain
    domain= Polygon(points)

    # Create a nice map
    if plot:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        plt.plot(*domain.exterior.xy, color = 'red', linewidth = 2)
        ax.set_extent([-14,18,43,62], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Return the domain
    return domain

def clip_dataset(ds: xr.Dataset, domain: Polygon):
    """ Restrict a dataset to the given domain
    ds: the dataset, in xarray format
    domain: the domain to restrict to, a shapely polygon
    """
    varname = 'lon'
    if 'longitude' in ds.coords: varname = 'longitude'
    if (ds[varname]>180).sum() > 0:
            print("Warning: a longitude > 180 is found in the coordinates of the dataset, while negative longitudes are assumed for the domain.")

    ds.rio.write_crs('WGS84', inplace=True) # set the coordinate reference system of the dataset
    # WGS84 is the typical one for lon-lat coordinates, see doctoral school course https://github.com/jorisvandenbossche/DS-python-data-analysis open source material

    clipped = ds.rio.clip([domain])
    return clipped

def split_into_simulations(path_list: list, resolutions: list, experiments: list = ['hist-1950', 'highres-future' ]):
    """ Split a list of paths into lists of different simulations: high and low resolution future and historical 
    path_list: the list of paths to split
    resolutions: strings to identify the different resolutions, e.g. ['LL', 'HM'] for HadGEM
    experiments: the identifiers of the different experiments
    """

    # Create the substrings to identify the different simulations, combines resolutions and experiments
    if len(experiments) == 2:
        substrings = sum([[res + '_' + exp for exp in experiments] for res in resolutions],[]) # the sum is a little trick to flatten the list
    # if 4 experiments are given, use first two for one resolution and last two for the other resolution
    elif len(experiments) == 4:
        substrings_res1 = [resolutions[0] + '_' + exp for exp in experiments[0:2]] 
        substrings_res2 = [resolutions[1] + '_' + exp for exp in experiments[2:4]]
        substrings = sum([substrings_res1, substrings_res2],[]) # the sum is a little trick to flatten the list
        
    groups = {sub: [] for sub in substrings} # create a dictionary with a list for each experiment
    print('The substrings that will be used:')
    print(substrings)
    print('Warning: make sure that if there are strings contained within one another, \nthat the longest one is listed first and the one contained in it is listed afterwards, \nso the longer ones are selected out first.\n')

    # Only keep netcdf files in the paths
    path_list = [p for p in path_list if '.nc' in p]

    # Loop over the paths an add them to the right group
    for p in path_list:
        for sub in substrings:
            if sub in p:
                groups[sub].append(p)
    return groups
    
#######################################################################################
## When implementing new interpolation method, add attribute describing it to file!
#######################################################################################

class Processor:
    """handles processing from raw climate model/reanalysis data to dataset 
    with wind speed and direction at hub height:
    1. reading data
    2. interpolating to hub height if needed
    3. restricting the domain if needed
    4. constructing netcdf file
    """
    
    def __init__(self, interpolation_method: str = 'power', alpha: str = None, level: float = 100., lat: list = None, lon: list = None, domain: Polygon = None): 
        """initialise a processor object with specified levels and interpolation method to perform vertical interpolation
        interpolation_method: String specifying the method, should be power, log or linear
        alpha: method to estimate the power law exponent alpha, options see set_alpha method
        level: the level to interpolate to
        lat: the min and max latitudes of the domain in a list, if not given, all data is used
        lon: the min and max longitudes of the domain in a list, if not given, all data is used
        domain: for a non rectangular domain, a Polygon created by create_domain, in this case lat and lon are not needed
        """

        self.interpolation_method = interpolation_method
        self.alpha = alpha
        self.level = level
        self.lat = lat
        self.lon = lon
        self.domain = domain
        print( f'a processor object was created with the following specs: \ninterpolation method: {interpolation_method} \nlevel: {level}')

    def set_alpha(self, wind_speed : xr.DataArray = None, surface: xr. DataArray = None, height: xr.DataArray = None):
        """ Set the alpha values to use for the power law interpolation
        method: the kind of alpha to use, standard is 1/7, standard_sea is 1/8.5, era5 is estimated at each grid point of era5 from 10 and 100 m
            era5_mean is the mean over all grid points from the alpha estimated from era5 at 10 and 100 m 
            surface means that alpha is estimated from two levels of wind speeds: surface and the given pressure level at the given height, only usable for cmip6 not era5
        wind_speed, surface, height: optional, dataarrays containing these variables to estimate alpha from two levels
        """

        match self.alpha:
            case 'standard':
                alpha = 1/7 # e.g. Jung 2022, Gualtieri 2019
            case 'standard_sea':
                alpha = 1/8.5 # Dawenport 1960 (source in Gualtieri 2019)
            case 'era5':
                # inspired roughly by Carvalho 2021 (uses iterative method to estimate alpha from ERA5, here just inverting equation)
                era5_100m = xr.open_dataset("") # load processed era5 data #TODO SET PATH TO ERA5 100m
                era5_10m = xr.open_dataset("")    # TODO SET PATH TO ERA5 10m
                alpha = np.log10(era5_100m.wind_speed/era5_10m.wind_speed).mean(dim = 'time', dtype=np.float64) # inverted power law equation, mean over time
            case 'era5_mean':
                era5_100m = xr.open_dataset("") # load processed era5 data # TODO SET PATH TO ERA5 100m
                era5_10m = xr.open_dataset("")   # TODO SET PATH TO ERA5 10m
                alpha = np.log10(era5_100m.wind_speed/era5_10m.wind_speed).mean(dtype=np.float64).data # inverted power law equation, mean over time and space
                print(f'mean alpha estimated from ERA5 is {alpha}')
            case 'surface':
                # only positive heights to avoid imaginary alphas, negative heights: set to nan
                alpha = np.log10(wind_speed/surface.sfcWind)/np.log10(height.height.where(height.height>0)/10, dtype = np.float64)
            
        return alpha

    def interpolate_vertically_era5(self, wind_speed: xr.DataArray):
        """Perform vertical interpolation of an array a to specified levels, specifically for ERA5 single level data
        wind_speed = the DataArray to interpolate
        """
        # only for interpolating to a single level at the moment for log and power
        # might be possible to integrate this into interpolate vertically, but not done since this is just for testing at the moment
        
        #Perform the interpolation depending on the method
        match self.interpolation_method:
            case 'linear':
                raise ValueError('This does not work, single lvl ERA5 has only a single level so no linear interpolation is possible')
            case 'power':
                closest_level = 10 #specifically for era5: interpolating from 10 m to 100 m
                if self.alpha == 'surface': raise ValueError('Estimation of alpha from surface is not an option for ERA5. Use option ear5 or era5_mean to estimate alpha between 10 and 100 m era5.')
                alpha = self.set_alpha() #set the exponent alpha
                if isinstance(alpha, xr.DataArray):
                    alpha = alpha.rename({'lon': 'longitude','lat': 'latitude'}) # rename dimensions to match those of ERA5
                wind_speed_lev = wind_speed*((self.level/closest_level)**alpha) # apply the power law (formula from e.g. Carvalho 2021, eq 1)
            case 'log':
                closest_level = 10 #specifically for era5: interpolating from 10 m  to 100 m
                z_0 = 1.52*10**(-4) #local roughness length over the ocean (deCastro 2018)
                #np.log is natural log with base e, np.log10 is log with base 10
                wind_speed_lev = wind_speed*(np.log(self.level/z_0)/np.log(closest_level/z_0)) #apply log law (formula from deCastro 2018, eq 4)
        
        # Return the interpolated version of the array
        return wind_speed_lev

    def process_era5(self, path_u: str, path_v: str, freq: str, interpolate: bool = False, test_mode: bool = False):
        """Given paths for the u and v components on a single level for ERA5, process downloaded reanalysis data
        to a netcdf dataset with wind speed and wind direction.
        path_u: path of the file containing the u component on a single level
        path_v: path of the file containing the v component on a single level
        freq: either 1h for native resolution, 3h, 6h for downsampling or day for daily means
        interpolate: should vertical interpolation be applied?
        test_mode: if True, only the first 100 timesteps are loaded, for quick testing
        """
        
        # Read the NetCDF files
        print('Opening files')
        ds_u = xr.open_dataset(path_u)
        ds_v = xr.open_dataset(path_v)

        # If the latitude dimension is reversed, fix it (this is the case for ERA5)
        if ds_u.latitude[-1] < ds_u.latitude[0]:
            ds_u = ds_u.reindex(latitude = ds_u.latitude[::-1])
            ds_v = ds_v.reindex(latitude = ds_v.latitude[::-1])

        print('Restricting domain')
        # Restrict the domain if needed
        if self.lat is not None:
            ds_u = ds_u.sel(dict(latitude = slice(self.lat[0],self.lat[1]),
             longitude = slice(self.lon[0], self.lon[1])))
            ds_v = ds_v.sel(dict(latitude = slice(self.lat[0],self.lat[1]),
             longitude = slice(self.lon[0], self.lon[1])))
        elif self.domain is not None:
            ds_u = clip_dataset(ds_u, self.domain)
            ds_v = clip_dataset(ds_v, self.domain)

        # Construct variable name, the part that varies between 10 and 100 m level
        if '100' in path_u:
            var = '100'
        else: var = '10'

        print('Extracting variables')
        # Extract variables from netcdf files and time slice if needed
        if test_mode:
            u = ds_u['u'+ var].isel(valid_time=slice(1,103)) #u at 10/100 m
            v = ds_v['v'+ var].isel(valid_time=slice(1,103)) #v at 10/100 m 
            valid_time = ds_u['valid_time'].isel(valid_time=slice(1,103))
        else:
            u = ds_u['u'+ var] #u at 10/100 m
            v = ds_v['v'+ var] #v at 10/100 m 
            valid_time = ds_u['valid_time']

        lat = ds_u['latitude'] 
        lon = ds_u['longitude']

        print('Interpolating')
        # Calculate wind speed and wind direction, interpolation only applied if specified
        # since era5 is available at 100 m, interpolation is not always needed
        if interpolate: # linear does not work since only one level, to try this era5 on pressure levels is needed
            # calculate the wind speed (before interpolating)
            wind_speed = calculate_wind_speed(u,v)

            #interpolate wind speed to vertical levels
            wind_speed =  self.interpolate_vertically_era5(wind_speed)

            # interpolate components u and v and calculate wind direction
            # order is different than for wind speed
            # in fact, vertical interpolation is not needed for power and log law (write out equations, factors from log and power law cancel out 
            # when computing arctan (u/v)), but for linear this is needed, this is why it is implemented
            if self.interpolation_method == 'linear': #raises error, there's only one level
                u_lev = self.interpolate_vertically_era5(u) #interpolation from model levels to specified levels
                v_lev = self.interpolate_vertically_era5(v)
            else:
                u_lev = u #era5 single levels: no selection of correct level needed
                v_lev = v

            # calculate wind direction from u and v at desired level (linear) or original level (other methods)
            wind_direction = calculate_wind_direction(u_lev,v_lev) #calculate wind direction

        # without any interpolation
        else: 
            wind_speed = calculate_wind_speed(u,v) #calculate length of the wind vector with components u and v
            wind_direction = calculate_wind_direction(u,v) #convert to wind direction (from length of vectors to angle)
        
        # Construct the attributes to for the new, processed file 
        attributes= dict(
                data_source = "Processed data from ERA5",
                frequency = freq, #downloaded as such with 6 h intervals
                resolution = '0.25° x 0.25° (atmosphere), 0.5° x 0.5° (ocean waves)',
                calendar = 'Gregorian'
                )
        attributes.update(ds_u.attrs) #add all attributes from original file as well

        #Create the dataset
        ds = create_dataset(wind_speed.values, wind_direction.values, lat.values, lon.values, valid_time.values, attributes)
        
        # Close files again
        ds_u.close()
        ds_v.close()

        print('Downsampling/calculating mean')
        # Calculate daily means or downsample if needed
        match freq:
            case 'day':
                ds = ds.resample(time = 'D').mean(skipna = True) # resample by day and then take the mean
            case '6h':
                ds = ds.sel(time=(ds['time'].dt.hour == 6)|(ds['time'].dt.hour == 12)|(ds['time'].dt.hour == 18)|(ds['time'].dt.hour == 0))
            case '3h':
                ds = ds.sel(time=(ds['time'].dt.hour == 6)|(ds['time'].dt.hour == 12)|(ds['time'].dt.hour == 18)|(ds['time'].dt.hour == 0)|(ds['time'].dt.hour == 3)|(ds['time'].dt.hour == 9)|(ds['time'].dt.hour == 15)|(ds['time'].dt.hour == 21))
            case '1h': pass # in this case no action is needed    

        # Return the processed dataset
        return ds
    
    def process_cerra(self, path_wspd: str, path_wdir: str, test_mode: bool = False, daily_mean: bool = False):
        """Given paths for the wind speed and wind direction on a single level for CERRA, process downloaded reanalysis data
        to a netcdf dataset with wind speed and wind direction. The CERRA data is downloaded on a curvilinear grid (so no parallel axes).
        Here the wind speeds are horizontally interpolated to a rectilinear grid with a resolution of 0.05°.
        No vertical interpolation is performed
        path_wspd: path of the file containing the wind speed on a single level
        path_wdir: path of the file containing the wind direction on a single level, 
        test_mode: if True, only the first 7 timesteps are loaded, for quick testing
        daily_mean: should the daily means be calculated?
        TODO note: interpolation wind direction does not know that 0° and 360° are the same. 
        This results in discontinuities: the regridding does not conserve the sharp boundaries (in the original there's a jump from 0 to 360). 
        After interpolating there appear some intermediate values between 0 and 360, which alter the values of the sin and cos, so the u and v component, removing the continuity.
        """
        
        # Read the NetCDF files containing the wind speed and wind direction
        ds_wspd = xr.open_dataset(path_wspd)
        if path_wdir is not None:
            ds_wdir = xr.open_dataset(path_wdir) 

        # if the domain is not rectangular, we still need the other bounds of a rectangle around it to create rectilinear coordinates
        if self.domain is not None:
            bounds = self.domain.envelope.bounds #envelope creates a rectangle around the domain
            self.lat = [bounds[1]+0.1, bounds[3]+0.1] # add a small buffer
            self.lon = [bounds[0]+0.1, bounds[2]+0.1]

        # Construct the unique longitudes and latitudes for the new rectilinear grid
        # could maybe be done a little more efficient using rasterize/to_dataframe, but this works so ...
        if self.lat is not None:
            unique_lats = np.arange(self.lat[0],self.lat[1], 0.05) # grid with resolution of 0.05° (5.5 km resolution of the data, while ERA5 is 0.25° which is roughly 30 km)
            unique_lons = np.arange(self.lon[0],self.lon[1], 0.05)
        else: # min and max lat based on downloading of ERA5 if no lat and lon are given
            unique_lats = np.arange(43,61, 0.05) # grid with resolution of 0.05°
            unique_lons = np.arange(-12, 15, 0.05)

        # Convert longitudes above 180° to negative values (otherwise interpolation does not recognize that .1 and -.1 are close)
        ds_wspd.longitude.data[ds_wspd.longitude.data>180] = ds_wspd.longitude.data[ds_wspd.longitude.data>180] -360
        if path_wdir is not None:
            ds_wdir.longitude.data[ds_wdir.longitude.data>180] = ds_wdir.longitude.data[ds_wdir.longitude.data>180] -360

        # Repeat the lon and lat values to generate two columns containing the coordinates to each grid point 
        lats = np.repeat(unique_lats, len(unique_lons))
        lons = np.repeat([unique_lons], len(unique_lats), axis = 0).flatten()

        # Construct variable name, the part that varies between 10 and 100 m level
        if '100' in path_wspd:
            var = '100'
            var_wdir = '' # for some reason the wind direction does not get 100 after the variable name
        else: 
            var = '10'
            var_wdir = '10'

        # Extract variables wind speed, wind direction and valid time from netcdf files and time slice if needed
        if test_mode:
            wspd = ds_wspd['si'+ var].isel(valid_time=slice(1,20)) 
            if path_wdir is not None:
                wdir = ds_wdir['wdir' + var_wdir].isel(valid_time=slice(1,20)) 
            valid_time = ds_wspd['valid_time'].isel(valid_time=slice(1,20))
        else:
            wspd = ds_wspd['si'+ var] 
            if path_wdir is not None:
                wdir = ds_wdir['wdir' + var_wdir]
            valid_time = ds_wspd['valid_time']
            
        # Close files again
        ds_wspd.close()
        if path_wdir is not None:
            ds_wdir.close()

        # Extract and flatten the longitudes and latitudes
        lat_flat = ds_wspd.latitude.data.flatten()
        lon_flat = ds_wspd.longitude.data.flatten()

        # Create an empty lists to save the interpolated wind speed and wind direction values at each time step
        wspd_timesteps = []
        wdir_timesteps = []

        # use downloaded module to calculate the horizontal interpolation weights once and for all
        vtx, wts = sg.interp_weights(lat_flat, lon_flat, lats, lons)

        # Loop over the timesteps and horizontally interpolate the wind speed and wind direction values to the rectilinear grid
        for j in range(len(wspd.valid_time)):
            if j%100==0: print(f'timestep {j}')
            wspd_flat = wspd.isel(valid_time = j).data.flatten() # flatten the wind speed for one time step

            # perform the interpolation using the precalculated weights
            interpolated_wspd = sg.interpolate(wspd_flat, vtx, wts, shape =(len(unique_lats),len(unique_lons)))
            #interpolated_long = scipy.interpolate.griddata((lat_flat,lon_flat), wspd_flat, (lats,lons)) # perform the interpolation
            #interpolated = np.reshape(interpolated_long,(len(unique_lats),len(unique_lons))) # convert from long format to 2 d array in shape of lat/lon
            wspd_timesteps.append(interpolated_wspd) # add this timestep to the list

            if path_wdir is not None:
                wdir_flat = wdir.isel(valid_time = j).data.flatten()
                interpolated_wdir = sg.interpolate(wdir_flat, vtx, wts, shape =(len(unique_lats),len(unique_lons)))
                wdir_timesteps.append(interpolated_wdir)

        # Convert the list to an array
        wind_speed_interpolated = np.array(wspd_timesteps)
        if path_wdir is not None:
            wind_direction_interpolated = np.array(wdir_timesteps)
        else: wind_direction_interpolated = np.full(wind_speed_interpolated.data.shape, None)

        # Construct the attributes to for the new, processed file 
        attributes= dict(
                data_source = "Processed data from CERRA",
                frequency = '3h', #downloaded as such with 3 h intervals
                resolution = '0.05° x 0.05° (regridded from curvilinear grid)',
                calendar = 'Gregorian',
                )
        attributes.update(ds_wspd.attrs) #add all attributes from original file as well

        # Create the dataset
        ds = create_dataset(wind_speed_interpolated, wind_direction_interpolated, #np.full(wind_speed_interpolated.data.shape, None), 
            unique_lats, unique_lons, valid_time.data, attributes)
        
        # If needed, clip this dataset to the non rectangular domain
        if self.domain is not None:
            ds = clip_dataset(ds, self.domain)

        # Calculate daily means if needed
        if daily_mean:
            ds = ds.resample(time = 'D').mean(skipna = True) # resample by day and then take the mean
            ds.attrs['frequency'] = 'day'

        # Return the processed dataset
        return ds
    
    def path_wind_to_path_height(self, path_wind: str):
        """ Given the unprocessed path of the wind speed, give the path of the processed heights
        path_wind: the wind speed path 
        """
        if 'hist-1950' in path_wind or 'highresSST-present' in path_wind: period = 'hist'
        else: period = 'fut'
        groups = path_wind.split('/')
        path_height = '/'.join(groups[:6]) + f'/processed/highresmip/geometric_height/{period}/' + '/'.join(groups[9:])
        path_height = path_height.replace('_100m_day', '')
        # some added complexity cause the run and simulation are in the opposite order here
        _, run, *other = path_height.split('/')[-1].split('_')
        path_height = path_height.replace('_day', '')
        path_height = path_height.replace(run+'_', '')
        path_height = path_height.replace('.nc', '_'+run+'.nc')
        return path_height

    def path_wind_to_path_surface_wind(self, path_wind: str):
        """ Given the unprocessed path of the wind speed, give the path of the corresponding surface level daily mean wind speed
        path_wind: the wind speed path 
        """
        path_surface = path_wind.replace('wind_speeds','surface_wind_speeds')
        index = path_surface.rfind('/')
        path_surface = path_surface[:index+1] + 'sfcWind_' + path_surface[index+1:]
        return path_surface

    def interpolate_vertically_cmip6(self, wind_speed: xr.DataArray, height: xr.DataArray = None, surface: xr.DataArray = None, filter: float = None): 
        """Perform vertical interpolation of an array wind_speed to specified levels, using the specified method
            Also returns the alpha used in the power law, or None if another method than power is used
        wind_speed = the DataArray to interpolate
        height = optional, the accompanying heights, in case the data is on pressure levels
        surface = optional, surface level wind speed, for linear interpolation or alpha from two levels
        filter: optional, for linear interpolation: should the data be filtered based on the geopotential height (e.g. only use where the geopot height of the 1000 hPa level is above the desired level)
        """

        # Perform the interpolation
        match self.interpolation_method:

            case 'linear_single_level'|'loglinear_single_level':
                # (log)linear interpolation using only surface and 1000 hPa/850 hPa (chosen with select_level parameter in top function)
                # filter to filter out values for which 1000 hPa is below a certain height, e.g. the desired level for safe interpolation without extrapolation
                h_sfc = surface.height.data
                if h_sfc > self.level: raise ValueError('Warning: desired height is below surface level.')

                # Filter the data if needed
                if filter is not None:
                    height = height.where(height.height > filter)

                # Perform the actual interpolation
                if self.interpolation_method =='linear_single_level':
                    wind_speed_lev = surface.sfcWind + (self.level - h_sfc)*(wind_speed-surface.sfcWind)/(height.height-h_sfc)
                    # alternative: using wrf.interplevel method but more convoluted and slower
                else:
                    # linear interpolation in log height, like in scripts Hahmann
                    wind_speed_lev = surface.sfcWind + (np.log(self.level) - np.log(h_sfc))*(wind_speed-surface.sfcWind)/(np.log(height.height)-np.log(h_sfc))       
            
            case 'linear_multilevel'|'loglinear_multilevel':
                # linear interpolation using surface, 1000 hPa and 850 hPa if 1000 hPa is below the desired level
                # constructed so that here height and wspd contain both 850 and 1000 hPa
                h_sfc = surface.height.data
                if h_sfc > self.level: raise ValueError('Warning: desired height is below surface level.')
                
                # Select the right height and wind speed to interpolate from 
                if 'plev' in wind_speed.coords:
                    wind_high = xr.where(height.height.sel(plev = 100_000) > self.level, wind_speed.sel(plev = 100_000), wind_speed.sel(plev = 85_000))
                    wind_low = xr.where((height.height.sel(plev = 100_000) > h_sfc) & (height.height.sel(plev = 100_000) < self.level), wind_speed.sel(plev = 100_000), surface.sfcWind)
                    height_high = xr.where(height.height.sel(plev = 100_000) > self.level, height.sel(plev = 100_000), height.sel(plev = 85_000))
                    height_low = xr.where((height.height.sel(plev = 100_000) > h_sfc) & (height.height.sel(plev = 100_000) < self.level), height.sel(plev = 100_000), h_sfc)
                elif 'lev' in wind_speed.coords:
                    wind_high = xr.where(height.height.sel(lev = 100_000) > self.level, wind_speed.sel(lev = 100_000), wind_speed.sel(lev = 85_000))
                    wind_low = xr.where((height.height.sel(lev = 100_000) > h_sfc) & (height.height.sel(lev = 100_000) < self.level), wind_speed.sel(lev = 100_000), surface.sfcWind)
                    height_high = xr.where(height.height.sel(lev = 100_000) > self.level, height.sel(lev = 100_000), height.sel(lev = 85_000))
                    height_low = xr.where((height.height.sel(lev = 100_000) > h_sfc) & (height.height.sel(lev = 100_000) < self.level), height.sel(lev = 100_000), h_sfc)
                
                # Perform the actual interpolation
                if self.interpolation_method =='linear_multilevel':
                    wind_speed_lev = wind_low + (self.level - height_low.height)*(wind_high-wind_low)/(height_high.height-height_low.height)
                else:
                    wind_speed_lev = wind_low + (np.log(self.level) - np.log(height_low.height))*(wind_high-wind_low)/(np.log(height_high.height)-np.log(height_low.height))
            
            case 'log':
                z_0 = 1.52*10**(-4) #local roughness length over the ocean (deCastro 2018)
                #np.log is natural log with base e, np.log10 is log with base 10
                # in case no height was given, the data is at a specific level in m
                if height is None:
                   wind_speed_lev = wind_speed * (np.log(self.level/z_0)/np.log(wind_speed.height/z_0)) #apply log law (formula from deCastro 2018, eq 4)
                
                # in case the height was given, the data is on pressure levels
                else:
                   wind_speed_lev = wind_speed * (np.log(self.level/z_0)/np.log(height/z_0)) #apply log law (formula from deCastro 2018, eq 4)
                   wind_speed_lev = wind_speed_lev.rename_vars(height = 'wind_speed') # fix the array variable name
                   wind_speed_lev = wind_speed_lev.wind_speed # select the variable we need
            
            case 'power':
                # Set exponent alpha
                alpha = self.set_alpha(wind_speed = wind_speed, height = height, surface = surface) 
                
                # Data on a single height in m
                if height is None: 
                    wind_speed_lev = wind_speed * ((self.level/wind_speed.height)**alpha) # apply the power law (formula from e.g. Carvalho 2021, eq 1)
                
                # Data at pressure levels: height from geopot height file
                elif self.alpha == 'surface':
                    # apply the power law to surface cause alpha was determined from surface (formula from e.g. Carvalho 2021, eq 1)
                    wind_speed_lev = surface * ((self.level/10)**alpha) 
                    wind_speed_lev = wind_speed_lev.rename_vars(sfcWind = 'wind_speed')
                    wind_speed_lev = wind_speed_lev.wind_speed
                else:
                    # otherwise apply power law to the wind speed at pressure levels
                    wind_speed_lev = wind_speed * ((self.level/height)**alpha) 
                    wind_speed_lev = wind_speed_lev.rename_vars(height = 'wind_speed')
                    wind_speed_lev = wind_speed_lev.wind_speed
            
            case _: raise ValueError('This is not a valid interpolation method. Choose between power, log, linear_single_level, loglinear_single_level, linear_multilevel and loglinear_multilevel.')

        # Set alpha to none if the interpolation method is not power 
        if self.interpolation_method != 'power': alpha = None

        # Return the interpolated levels and alpha
        return alpha, wind_speed_lev

    def process_CMIP6_single_dataset(self, path: str, select_level: float = None, interpolate: bool = True, filter: float = None, nbuffer: int = 0, test_mode: bool = False):
        """Given path, process downloaded CMIP6 model data in a netcdf dataset with wind speed and wind direction. 
            Also returns alpha calculated from surface in case of power law with alpha from surface
        path: path to the file containing the u and v components
        select_level: optional, level to select if there are multiple levels in the file, expressed in Pa
        interpolate: True if interpolation should be performed, otherwise False
        filter: optional, to filter out datapoints based on the geopotential height for the linear interpolation
        nbuffer: optional, number of gridpoints to be added to the borders of the domain as a buffer for interpolation later on
        test_mode: if True, only the first 100 timesteps are loaded, for quick testing
        """

        # Read the NetCDF file
        ds = xr.open_dataset(path)

        # Restrict the domain if needed
        if self.lat is not None:
            ds = ds.sel(dict(lat = slice(self.lat[0],self.lat[1]), lon = slice(self.lon[0], self.lon[1])))
        elif self.domain is not None:
            ds = clip_dataset(ds, self.domain.buffer(nbuffer)) # add a buffer zone, used when regridding data (but does not match era5 and cerra so removed now)
        # could be added for Cerra and ERA5 too, now just left since we only use central grid points and add buffer along coastlines, also eliminating border effects

        # Load variables
        # If needed, select the correct level (and some sanity checks to make sure keyword arguments and interpolation method are not contradicting each other)
        if (select_level is None) & interpolate & ('multilevel' not in self.interpolation_method):
            raise ValueError('The wind speed and geopotential height are on two levels, 850 and 1000 hPa. Either use a multilevel interpolation method such as linear_multilevel or loglinear_multilevel to make use of both levels, or use the select_level keyword set to 85.000 or 100.000 to select one of the levels.')
        elif (select_level is not None) & interpolate & ('multilevel' in self.interpolation_method):
            raise ValueError('A multilevel interpolation method is used, but at the same time the select_level keyword is used to select a level. This is impossible, either the interpolation method or set the select_level keyword to None.')
        elif (select_level is not None):
            if 'plev' in ds.coords:
                ds = ds.sel(plev = select_level)
            else:
                ds = ds.sel(lev = select_level)
            ds.drop_dims(['lev', 'plev'], errors = 'ignore') # drop the now redundant dimension

        u = ds['ua']
        v = ds['va']

        time = ds['time']
        lat = ds['lat'] 
        lon = ds['lon'] 

        # Close the file again
        ds.close() 

        # Perform time slicing for test mode if needed
        if test_mode:
            u = u.isel(time=slice(1,100))
            v = v.isel(time=slice(1,100))
            time = time.isel(time=slice(1,100))

        # Interpolate if needed
        if interpolate:
            # Calculate the wind speed (before interpolating)
            wind_speed = calculate_wind_speed(u,v)

            # Load the geopotential height
            height_raw = xr.open_dataset(self.path_wind_to_path_height(path))
            # select the correct pressure level if only one is used
            if select_level is not None:
                if 'plev' in height_raw.coords:
                    height_raw = height_raw.sel(plev = select_level)
                elif 'lev' in height_raw.coords:
                    height_raw = height_raw.sel(lev = select_level)
            # align the dimensions: spatial differences due to buffers and clipping
            height_raw['lat'] = height_raw['lat'].astype('float32') # convert from float64 to float32, otherwise aligning does not work
            wind_speed['lat'] = wind_speed['lat'].astype('float32') # in some cases this is also needed if both were in float64
            wind_speed, height = xr.align(wind_speed, height_raw)

            # Load the surface wind data
            surface = xr.open_dataset(self.path_wind_to_path_surface_wind(path))
            # align the surface level wind with the wind speed at pressure level
            surface['lat'] = surface['lat'].astype('float32') # convert from float64 to float32, otherwise aligning does not work
            wind_speed, surface = xr.align(wind_speed, surface)

            # Interpolate wind speed to vertical levels
            alpha, wind_speed_lev = self.interpolate_vertically_cmip6(wind_speed, height = height, surface = surface, filter = filter)

            # in fact, vertical interpolation is not needed for power and log law (write out equations, factors from log and power law cancel out)
            wind_direction_lev = calculate_wind_direction(u, v) 

        # Just calculate wind speed and direction at surface level if no interpolation is needed
        else: 
            if select_level is None:
                raise ValueError("If no vertical interpolation is done, a level should be selected using the select_level keyword")
            wind_speed_lev = calculate_wind_speed(u,v)
            wind_direction_lev = calculate_wind_direction(u,v)

        # Construct the attributes to for the new, processed file 
        attributes = ds.attrs.copy()
        attributes.update(dict(
            data_source = f"Processed data from CMIP6, interpolation: {interpolate}, method: {self.interpolation_method}, {self.alpha}",
            height = self.level,
            height_units = 'm'))

        # Create the dataset
        # a little bit of hacking is needed in case of pressure level data: 
        # as the height files were not fully downloaded, the lat and lon don't match those of the full file (for CNRM and HadGEM)
        if interpolate is True: 
            # the wind direction is set to None because this method was not applied to wind direction
            print('Wind direction was not vertically interpolated, still on different heights from pressure levels. \nJust removed, except if log interpolation was used or basic power law with constant exponent.')
            attributes.update(dict(note =  'Vertical interpolation applied to the height calculated from the geopotential. Not applied to wind direction, so set to None.'))
            output = create_dataset(wind_speed_lev.values, np.full(wind_speed_lev.data.shape, None), wind_speed_lev.lat.values, wind_speed_lev.lon.values, time.values, attributes)
        else:
            output = create_dataset(wind_speed_lev.values, wind_direction_lev.values, lat.values, lon.values, time.values, attributes)

        # Downsample to 6 hr if data is at 3 hr frequency
        if '3hr' in output.attrs['frequency']:
            # depending on which timesteps are available, select 6 hourly frequency
            if 6 in np.unique(output.time.dt.hour):
                output = output.sel(time=(output['time'].dt.hour == 6)|(output['time'].dt.hour == 12)|(output['time'].dt.hour == 18)|(output['time'].dt.hour == 0))
            elif 1 in np.unique(output.time.dt.hour): # this is apperently only for FGOALS
                output = output.sel(time=(output['time'].dt.hour == 1)|(output['time'].dt.hour == 7)|(output['time'].dt.hour == 13)|(output['time'].dt.hour == 19))        
            output.attrs['frequency'] = '6h' # update the attribute

        # Check if there is still data remaining, otherwise throw error that something is wrong
        if len(output.time) == 0:
            raise ValueError('No more timesteps remaining, something went wrong in downsampling probably')
        
        # Return the output, and alpha if we're using the power law with alpha estimated from the surface
        if self.alpha == 'surface' and self.interpolation_method == 'power': return output, alpha
        return output

class HeightConvertor:
    """ Handles the conversion of geopotential height, downloaded for CMIP6 models, to geometric height"""         

    def geopotential_height_to_height_single(self, dataset: xr.DataArray, method: str = 'metpy'):
        """ Given an xarray dataset containing the geopotential height, return the geometric height above sea level
        dataset: the geopotential height dataset
        method: the method to use to calculate the geopotential height: either from the metpy or metview package or manually
        """

        # Convert geopotential height to right format
        g0 = 9.80655 # standard gravitational acceleration at mean sea level [https://en.wikipedia.org/wiki/Geopotential_height]
        geopot = dataset.zg.data * g0 # convert from geopotential height to geopotential
        geopot_units = units.Quantity(geopot, 'm^2/s^2')

        # Actually calculate the height 
        if method == 'metpy':
            height = metpy.calc.geopotential_to_height(geopot_units).magnitude  #in meters, but units are left alone
        elif method == 'metview': # after a brief investigation this method is worse than the ohters (very weird values instead of nans, very low and negative)
            height = metview.geometric_height_from_geopotential(geopot)
            height = np.reshape(height, geopot.shape)
        elif method == 'manual':
            R_e = 6_378_388
            height = geopot*R_e/(g0*R_e-geopot)

        # Construct an xarray object in the same shape/just edit the original one
        output = xr.zeros_like(dataset.zg)
        output.data = height
        output = output.rename('height')
        output.attrs = {'long_name': 'height calculated from geopotential', 'units': 'm'}
        
        return output

    def geopotential_height_to_height(self, folder: str, resolutions: list, experiments: list = ['hist-1950', 'highres-future' ], method: str = 'metpy'):
        """ Convert the geopotential height to geometric heights for the files in the folder and save as one concatenated file for each simulation (so fut/hist and high/low resolution)
        folder: the path to the folder to get the files from
        resolutions: strings to identify the different resolutions
        experiments: strings to identify the experiments
        """

        # Get the list of paths and sort them by simulation
        path_list = os.listdir(folder)
        paths_by_sim = split_into_simulations(path_list, resolutions = resolutions, experiments = experiments)

        # Loop over the simulations
        for paths_sim in list(paths_by_sim.values()):
            # Sort the files to make sure the time dimensions makes sense
            paths_sim.sort()

            # Create an empty list to save the heights from the files from this simulation in
            heights = []

            # Loop over the files, converting each of them to height
            for i, path in enumerate(paths_sim):
                zg = xr.open_dataset(folder+path)

                # Correct timesteps for FGOALS models
                if 'FGOALS' in path:
                    zg = self.resolve_timesteps_issue(zg)

                # One monthly file for HiRAM, to be interpolated here : not needed anymore cause file was sent by Tu
                #if 'HiRAM' in path and 'MONTHLY' in path: # specifically named file to avoid sorting of files being wrong
                #    zg = self.expand_monthly_to_daily(zg)
                #    # also replace the lat and lon values with the ones of the previous file because these are sometimes 10e-15 different which causes problems later
                #    prev = xr.open_dataset(folder+ paths_sim[i-1])
                #    zg['lat'] = prev.lat
                #    zg['lon'] = prev.lon

                height_file = self.geopotential_height_to_height_single(zg, method= method)
                heights.append(height_file)
            
            # Concatenate the files
            ds_sim = xr.concat(heights, dim = 'time')

            # Upsample the data to daily if it is monthly
            if 'mon' in path:
                ds_sim = self.expand_monthly_to_daily(ds_sim)
                print('Data was upsampled from monthly to daily')

            # Take daily means if 6 hourly
            if '6hr' in path:
                ds_sim = self.mean_6h_to_day(ds_sim)
                print('Data was averaged from 6 hours to daily')

            # Replace 0 geopotential height with closest value, because this results in infinite wind speeds when using power law
            if not np.sum(ds_sim == 0).data ==0:
                print(f'{np.sum(ds_sim == 0).data} zeros are linearly interpolated to avoid infinite wind speeds when applying power law')
            ds_sim = ds_sim.where(~np.isnan(ds_sim), np.inf) # replace nan with np.inf
            ds_sim = ds_sim.where(ds_sim != 0) # replace 0's with nans
            ds_sim = ds_sim.interpolate_na(dim = 'lat', method = 'linear') # interpolate nans (which were originally 0's)
            ds_sim = ds_sim.where(~np.isinf(ds_sim)) # reconvert np.inf to np.nan

            # Construct a filename and save the file
            if ('hist' in path) | ('present' in path) : time = 'hist'
            elif 'future' in path: time = 'fut'
            sim_name = '_'.join(paths_sim[0].split('_')[2:5])
            filename = f'.../geometric_height/{time}/{sim_name}.nc'  # TODO SET PATH TO SAVE GEOPOTENTIAL HEIGHT FILES
            ds_sim.to_netcdf(filename, mode="w",engine="netcdf4")
            print(f'Saved {filename}')

    def expand_monthly_to_daily(self, ds: xr.DataArray):
        """ Upsample a monthly dataset to a daily frequency
        ds: the dataset
        """
        # First apply resampling
        resampled = ds.resample(time= '1D').nearest()
        
        # Now some construction is needed to add the first part of the first month and the last part of the last month, which are missing if 
        # the day is e.g. the 15th: the resampling does not extrapolate
        time_extended = resampled.time.data
        if type(ds.time.values[0]) == np.datetime64: # for CNRM
            date_0 =np.datetime_as_string(time_extended[0])
            time_extended[0] = np.datetime64(date_0.replace(date_0.split('T')[0][-2:], '01')) # edit the first date
            date_last = np.datetime_as_string(time_extended[-1])
            time_extended[-1] = np.datetime64(date_last.replace(date_last.split('T')[0][-2:], '31')) # edit the last date
        else: # for HadGEM
            date_0 = time_extended[0]
            time_extended[0] = cftime.Datetime360Day(date_0.year, date_0.month, 1)
            date_last = time_extended[-1]
            time_extended[-1] = cftime.Datetime360Day(date_last.year, date_last.month, 30)

        # Switch out the time dimension with the extended timestamps
        resampled['time'] = time_extended 

        # Apply the resampling again (not possible to skip the first one because then the nearest item changes)
        resampled_full = resampled.resample(time = '1D').nearest()
        
        # Shift the timesteps to match the original once again
        if type(ds.time.values[0]) == np.datetime64: # for CNRM
            resampled_full['time'] = resampled_full.time + np.timedelta64(12,'h')
        else: # for HadGEM
            resampled_full['time'] = resampled_full.time + datetime.timedelta(hours = 12)

        return resampled_full

    def mean_6h_to_day(self, ds: xr.DataArray):
        """ Take daily means from 6 hourly data
        ds: the dataset 
        """
        daily_means = ds.resample(time = 'D').mean("time", skipna = True)
        return daily_means

    def resolve_timesteps_issue(self, ds: xr.DataArray):
        """Resolve the issue with timesteps being 1,1,1 for FGOALS model
        ds: the DataArray containing the geopotential heights
        """
        
        # Mask the values which are at a timestamp with 1,1,1 and are zero (sometimes there is actual data) with NaNs
        ds_masked = ds.zg.where(~((ds.time == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True)) & (ds.zg == 0)), np.nan)

        # Copy the timestamps
        time_edited = ds.time.data

        # Loop over the time dimension and check for 1,1,1 dates, and replace with the next timestamp
        counter = 0
        for i in range(len(time_edited)):
            if time_edited[i] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 0, 0, has_year_zero=True):
                counter = counter + 1 #+ np.sum(ds.zg.isel(time = i)==0).data # alternative to check numer of nans instead of number of timesteps
                time_edited[i] = time_edited[i-1] + datetime.timedelta(days = 1)
                #print(np.unique(ds.zg.isel(time = i)))
        
        # Print how many NaNs were introduced at how many timesteps
        if counter > 0: print(f'{np.sum(np.isnan(ds_masked)).data - np.sum(np.isnan(ds.zg)).data} NaNs added replacing zeros at timesteps (1,1,1), {counter} timesteps (1,1,1) corrected')
        
        # Replace the time and geopot heihgts components with the corrected values
        ds['time'] = time_edited
        ds['zg'] = ds_masked
        return ds