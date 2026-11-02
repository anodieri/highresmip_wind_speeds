# Import necessary packages

import numpy as np
import xarray as xr
import pandas as pd
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import datetime
import xskillscore as score
from scipy.stats import wasserstein_distance
import copy
from windrose import WindroseAxes
import regionmask
import warnings
from functools import reduce
import operator
import datetime #to time things

# Overall variables for plotting
#colors = ['blue','green', 'red', 'orange', 'purple', 'magenta', 'navy',
#        'teal', 'maroon', 'deeppink', 'olive','cyan', 'darkgoldenrod'] #list of colors to choose from
# color blind friendly version: Wong palette
colors = ['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7', '#000000']

# min and max domain lat and lon for the full download
latitude_min, latitude_max = 43.0, 61.5 
longitude_min, longitude_max = -13.0, 17.5
# and for the restricted domain
latitude_min_domain, latitude_max_domain = 50.7, 54.5
longitude_min_domain, longitude_max_domain = [0.8,4.7]

# Helper functions

def datetime_to_cftime(date):
    """Returns a cftime (Gregorian calendar) from a datetime
    date (string): date in the form yyyymmddhh
    Returns: cftime: date in cftime 
    Method adapted from future_wind, from Hahmann 2022
    """

    y,m,d,h = (
        date.dt.year, date.dt.month, date.dt.day, date.dt.hour)
    
    return cftime.DatetimeGregorian(y,m,d,h)

def convert_calendar(date, calendar = cftime.DatetimeGregorian):
    """Convert cftime objects between different calendars, handling the necessary exceptions for 360 day calendar
    date: the date to convert to a new calendar
    calendar: the calendar to convert to
    """

    try: 
        # use the built-in function to switch between calendars
        converted_date = calendar(date.year, date.month, date.day, date.hour, date.minute, date.second)
    except ValueError:
        # this handles february 29 and 30 from the 360 day calendar by putting data from that date two days before
        # with a shift in the seconds to ensure unique times
        # when calculating errors, these are ignored as they are not present in ERA5 (when lining up calendars)
        converted_date = calendar(date.year, date.month, date.day-2, date.hour, date.minute, date.second+5)
        #print(f'Exception: converted {date} into {converted_date}')
    return converted_date

def plot_domain(lat, lon):
    """ Plot a rectangular domain on the map
    lat: array containing the min and max latitude of the domain
    lon: array containing the min and max longitude of the domain
    """

    #  Create the map
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Plot the domain on the map, add borders and coastlines
    ax.set_extent([longitude_min, longitude_max, latitude_min, latitude_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black') # Coastlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder = -1) # Gridlines behind the other features
    plt.gca().add_patch(matplotlib.patches.Rectangle((lon[0],lat[0]),lon[1]-lon[0],lat[1]-lat[0], edgecolor='red',facecolor='none',lw=2)) # Rectangular domain
    
    # Show the map with the positions
    plt.show()

def define_frequency_resample(freq: str):
    """ Convert normal frequency strings to the one needed for xarray resample"""
    match freq.lower():
        case 'month': return 'ME' 
        case 'year': return 'YE'
        case 'season': return 'QS-DEC' # split in three month periods, starting at dec first so this becomes DJF, MAM, JJA, SON
        case 'day': return 'D'
        case _: raise ValueError('This period is not defined. Either implement it or choose one of month, year, season, day')

def close_factors(number):
    ''' 
    find the closest pair of factors for a given number
    Source - https://stackoverflow.com/a/77243426
    '''
    factor1 = 0
    factor2 = number
    while factor1 +1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1
        
    return factor1, factor2

def almost_factors(number):
    '''
    find a pair of factors that are close enough for a number that is close enough
    Source - https://stackoverflow.com/a/77243426
    '''
    while True:
        factor1, factor2 = close_factors(number)
        if 2/3 * factor1 <= factor2: # the fraction in this line can be adjusted to change the threshold aspect ratio
            break
        number += 1
    return factor1, factor2

class WindDataSet():
    """ Processed dataset containing wind speed and wind direction
    split up into SpatialWindDataSet and PointWindDataSet, so not really meant to explicitly be created
    """

    def __init__(self, name: str, path: str = None, data = None, test_mode: bool = False, chunks: int  = None, daily_mean: bool = False, freq: str = None):
        """ initialise a wind dataset 
        path: String specifying the location of the processed dataset (netcdf file)
        name: the name of the dataset
        data: alternative to path, a dataset can be given directly
        test_mode: Boolean, if True the dataset is sliced (first 5000 timesteps) for quicker loading of data
        chunks: specify chunks to use dask arrays instead of xarray dataset 
        daily_mean: should daily means of the data be calculated? can also already be done in processor.py
        freq: optional, to downsample to 3h or 6h if needed, not to be combined with daily_mean = true
        """

        # Construct the data
        winddata = self.construct_data(path = path, data = data, test_mode = test_mode, chunks = chunks, daily_mean = daily_mean, freq = freq)

        # Initialise the fields
        self.name = name
        self.data = winddata
        
    def construct_data(self, path: str = None, data = None, test_mode: bool = False, chunks: int = None, daily_mean: bool = False, freq: str = None):
        """ Load and prepare the data to use for this WindDataSet
        path: String specifying the location of the processed dataset (netcdf file)
        data: alternative to path, a dataset can be given directly, so either path or data has to be specified
        test_mode: Boolean, if True the dataset is sliced (first 5000 timesteps) for quicker loading of data
        chunks: specify chunks to use dask arrays instead of xarray dataset
        daily_mean: should daily means of the data be calculated? can also already be done in processor.py
        freq: optional, to downsample to 3h or 6h if needed, not to be combined with daily_mean = true
        """

        # Load the data depending on whether a path or a dataset is given, and if test mode or chunks are needed
        if data is None:
            if test_mode:
                winddata = xr.open_dataset(path).isel(time = slice(0,5000))
            elif chunks is None:
                winddata = xr.open_dataset(path)
            else:
                winddata = xr.open_dataset(path, chunks = chunks)
        elif path is None:
            winddata = data

        # convert date format from np.datetime64 to cftime if needed
        # does not work to do this in processor.py (some bug when saving converts it back to datetime64 for specific calendars)
        # check the type of the first timestep
        if isinstance(winddata.time.data[0], np.datetime64):
            time_cf = [] #empty array to save time in new format
            for timestep in winddata.time : #loop over all timesteps (in datetime64) format
                cfdate = datetime_to_cftime(timestep) #convert to cftime using helper funciton
                time_cf.append(cfdate) #add to array

            #now construct the actual dimension with attributes
            time = xr.DataArray(
                data=time_cf,
                dims="time",
                coords=dict(
                    time=(["time"], time_cf)
                ),
                attrs=dict(
                    long_name="time",
                    #units="YYYY-MM-DD hh:mm", 
                ),
            )

            #convert the time dimension to the new time format 
            winddata['time']=time
        
        #if the first day of the next year is also included, delete it (handiest to do this when datetime formats are uniform)
        last_timestep = winddata.time.data[-1] #get last timestep
        if last_timestep.day == 1 & last_timestep.month == 1: # check if its jan 1st
            winddata = winddata.isel(time = slice (0, len(winddata.time)-1)) #select every timestep but the last one

        # Calculate daily means if needed
        if daily_mean:
            winddata = winddata.resample(time = 'D').mean(skipna = True) # resample by day and then take the mean
        
        # Downsample to the required frequency if needed
        if freq is not None:
            match freq:
                case '6h':
                    winddata = winddata.sel(time=(winddata['time'].dt.hour == 6)|(winddata['time'].dt.hour == 12)|(winddata['time'].dt.hour == 18)|(winddata['time'].dt.hour == 0))
                case '3h':
                    winddata = winddata.sel(time=(winddata['time'].dt.hour == 6)|(winddata['time'].dt.hour == 12)|(winddata['time'].dt.hour == 18)|(winddata['time'].dt.hour == 0)|(winddata['time'].dt.hour == 3)|(winddata['time'].dt.hour == 9)|(winddata['time'].dt.hour == 15)|(winddata['time'].dt.hour == 21))
                case '1h': pass # in this case no action is needed    

        # return the resulting data
        return winddata

    def decide_subclass(self, data):
        """ Decide if this data should be PointWindDataSet or a SpatialDataSet"""
        
        # Check if the latitude dimension is a single float, then PointWindDataSet otherwise SpatialWindDataSet
        if isinstance(data.lat, (np.floating, float)):
            return 'Point'
        elif data.lat.size == 1 and data.lon.size == 1:
            return 'Point'
        else: return 'Spatial'

    def __repr__(self):
        """ Representation of the WindDataSet for printing
        """
        return f'Name: {self.name} \n {self.data.data_vars} \n' #only print vars, not dims and attrs

    def __add__(self, winddataset: 'WindDataSet'):
        """Implementing the addition of two WindDataSets, data concatenated along time variable
        """

        #set the name of the sum of the two WindDataSets
        if self.name == winddataset.name:
            name = self.name #if the names are equal, just use this name
        else:  
            name = f'{self.name} concatenated with {winddataset.name}' # if the names are different, concatenate them

        #construct the data of the new WindDataSet, conflicting attributes are dropped
        data = xr.concat([self.data, winddataset.data], dim='time', combine_attrs='drop_conflicts')

        #return the newly constructed WindDataSet (point or spatial depending on original object)
        return self.__class__(name, data = data)

    def sel(self, indexers, method: str = None):
        """ selection from https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html
        returns a new WindDataSet where the selection was applied
        indexers: dimensions along which to index, e.g. dict(time=model.data['time'].dt.month == 2), dict(lat= [50,51], lon = 3), dict(lat = slice(54,59), lon = slice(0.2, 5))
        method: selection method to use  for inexact matches: None, "nearest", "pad", "ffill", "backfill", "bfill"
        """
        
        # Apply the selection and return result
        data = self.data.sel(indexers, method)

        # Create a new winddataset as output, depending on whether the result is a point or a spatial winddataset
        if self.decide_subclass(data) == 'Point':
            return PointWindDataSet(name = self.name, data = data, remove_nan = False)
        else:
            return SpatialWindDataSet(name = self.name, data  = data)

    def align(self, other, dropna: bool = False):
        """Align the dimensions of this winddataset with the other given winddataset 
        other: the winddataset to align the dimensions to
        dropna: should the nans of the other dataset be dropped before aligning?"""

        # Make a copy of the original winddataset to not alter the original
        output = copy.copy(self)

        # If both winddatasets are spatial, check if their grids are the same, and if not raise a warning
        if isinstance(self,SpatialWindDataSet) and isinstance(other,SpatialWindDataSet):
            if not self.same_grid(other):
                warnings.warn(f'These two winddatasets do not have a common grid, so aligning dimensions will also impact the grids.')

        # Perform the alignment
        if dropna:
            aligned_data, _ = xr.align(self.data, other.data.dropna('time'), join = 'inner')
        else:
            aligned_data, _ = xr.align(self.data, other.data, join = 'inner')

        # Set the aligned data as the data for the output winddataset
        output.data = aligned_data

        if len(output.data.time) == 0: print('Warning: no timesteps left. Check if the two winddatasets have the same calendar, apply the uniform_timesteps function if needed')

        return output

    def shift_timestamps(self, days: float = 0, hours: float = 0, minutes: float = 0):
        """ Shift the timesteps by the specified amount of hours, days and minutes
        """

        # Change the timestamps of the object
        self.data['time'] = self.data.time + datetime.timedelta(days = days, hours = hours, minutes = minutes)
        print(f'Timestamps shifted by {days} days, {hours} hours and {minutes} minutes')

    def uniform_timestamps(self):
        """ Convert the timestamps of a winddataset to a uniform, Gregorian calendar with timesteps at 0,6,12,18 or 0 for daily data
        actually alters the data, does not return an altered copy
        """
        # Get the first timestamp
        timestamp = self.data.time.data[0]

        # Shift timestamps if the model has special timestamps, otherwise mean does not work
        # these are the options that appear in the models currently downloaded, this might need to be extended
        if timestamp.hour in [5,11,17,23]:
            self.shift_timestamps(hours = 1)
        elif timestamp.hour in [7,13,19,1]:
            self.shift_timestamps(hours = -1)
        if self.data.frequency == 'day' and timestamp.hour == 12: # specific for the daily downloads
            self.shift_timestamps(hours = -12)
        if timestamp.minute == 30:
            self.shift_timestamps(minutes = -30) #this only appears as 7:30, by the previous if the 7 is shifted to 6 and then the 30 minutes are removed

        # If the calendar is not Gregorian, convert it to Gregorian
        if not isinstance(timestamp, cftime._cftime.DatetimeGregorian):
            self.convert_calendar()

    def _sanity_check_calendars(self):
        """a little sanity check to see if all dates are present
        """

        #create a histogram to inspect the months
        months=[]
        for i in self.data.time.data:
            months.append(i.month)
        bins_to_use = np.linspace(0.5,12.5,13)
        counts, bins, patches = plt.hist(months, bins=bins_to_use)
        plt.title('Months')
        plt.show()
        # Print the counts and bins
        print('Months')
        print("Counts:", counts)
        print("Bins:", bins)

        #do the same for the hours
        hours=[]
        for i in self.data.time.data:
            hours.append(i.hour)
        bins_to_use=np.linspace(-0.5,23.5,25)
        counts, bins, patches = plt.hist(hours, bins=bins_to_use)
        plt.title('Hours')
        plt.show()
        # Print the counts and bins
        print('Hours:')
        print("Counts:", counts)
        print("Bins:", bins)

        #and for the days
        days=[]
        for i in self.data.time.data:
            days.append(i.day)
        bins_to_use=np.linspace(0.5,31.5,32)
        counts, bins, patches = plt.hist(days, bins=bins_to_use)
        plt.title('Days')
        plt.show()
        # Print the counts and bins
        print('Days')
        print("Counts:", counts)
        print("Bins:", bins)   

        #and for the years
        years=[]
        for i in self.data.time.data:
            years.append(i.year)
        bins_to_use=np.linspace(1969.5,2050.5,82)
        counts, bins, patches = plt.hist(years, bins=bins_to_use)
        plt.title('Years')
        plt.show()
        # Print the counts and bins
        print('Years')
        print("Counts:", counts)
        print("Bins:", bins)

    def _sanity_check_leap_year(self):
        """ check if/how many leap years there are 
        """

        # Select the leap days:
        feb = self.data.sel(time=self.data['time'].dt.month == 2) #select the data in February
        if len(feb.time) != 0:  # check if there is any data for February, for observations this is not always the case
            leapdays = feb.sel(time=feb['time'].dt.day == 29) # Select the 29th
            if len(leapdays.time) == 0: # else gives error if leapdays is empty
                number_leap_years = 0
            else:
                number_leap_years = len(leapdays.groupby('time.year'))# group by year and get the number of groups
            
            # Print the results
            print(f"Number of leap years: {number_leap_years}")
            print(f'First date: {self.data.time[0].data}')
            print(f'Last date: {self.data.time[-1].data}')
        else:
            print('No february in the dataset')

    def convert_calendar(self):
        """convert time dimension of the current WindDataSet to Gregorian (default of convert_calendar)
        (not really tested on PointWindDataSets since observations are immediately saved in the right format)
        """

        # convert the time dimension to the Gregorian calendar used by reanalysis data
        # Gregorian calendar: hardcoded, can be generalised to others by altering convert_calendar function
        converted_time = [convert_calendar(date) for date in self.data.time.values]
        #construct the time dimension in the new calendar
        new_time = xr.DataArray(
            data=converted_time,
            dims="time",
            coords=dict(time=(["time"], converted_time)),
            attrs=dict(long_name="time", calendar='converted to Gregorian'),
        )

        # set the new time in a copy of the original winddataset
        output = copy.copy(self)
        output.data['time'] = new_time

        # Print a little warning:
        print(f'Calendar of {self.name} was converted to Gregorian calendar, with some loss of timesteps possible')

        return output

    def select_season(self, season: str):
        """Select the specified season, and return a new WindDataSet containing only data from that season
        season: String winter, spring, summer, fall/autumn
        """

        output = copy.copy(self) #step 1: make a copy
        match season:
            case 'winter':
                output.data = output.data.sel(time=(output.data['time'].dt.month == 12)|
                    (output.data['time'].dt.month == 1)|(output.data['time'].dt.month == 2))
            case 'spring':
                output.data = output.data.sel(time=(output.data['time'].dt.month == 3)|
                    (output.data['time'].dt.month == 4)|(output.data['time'].dt.month == 5))
            case 'summer':
                output.data = output.data.sel(time=(output.data['time'].dt.month == 6)|
                    (output.data['time'].dt.month == 7)|(output.data['time'].dt.month == 8))
            case 'autumn'|'fall':
                output.data = output.data.sel(time=(output.data['time'].dt.month == 9)|
                    (output.data['time'].dt.month == 10)|(output.data['time'].dt.month == 11))
        #add the season to the name
        output.name = f'{output.name} {season}'
        return output

    def plot_wind_rose(self):
        """ Plot a basic wind rose
        """

        # Create a new figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='windrose')

        #construct bins
        bins = np.arange(0,30, 5) #a standard set of bins
        print("Warning: standard set of bins with maximum value of 30, might ignore some values")

        # Plot the data with specified bins
        ax.bar(self.data.wind_direction.data.flatten(), self.data.wind_speed.data.flatten(), normed=True, opening=0.9, 
            cmap= plt.cm.viridis, edgecolor='black', bins = bins)
        #label_position=ax.get_rlabel_position()
        #ax.text(np.radians(label_position+10),ax.get_rmax()/2.,'Frequency (%)',
        #    rotation=label_position,ha='center',va='center') #set polar axis label
        plt.title(f"{self.name}") # Add a title
        ax.legend(title="Wind speed (m/s)") # Add a legend

        return ax

    def plot_histogram_wspd(self, nbins: int):
        """Plot a histogram of the windspeed
        nbins =  number of bins to use
        """

        plt.hist(self.data.wind_speed.data.flatten(), bins=nbins, density=True, edgecolor='black') #note: fancier versions in seaborn package but these take very long
        plt.title(f'{self.name} wind speed distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Wind speed (m/s)')
        plt.show()
    
    def plot_time_series_line(self, ax, groupby: str, annotate: bool = False, return_values: bool = False, **kwargs):
        """" plot a line of the time series, without the surrounding plot elements
        ax: plot element to add this line to
        groupby: unit of time to group by, year or month or hour
        return_values: should the grouped mean be returned?
        annotate: should the name of the dataset be added to the line as a text label?
        """

        # Calculate the grouped means for the WindDataSet, different for point and spatial datasets due to different dimensions
        grouped_mean = self.grouped_mean('time.' + groupby)

        # Plot this time series line, split into groupby is needed for the x values
        if groupby == 'year':
            ax.plot(grouped_mean.year, grouped_mean.wind_speed, **kwargs) # plot the line
            if annotate:
                color = plt.gca().lines[-1].get_color() # get the color of the line that was last plotted, to make text same color
                plt.text(grouped_mean.year[-1] + 0.2, grouped_mean.wind_speed[-1], self.name, color = color) # add text annotation to the line
                        #weight = 'bold' to get bolded text, but can also be done afterwards
        if groupby == 'month':
            ax.plot(grouped_mean.month, grouped_mean.wind_speed, **kwargs)
            if annotate:
                color = plt.gca().lines[-1].get_color()
                plt.text(grouped_mean.month[-1] + 0.2, grouped_mean.wind_speed[-1], self.name, color = color) # add text annotation
        if groupby == 'hour':
            ax.plot(grouped_mean.hour, grouped_mean.wind_speed, **kwargs)
            if annotate:
                color = plt.gca().lines[-1].get_color()
                plt.text(grouped_mean.hour[-1] + 0.2, grouped_mean.wind_speed[-1], self.name, color = color) # add text annotation
      

        # Return ax and the actual group means if needed
        if return_values:
            return ax, grouped_mean
        else:
            return ax

    def plot_time_series(self, groupby: str = 'year', ax = None, **kwargs):
        """ plot time series
        groupby: unit of time to group by, year, month, or hour
        ax: optional, to add the plot to
        kwargs: optional extra keyword arguments for the plot, e.g. color
        """
        # Create the figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
    
        # Calculate the grouped means for the WindDataSet, different for point and spatial datasets due to different dimensions
        grouped_mean = self.grouped_mean('time.' + groupby)

        # Plot depending on the group
        if groupby == "year":
            self.plot_time_series_line(ax, groupby, **kwargs)
            ax.set_title(f'Yearly mean wind speed {self.name}')
            ax.set_xlabel('Year')

        elif groupby == "month":
            self.plot_time_series_line(ax, groupby, **kwargs)
            ax.set_title(f'Monthly mean wind speed {self.name}')
            ax.set_xlabel('Month')
            ax.set_xticks(labels=['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ticks=np.arange(1, 13))
        
        elif groupby == "hour":
            self.plot_time_series_line(ax, groupby, **kwargs)
            ax.set_title(f'Hourly mean wind speed {self.name}')
            ax.set_xticks(ticks=np.arange(0, 24))

        # Add plot generalities
        ax.set_ylabel('Wind speed (m/s)')
        ax.grid(True)

        return ax
    
    def plot_density(self, nbins: int, ax = None, color: str = 'black', label: str = None, ylim: float  = None, plot_mean: bool = False, 
        offset_y: float = 0.02, offset_x: float = 0.5,  text_kwargs: dict = {}, **kwargs):
        """ plot the probability density for a winddatset
        nbins: number of bins used to estimate the density function
        ax: optional, ax to plot on
        color, label: for plotting
        ylim: optional, y axis limit
        plot_mean: optional, plot a line and text indicating the mean or not
        offset_x, offset_y: optional, offset when adding text indicating the mean
        kwargs: extra arguments for plot, like alpha, linewidth, ...
        """
        # Create the figure if needed
        if ax is None:
            fig, ax = plt.subplots()

        # Estimate the probability density from the data without nans
        data_no_nan = self.data.wind_speed.data.flatten()
        data_no_nan = data_no_nan[~np.isnan(data_no_nan)]
        density = scipy.stats.gaussian_kde(data_no_nan)
        n, x, _ = ax.hist(data_no_nan, bins=nbins, density=True, alpha = 0) # plot the histogram seethrough

        # Plot it
        if label is None: label = self.name
        ax.plot(x, density(x),  color = color, label = label, **kwargs)
        ax.legend(frameon = False)
        ax.spines[['right', 'top']].set_visible(False) # remove right and top axis
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Wind speed (m/s)')

        # Set ylim if needed
        if ylim is not None:
            ax.set_ylim(top = ylim)
            ax.set_yticks(np.linspace(0, ylim, 4))

        # Add the mean if needed
        if plot_mean:
            mean = data_no_nan.mean(dtype=np.float64)
            ax.vlines(x = mean, ymin = 0, ymax = density(mean), colors= color, **kwargs)
            ax.text(x = mean + offset_x, y = offset_y, s = round(mean, 2), color = color, **text_kwargs)

        return ax
        
class PointWindDataSet(WindDataSet):
    """ Processed single point dataset containing wind speed and wind direction
    """

    def __init__(self, name: str, path: str = None, data = None, test_mode: bool = False, chunks: int  = None, daily_mean: bool = False, freq: str = None,
        threshold: float = 1000, fill_nan: bool = False, max_gap = None, remove_nan: bool = False):
        """ initialise a point wind dataset and filter speed for very high values
        same attributes as WindDataSet __init__
        threshold: value to filter: all values above the threshold are set to nan
        fill_nan: should the nan values be filled in by interpolating?
        max_gap: optional, the maximum gap to bridge when interpolating for nans, important to distinguish between prolonged periods of inactive lidar and occasional gaps in data
        remove_nan: should the nan values be removed?
        """
        # Use initialiser of the parent class
        super().__init__(name, path, data, test_mode, chunks, daily_mean, freq)

        # Filter the data for values that are too high, should not be necessary because these are already removed when processing the raw lidar data to avoid problems with daily means
        self.filter(threshold)

        # Fill in the missing timesteps in the data by resampling depending on the frequency
        # missing timesteps appear sometimes individually and sometimes in longer inactive periods of the lidar
        self.resample()

        # Interpolate the nan values if needed
        if fill_nan:
            # Determine the maximal gap that the interpolation can bridge based on the frequency if not given
            if max_gap is None:
                if 'day' in self.data.frequency:
                    max_gap = datetime.timedelta(days = 3) # set a default value for the gap if not given, determined by testing stuff in notebook observations.ipynb
                elif '6h' in self.data.frequency:
                    max_gap = datetime.timedelta(hours = 36)
            self.fill_nans(max_gap)

        # Remove the nan values, easier to match timesteps with models/reanalysis if the nans are removed
        # pay attention when making raw time series (with all timesteps) tho, 
        # will result in continuous line with jumps in the dates
        if remove_nan:
            num_nan = np.sum(np.isnan(self.data.wind_speed)).data # get number of nan values in the data
            print(f'{num_nan} timesteps with nan values are removed from {self.name}')
            self.data = self.data.dropna('time')

    def grouped_mean(self, groupby: str):
        """Calculate the mean values over the full dataset grouped by the specified string
        """
        return self.data.groupby(groupby).mean(dtype=np.float64)

    def grouped_median(self, groupby: str):   
        """Calculate the median values over the full dataset grouped by the specified string
        """
        return self.data.groupby(groupby).median()
    
    def sampled_mean(self, groupby: str):
        """ Calculate the mean values over the full dataset sampled by the specified string
        so a mean for every e.g. month of the full dataset, as opposed to grouped mean which would give 12 means, one for each month
        groupby: the unit of time to sample by, month, year, season or day
        """
        return self.data.resample(time = define_frequency_resample(groupby)).mean(dim = 'time', dtype = np.float64)

    def sampled_median(self, groupby: str):
        """ Calculate the median values over the full dataset sampled by the specified string
        so a median for every e.g. month of the full dataset, as opposed to grouped median which would give 12 medians, one for each month
        groupby: the unit of time to sample by, month, year, season or day
        """
        return self.data.resample(time = define_frequency_resample(groupby)).median(dim = 'time')
        
    def plot_location(self, ax = None, offset_lat: float = 0.1, offset_lon: float = 0, label: str = None, color: str  = 'r', alpha_gl: float = 0.5, **kwargs):
        """Plot the location of this pointdataset on a map
        ax: optional, plot the location on top of the given ax
        offset_lat: the offset of the text from the object in the lat direction
        offset_lon: same but in the lon direction
        label: optional, the text to be added to the point, if not given this is just the name of the dataset
        color : color of the dot
        alpha_gl: alpha of the gridlines
        """
        
        # Create the map if ax is not given
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            # Add borders and coastlines and gridlines
            ax.set_global()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            gls = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha = alpha_gl)
            gls.top_labels=False   # suppress top labels
            gls.right_labels=False # suppress right labels
            ax.set_extent([longitude_min_domain, longitude_max_domain, latitude_min_domain, latitude_max_domain], crs=ccrs.PlateCarree())

        # Plot the position on the map
        ax.scatter(self.data.lon, self.data.lat, color = color, **kwargs)
        if label is None: label = self.name
        plt.text(self.data.lon+offset_lon, self.data.lat+offset_lat, label, color = color)

        return ax

    def filter(self, threshold: float = 1000):
        """Filter out the wind speed values above a certain threshold in m/s by setting them to nan, without creating a new wind speed object
        threshold: the value above which the wind speeds are set to nan
        """
        self.data = self.data.where(self.data.wind_speed<threshold)

    def resample(self):
        """ Resample the data so that missing timesteps are filled in with nans
        """

        # depending on the frequency of the data, resample with 24 or 6h
        if 'day' in self.data.frequency: 
            self.data = self.data.resample(time = '24h').first(keep_attrs = True)
        elif '6h' in self.data.frequency:
            self.data = self.data.resample(time = '6h').first(keep_attrs = True)
        else: print('Frequency not recognized in the resample function')

    def fill_nans(self, max_gap: datetime.timedelta, method: str = 'linear'):
        """Interpolate the nan values in the data, without creating a new winddataset
        method: interpolation method, e.g. linear or nearest, see https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interpolate_na.html#xarray.DataArray.interpolate_na for more options
        max_gap: maximal gap to bridge with the interpolation
        """
        
        # Perform the interpolation on both the wind speed and the wind direction (if it is present in the winddataset, not always the case for lidar interpolated)
        self.data['wind_speed'] = self.data.wind_speed.interpolate_na(dim ='time', method = method, max_gap = max_gap, keep_attrs = True)
        if 'wind_direction' in list(self.data.data_vars):
            self.data['wind_direction'] = self.data.wind_direction.interpolate_na(dim ='time', method = method, max_gap = max_gap, keep_attrs = True)

    def same_grid(self, pointdataset: 'PointWindDataSet', weak: bool = False):
        """ Checks if this dataset has the same lat and lon values as the given dataset
        pointdataset: dataset to compare to
        weak: should there be an exact match? in that case weak = False, if weak = True both being pointdatasets is enough"""

        if weak:
            return(type(pointdataset) == PointWindDataSet)
        else:
            if self.data.lat != pointdataset.data.lat or self.data.lon != pointdataset.data.lon:
                return False
            else: return True

    def plot_nans(self, **kwargs):
        """ Plot a timeseries of the dataset, only showing where the nans are in the wind speed
        """
        # Get the number of timesteps and the number of nans
        n_steps = len(self.data.time) # number of timesteps
        num_nans = np.sum(np.isnan(self.data.wind_speed)).data # get number of nans

        # Create the figure
        fig, ax = plt.subplots(figsize = (15,1))
        plt.scatter(np.arange(n_steps), np.isnan(self.data.wind_speed), s =0.2, **kwargs)

        # Add plot decoration
        ax.set_title(f'{self.name} : {num_nans} NaNs in wind speed')
        #ax.set_xlabel('Time')
        plt.xticks(np.arange(n_steps)[::round(n_steps/7 )], rotation = 45,
            labels=[str(step.dt.month.data)+'-'+str(step.dt.year.data) for step in self.data.time[0:n_steps][::round(n_steps/7)]])
        ax.set_ylabel('To NaN or not to NaN')
        ax.set_yticks([0,1],['Not NaN', 'NaN'])
        #plt.show()

class SpatialWindDataSet(WindDataSet):
    """ Processed spatial dataset containing wind speed and wind direction
    """

    def __init__(self, name: str, path: str = None, data = None, test_mode: bool = False, chunks: int = None, daily_mean: bool = False, freq: str = None):
        """ initialise a spatial wind dataset, using basic setup and reversing the latitude dimension if needed
        same attributes as WindDataSet __init__
        """
        
        # use initialiser of the parent class
        super().__init__(name, path, data, test_mode, chunks, daily_mean, freq)

        # if the latitude dimension is reversed, fix it (this is the case for ERA5)
        if self.data.lat.size > 1: 
            if self.data.lat[-1] < self.data.lat[0]:
                self.data = self.data.reindex(lat = self.data.lat[::-1])

    def __sub__(self, winddataset: 'SpatialWindDataSet'):
        """Implementing subtraction of two WindDataSets, by subtracting the variables from each other
        mainly meant for maps so not implemented for PointWindDataSets (but could be extended, deal with grid in some way)
        """
        
        #interpolate to the grid of the first WindDataSet
        winddataset = winddataset.interpolate_horizontally(self)

        #Convert the calendars to Gregorian, to be able to compare
        winddataset = winddataset.convert_calendar()
        print('calendars converted to Gregorian')
        self_cal = self.convert_calendar()

        #return the result of the subtraction (alignment of dimensions happens automatically)
        return self.__class__(name = f'{self.name} - {winddataset.name}', data = self_cal.data-winddataset.data)

    def interpolate_horizontally(self, grid: 'SpatialWindDataSet', method: str = 'linear', fill_value: str = "extrapolate"):
        """horizontal interpolation of a WindDataSet, returns a new WindDataSet on the new grid
        grid: the WindDataSet from which the grid can be taken
        method:  interpolation method, can be linear, nearest, ... 
        fill_value: str indicating whether out of range data should be extrapolated or not
        (see https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html#xarray.DataArray.interp)
        """

        # Perform the interpolation, extrapolating outside of the domain
        data = self.data.interp(lat = grid.data.lat, lon = grid.data.lon, method=method,  
            kwargs={"fill_value": fill_value}) 

        # Return a point or spatial winddataset depending on the resulting data
        if self.decide_subclass(data) == 'Point':
            return PointWindDataSet(name = self.name, data = data, remove_nan = False)
        else:
            return SpatialWindDataSet(name = self.name, data  = data)

    def apply_nans_other(self, other: 'SpatialWindDataSet'):
        """ Given a spatial winddataset, set values from this winddataset to nan apply the same mask as the given dataset(e.g. land sea, certain domain). 
        Alters the wind dataset in place.
        other: spatial winddataset to copy the mask from, is assumed to be on the same grid  
        """
        # Check if the datasets have the same horizontal grid
        if not self.same_grid(other):
            raise ValueError('NaNs of the other dataset cannot be applied since the datasets do not have the same horizontal grid.')

        # Apply the mask of nan gridpoints
        mask = np.isnan(other.data.wind_speed.mean(dim = 'time')) # check if the mean is nan, if yes that means the full time series is nan at this location since mean of xarray ignores nans by default
        self.data = self.data.where(~ mask, np.nan) # Apply the mask

    def grouped_mean(self, groupby: str, spatial: bool = False):
        """Calculate the mean values over the full dataset grouped by the specified string, also taking the mean over the domain
        groupby: str, entity to group by
        spatial: if True, spatial dimensions are kept, otherwise the mean is also taken
        """
        spatial_grouped_mean = self.data.groupby(groupby).mean(dtype=np.float64)
        if spatial: return spatial_grouped_mean
        else: return spatial_grouped_mean.mean(['lat','lon'], dtype = np.float64)

    def grouped_median(self, groupby: str, spatial: bool = False):
        """Calculate the mean values over the full dataset grouped by the specified string, also taking the mean over the domain
        groupby: str, entity to group by
        spatial: if True, spatial dimensions are kept, otherwise the mean is also taken
        """
        spatial_grouped_median = self.data.groupby(groupby).median()
        if spatial: return spatial_grouped_median
        else: return spatial_grouped_median.median(['lat','lon'])

    def sampled_mean(self, groupby: str, spatial: bool = False):
        """ Calculate the mean values over the full dataset sampled by the specified string
        so a mean for every e.g. month of the full dataset, as opposed to grouped mean which would give 12 means, one for each month
        groupby: the unit of time to sample by, month, year, season or day
        """
        spatial_sampled_mean =  self.data.resample(time = define_frequency_resample(groupby)).mean(dim = 'time', dtype = np.float64)
        if spatial: return spatial_sampled_mean
        else: return spatial_sampled_mean.mean(['lat', 'lon'], dtype = np.float64)

    def sampled_median(self, groupby: str, spatial: bool = False):
        """ Calculate the median values over the full dataset sampled by the specified string
        so a median for every e.g. month of the full dataset, as opposed to grouped median which would give 12 median, one for each month
        groupby: the unit of time to sample by, month, year, season or day
        """
        spatial_sampled_median =  self.data.resample(time = define_frequency_resample(groupby)).median(dim = 'time')
        if spatial: return spatial_sampled_median
        else: return spatial_sampled_median.median(['lat', 'lon'])

    def plot_location(self, ax = None, alpha_gl: float = 0.5, **kwargs):
        """Plot the locations of the points of this spatial wind dataset on a map
        ax: optional, plot the location on the given ax
        alpha_gl: alpha of the gridlines
        """
        
        # Create the map fi no ax is given
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()
            ax.set_extent([min(self.data.lon), max(self.data.lon), min(self.data.lat), max(self.data.lat)], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            gls = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha = alpha_gl)
            gls.top_labels=False   # suppress top labels
            gls.right_labels=False # suppress right labels
            # Add a blue ocean
            ocean110 = cfeature.NaturalEarthFeature('physical', 'ocean', 
                scale='10m', edgecolor='none', facecolor=cfeature.COLORS['water'],zorder = 0)
            ax.add_feature(ocean110)

        # Construct the lon and lat as a 2d list of combinations instead of two lists from which each combination can be taken
        lons = np.repeat(self.data.lon, len(self.data.lat))
        lats = np.repeat([self.data.lat], len(self.data.lon), axis = 0).flatten()

        # Plot the data on the map
        ax.scatter(lons, lats, transform=ccrs.PlateCarree(), label = self.name, **kwargs)
        
        ax.legend()

        return ax

    def mask_land_sea(self, sea: bool = True, n_buffer: int = 0):
        """ Construct a buffered a land sea mask, by creating a border around the two defined regions of the mask
        the land sea mask by default contains the values 0 (land) and nan (sea), in the returned mask the buffer zone gets the value 1
        n_buffer: the number of neighbours used to buffer, by default no buffer is used, 
        """

        # Construct the mask for the grid of this WindDataSet
        region = regionmask.defined_regions.natural_earth_v5_0_0.land_10 #using the most detailed map available in the package
        mask = region.mask(self.data.lon, self.data.lat) # create a mask specific to the WindDataSet using this map

        # Get the min and max values for running over mask
        max_i = mask.sizes['lat']
        max_j = mask.sizes['lon']

        # First add buffer layers from the sea (half of the number of layers is taken from the sea, the other half from the land)
        # Loop over the full mask for each of the number of buffer layers divided by two (+ one for an odd number of layers)
        # For each buffer layer, alter the outer border, so the neighbours of the previous border (b, number of the previous buffer layer) 
        # that are sea (nan), to become b + 1 (number of the current buffer layer)
        # different layers of the buffer get a different number to easily select the outer layer (would also work by altering procedure after first layer)
        for b in range(n_buffer//2 + n_buffer%2):
            for i in range(max_i):
                for j in range(max_j):
                    if mask.data[i, j] == b:
                        if i > 0:
                            lower_neighbour = mask.data[i-1, j]
                            if np.isnan(lower_neighbour):
                                mask.data[i-1, j] = b+1
                        if i < max_i - 1:
                            upper_neighbour = mask.data[i+1, j]
                            if np.isnan(upper_neighbour):
                                mask.data[i+1, j] = b+1
                        if j > 0:
                            left_neighbour = mask.data[i, j-1]
                            if np.isnan(left_neighbour):
                                mask.data[i, j-1] = b+1
                        if j < max_j - 1:
                            right_neighbour = mask.data[i, j+1]
                            if np.isnan(right_neighbour):
                                mask.data[i, j+1] = b+1
        
        # Now do a similar procedure, adding buffer layers from the land
        # number of layers is just divided by two here, so if the number is odd, one more layer is taken from the sea than from the land
        for b in range(n_buffer//2):
            for i in range(max_i):
                for j in range(max_j):
                    if  b == 0: cond = 1
                    else: cond = b/3
                    if mask.data[i, j] == cond:
                        if i > 0:
                            lower_neighbour = mask.data[i-1, j]
                            if lower_neighbour == 0:
                                mask.data[i-1, j] = (1+b)/3
                        if i < max_i - 1:
                            upper_neighbour = mask.data[i+1, j]
                            if upper_neighbour == 0:
                                mask.data[i+1, j] = (1+b)/3
                        if j > 0:
                            left_neighbour = mask.data[i, j-1]
                            if left_neighbour == 0:
                                mask.data[i, j-1] = (1+b)/3
                        if j < max_j - 1:
                            right_neighbour = mask.data[i, j+1]
                            if right_neighbour == 0:
                                mask.data[i, j+1] = (1+b)/3

        # Now set all buffer layers to 1
        for i in range(max_i):
            for j in range(max_j):
                if mask.data[i, j] > 0 : # for nan this returns false
                    mask.data[i, j] = 1

        # Copy the current WindDataSet
        output = copy.copy(self)

        # Apply the mask, depending on whether we want to keep land or sea
        if sea:
            output.data = output.data.where(np.isnan(mask)) # the mask is Nan over sea
        else:
            output.data = output.data.where(mask == 0) # and 0 over land

        # Return the masked WindDataSet
        return output

    def get_closest_point(self, pointdataset: PointWindDataSet):
        """Get the closest point to the location of a given PointWindDatasSet from each ensemble member, not for the reference
        pointdataset: PointWindDataSet from which to take the coordinates to get the closest points
        """
        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Select the closest point
        output = output.sel(dict(lat = pointdataset.data.lat , lon = pointdataset.data.lon), method = 'nearest')

        return output

    def plot_map_mean_wspd(self, plot_range=None, cmap='viridis', ax = None):
        """plot a map of the mean wind speed in the domain over the whole timerange, infinite values are replaced by 9999 for plotting purposes
        (to be able to use a color scale that differentiates between nan and np.inf using set_bad and set_over functions on a colormap obtained by matplotlib.colormaps.get_cmap('viridis'))
        plot_range: specifies the range of the colorbar is possible for comparing to other maps
        cmap: the color map used in the plot, see https://matplotlib.org/stable/users/explain/colors/colormaps.html for all options
        ax: optional, to use as a background for another plot
        """

        # Create the map
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()

        #calculate mean wind speed over the full time period, ignoring nans (also the case by default)
        mean_wind_speed=self.data.wind_speed.mean(dim = 'time', skipna =True, dtype=np.float64)

        # replace infinite values by 9999
        mean_wind_speed = xr.where(~np.isinf(mean_wind_speed), mean_wind_speed, 9999)

        #if the range was not specified beforehand, just set the min and max as the range
        if plot_range is None:
            plot_range=[mean_wind_speed.min().data, mean_wind_speed.max().data]

        # Plot the data on the map
        mesh = ax.pcolormesh(self.data.lon, self.data.lat, mean_wind_speed, transform=ccrs.PlateCarree(), 
        cmap=cmap, vmin=plot_range[0], vmax=plot_range[1])
        ax.set_extent([min(self.data.lon), max(self.data.lon), min(self.data.lat), max(self.data.lat)], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        gls = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gls.top_labels=False   # suppress top labels
        gls.right_labels=False # suppress right labels

        # Add a colorbar
        plt.colorbar(mesh, ax=ax, label = 'Wind speed at 100 m a.g.l. (m/s)', pad=0.08, aspect=40, shrink=0.6)

        # Add a title
        plt.title(f'{self.name} mean wind speed')

        return ax

    def same_grid(self, winddataset: "SpatialWindDataSet", weak: bool = None):
        """ Check if this winddataset is on the same grid as the given one, returns true if this is the case and false if not
        winddataset: the winddataset to compare to
        weak: irrelevant, for uniformity with the same function for pointwinddatasets
        """

        # First check if the lengths of the lat and lon dimension are equal 
        # (needed since otherwise the comparison below does not work as intended, empty arrays)
        if len(self.data.lat) != len(winddataset.data.lat) or len(self.data.lon) != len(winddataset.data.lon):
            return False
        
        # Check if any of the latitude or longitude values are not equal
        if (self.data.lat != winddataset.data.lat).any() or (self.data.lon != winddataset.data.lon).any():  
            return False
        
        # If not the grids are equal
        return True
    
    def plot_nans(self, lat: int, lon: int):
        """ Plot a timeseries of the dataset, only showing where the nans are in the wind speed for a specific location
        lat: the index of the latitude to select
        lon: the index of the longitude to select
        """
        # Get the number of timesteps and the number of nans
        n_steps = len(self.data.time) # number of timesteps
        num_nans = np.sum(np.isnan(self.data.wind_speed.isel(lat =lat, lon = lon))).data # get number of nans

        # Create the figure
        fig, ax = plt.subplots(figsize = (15,3))
        plt.scatter(np.arange(n_steps), np.isnan(self.data.wind_speed.isel(lat =lat, lon = lon)), s =0.2)

        # Add plot decoration
        ax.set_title(f'{self.name} : {num_nans} NaNs in wind speed at {self.data.lat[lat].data}, {self.data.lon[lon].data}')
        ax.set_xlabel('Time')
        plt.xticks(np.arange(n_steps)[::round(n_steps/7 )], rotation = 45,
            labels=[str(step.dt.month.data)+'-'+str(step.dt.year.data) for step in self.data.time[0:n_steps][::round(n_steps/7)]])
        ax.set_ylabel('To NaN or not to NaN')
        ax.set_yticks([0,1],['Not NaN', 'NaN'])
        plt.show()

    def fill_nans(self):
        """ For this, the rioxarray package is needed which causes conflicts with the currently needed packages, so filling of nans in spatial data is handled in a separate notebook, rio_interpolation.ipynb"""
        pass

class Ensemble:
    """A collection of WindDataSets"""

    def __init__(self, name, path_list: list = [], point_path_list: list = [], name_list: list = None, ref: WindDataSet = None, include_future: bool = False, groups: list = None, include_run_name: bool = True, test_mode: bool = False, chunks: int = None, **kwargs):
        """ initialisation of an ensemble 
        name: name or short description of the ensemble
        path_list: list of paths to get spatial datasets from
        point_path_list: list of paths to get pointdatasets from
        name_list: optional, list of names for the datasets, otherwise the truncated paths are used as names
        ref: a WindDataSet to serve as reference when calculating statistics
        include_future: if True, the future simulations are inlcluded in the WindDataSets, if False they are not
        groups: a list of strings, dividing the ensemble in groups
        include_run_name: in case of automatically constructed names, should the run be included?
        test_mode: Boolean, if True the dataset is sliced (first 5000 timesteps) for quicker loading of data
        kwargs: applied to the pointwinddatasets
        """

        # Set the name of the ensemble
        self.name = name 

        # Construct the namelist if not given
        if name_list is not None:
            #check if namelist has the same length as the path list, if not raise a warning
            if len(name_list)!=len(path_list)+len(point_path_list):
                warnings.warn(f'Length of list of names ({len(name_list)}) and list of paths ({len(path_list)}) do not match.')
        else:
            # if no name_list is given, truncate the paths for the CMIP6 models
            name_list = [] # create empty list
            for path in path_list: 
                path = path.split('/')[-1] #select part after last occurence of '/'
                name = path.split('_')[0] 
                if include_run_name: # if needed add the run
                    name = name + ' ('+ path.split('_')[1]+')' # only keep model name and run, not experiment
                name_list.append(name) # add to the list of names
            # do the same thing for the point datasets
            for path in point_path_list:
                name_list.append(path.split('/')[-1]) #select part after last occurence of '/' as the name for the observational point datasets

        # Initialise empty list 
        data_list = []     

        # For each path in the list, initialise WindDataSet and add it to the ensemble 
        for i in range(len(path_list)):

            print(f'{round((i+1)/len(path_list)*100, 1)}% done') #print progress

            # Select the name from namelist
            name = name_list[i]

            # Select the path 
            path_hist = path_list[i]

            # If the future simulation does not have to be included, just create a WindDataSet containing the historical part
            if not include_future:
                data_list.append(SpatialWindDataSet(name, path=path_hist, test_mode=test_mode, chunks=chunks))
            # If the future simulation has to be included, load both and add them to each other
            else:
                hist = SpatialWindDataSet(name, path=path_hist, test_mode=test_mode, chunks = chunks) #initialise historical part
                path_fut = path_hist.replace('/hist/', '/fut/') #switch folders
                if 'hist-1950' in path_fut: path_fut = path_fut.replace('hist-1950', 'highres-future') #edit the path to get path for the future simulation
                elif 'highresSST-present' in path_fut: path_fut = path_fut.replace('present', 'future') 
                fut = SpatialWindDataSet(name, path=path_fut, test_mode=test_mode, chunks = chunks) #initialise future part
                data_list.append(hist+fut) #concatenate historical and future simulation

        # Add the PointWindDataSets, these are usually observations so no future is included ever
        for i in range(len(point_path_list)):
            
            # Select the name
            name = name_list[i + len(path_list)] # names of the point datasets were added after the names of the spatial datasets

            # Construct the PointWindDataSet and add it to the list
            data_list.append(PointWindDataSet(name = name, path = point_path_list[i], test_mode = test_mode, chunks = chunks, **kwargs))

        # Set the data list of WindDataSets that was just constructed  
        self.data_list = data_list
        
        # Set the reference dataset
        self.ref = ref #just set to None if not specified

        # Set the path list (used in projector to add future simulations
        self.path_list = path_list
        self.point_path_list = point_path_list

        # Set the groups if specified
        if groups is not None:
            #check if groups has the same length as the path list, if not raise a warning
            if len(groups)!=len(path_list):
                warnings.warn(f'Length of list of groups ({len(groups)}) and list of paths ({len(path_list)}) do not match.')

            self.groups = groups #set the groups, this should be a list of the same length as the name_list
            #print a little reminder
            print('Watch out! The groups setting assumes that members of the same GCM but different group are listed together, so make sure this is the case.')
        else:
            self.groups = None

    def __repr__(self):
        """ Representation of the Ensemble for printing
        """
        return f'Name: {self.name} \n {self.data_list}'

    def _sanity_check_calendars(self):
        """ perform sanity check of the calendars for each member and the reference if given
        """

        for member in self.data_list:
            print(member.name)
            member._sanity_check_calendars()
        if self.ref is not None:
            print(f'Reference: {self.ref.name}')
            self.ref._sanity_check_calendars()

    def _sanity_check_leap_year(self):
        """ perform sanity check of the leap years for each member and the reference if given 
        """

        for member in self.data_list:
            print(member.name)
            member._sanity_check_leap_year()
        if self.ref is not None:
            print(f'Reference: {self.ref.name}')
            self.ref._sanity_check_leap_year()

    def one_grid(self, grid: WindDataSet, verbose: bool = True):
        """convert all spatial WindDataSets in the ensemble (including the reference) to the same grid, PointWindDataSets are just left at their points
        grid: the WindDataSet to take the grid from
        verbose: should progress be printed?
        """

        # Copy the ensemble to not alter the original ensemble
        output = copy.deepcopy(self) #deepcopy is needed since this is essentialy a nested list: a list of WindDataSets which are lists of data
        
        # Regrid each ensemble member if it is a spatial winddataset
        for i in range(len(output.data_list)):
            dataset = output.data_list[i]
            if isinstance(dataset, SpatialWindDataSet):
                output.data_list[i] = dataset.interpolate_horizontally(grid)
                if verbose: print(f'{round((i+1)/len(output.data_list)*100, 1)}% done') #print progress
            else: 
                print(f'{dataset.name} is a single point dataset, so not interpolated')

        # Regrid the reference if there is one and it is a spatial winddataset
        if self.ref is not None:
            if isinstance(self.ref, SpatialWindDataSet):
                if verbose: print('Regridding reference')
                output.ref =  output.ref.interpolate_horizontally(grid)
            else:
                print(f'{self.ref.name} is a single point dataset, so not interpolated')

        # Return the regridded ensemble
        return output

    def check_one_grid(self):
        """ Checks if this ensemble is regridded to a common grid, returns true if yes, false if not
        weak comparison for pointwinddatasets
        """
        
        # Print warning weak comparison
        print('Warning: for PointWindDataSets the coordinates matches are not checked')

        # Initialise the index and whether ensemble still seems to have a common grid
        one_grid = True
        i = 0

        # Loop over all ensemble members, check if grids are the same as for the next member
        while one_grid and i < len(self.data_list)-1:
            if not self.data_list[i].same_grid(self.data_list[i+1], weak = True):  # check if member i and i+1 have the same grid
                one_grid = False # if not, this is not one grid anymore
            i+=1 # increase the index to move to the next member

        # Check if the reference, if there is one, is on the same grid as the first ensemble member
        if one_grid and self.ref is not None:
            if not self.data_list[0].same_grid(self.ref, weak = True):
                one_grid = False # if not, this is not on one grid anymore

        return one_grid

    def sel(self, indexers, method: str = None):
        """ selection from https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html
        returns a new Ensemble where the selection was applied to all WindDataSets (including the ref)
        indexers: dimensions along which to index, e.g.  dict(lat = slice(54,59), lon = slice(0.2, 5)), 
            dict(lat= [50,51], lon = 3) specific values needed for this or method set to something else then None so inexact matches are included
            for time: dict(time = slice(cftime.DatetimeGregorian(1970, 1, 1, 12, 0, 0, 0, has_year_zero=False),cftime.DatetimeGregorian(1971, 1, 1, 12, 0, 0, 0, has_year_zero=False)))
            does not really work for selecting specific months or so on the full ensemble
        method: selection method to use  for inexact matches: None, "nearest", "pad", "ffill", "backfill", "bfill", default "None" means only exact matches
        """
        
        # Copy the Ensemble
        output = copy.deepcopy(self)

        # Apply the selection for each ensemble member
        for i in range(len(output.data_list)):
            output.data_list[i] = self.data_list[i].sel(indexers, method)
            # Alternative (but bad practice)
            # Use try block to skip over the selection if the dimension is not present in a certain dataset, e.g. for observations no lat/lon can be selected
            #try:
                #output.data_list[i] = self.data_list[i].sel(indexers, method)
            #except KeyError as e:
                #print(e)
                #print(f'Dataset {output.data_list[i].name} in the ensemble does not have the dimension that is being selected.')

        # Apply the selection to the reference if there is one
        if self.ref is not None:
            output.ref = self.ref.sel(indexers, method)

        return output
    
    def align(self, winddataset: WindDataSet, dropna: bool = False):
        """Aligns the dimensions of all ensemble members and the reference to the given winddataset
        winddataset: the winddataset to align to"""

        # Copy the Ensemble
        output = copy.deepcopy(self)

        # Align the ensemble members
        for i in range(len(self.data_list)):
            output.data_list[i] = self.data_list[i].align(winddataset, dropna)

        # Align the reference
        if self.ref is not None:
            output.ref = self.ref.align(winddataset, dropna)

        return output
    
    def get_closest_point(self, pointdataset: PointWindDataSet):
        """Get the closest point to the location of a given PointWindDatasSet from each ensemble member, not for the reference
        pointdataset: PointWindDataSet from which to take the coordinates to get the closest points
        """
        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Select the closest point
        output = output.sel(dict(lat = pointdataset.data.lat , lon = pointdataset.data.lon), method = 'nearest')

        return output

    def grouped_mean(self, groupby: str, spatial: bool = False):
        """Calculate the mean values over the full ensemble grouped by the specified string, also taking the mean over the domain
        groupby: str, entity to group by
        spatial: if True, spatial dimensions are kept, otherwise the mean is also taken
        """

        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Apply the grouped mean to the ensemble members, depending on type we need spatial parameter or not
        for i in range(len(self.data_list)):
            if type(output.data_list[i]) == SpatialWindDataSet:
                output.data_list[i].data = self.data_list[i].grouped_mean(groupby = groupby, spatial = spatial)
            else:
                output.data_list[i].data = self.data_list[i].grouped_mean(groupby = groupby)

        # And the same for the reference
        if self.ref is not None:
            if type(output.ref) == SpatialWindDataSet:
                output.ref.data = self.ref.grouped_mean(groupby = groupby, spatial = spatial)
            else: 
                output.ref.data = self.ref.grouped_mean(groupby = groupby)

        return output

    def grouped_median(self, groupby: str, spatial: bool = False):
        """Calculate the median values over the full ensemble grouped by the specified string, also taking the mean over the domain
        groupby: str, entity to group by
        spatial: if True, spatial dimensions are kept, otherwise the mean is also taken
        """

        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Apply the grouped median to the ensemble members
        for i in range(len(self.data_list)):
            if type(output.data_list[i]) == SpatialWindDataSet:
                output.data_list[i].data = self.data_list[i].grouped_median(groupby = groupby, spatial = spatial)
            else:
                output.data_list[i].data = self.data_list[i].grouped_median(groupby = groupby)
                
        # And the same the reference
        if self.ref is not None:
            if type(output.ref) == SpatialWindDataSet:
                output.ref.data = self.ref.grouped_median(groupby = groupby, spatial = spatial)
            else:
                output.ref.data = self.ref.grouped_median(groupby = groupby)
        return output

    def sampled_mean(self, groupby: str, spatial: bool = False):
        """Calculate the mean values over the full ensemble sampled by the specified string, also taking the mean over the domain if needed
        groupby: str, entity to sample by
        spatial: if True, spatial dimensions are kept, otherwise the mean is also taken
        """

        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Apply the grouped mean to the ensemble members, depending on type we need spatial parameter or not
        for i in range(len(self.data_list)):
            if type(output.data_list[i]) == SpatialWindDataSet:
                output.data_list[i].data = self.data_list[i].sampled_mean(groupby = groupby, spatial = spatial)
            else:
                output.data_list[i].data = self.data_list[i].sampled_mean(groupby = groupby)

        # And the same for the reference
        if self.ref is not None:
            if type(output.ref) == SpatialWindDataSet:
                output.ref.data = self.ref.sampled_mean(groupby = groupby, spatial = spatial)
            else: 
                output.ref.data = self.ref.sampled_mean(groupby = groupby)

        return output

    def sampled_median(self, groupby: str, spatial: bool = False):
        """Calculate the median values over the full ensemble sampled by the specified string, also taking the median over the domain if needed
        groupby: str, entity to sample by
        spatial: if True, spatial dimensions are kept, otherwise the median is also taken
        """

        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Apply the grouped median to the ensemble members
        for i in range(len(self.data_list)):
            if type(output.data_list[i]) == SpatialWindDataSet:
                output.data_list[i].data = self.data_list[i].sampled_median(groupby = groupby, spatial = spatial)
            else:
                output.data_list[i].data = self.data_list[i].sampled_median(groupby = groupby)
                
        # And the same the reference
        if self.ref is not None:
            if type(output.ref) == SpatialWindDataSet:
                output.ref.data = self.ref.sampled_median(groupby = groupby, spatial = spatial)
            else:
                output.ref.data = self.ref.sampled_median(groupby = groupby)
        return output

    def select_members(self, selection: list[bool]):
        """Select members from the ensemble, returning a copy of the ensemble with selected members
        selection: list of bools, indicating which members should be selected
        """

        # Copy the original ensemble
        output = copy.deepcopy(self)

        # Check if selection list is the same length as the number of ensembles, otherwise unexpected things can happen
        if len(selection) != len(output.data_list):
            raise ValueError(f'Selection should be the same length as the number of ensembles ({len(output.data_list)}), but is {len(selection)}.')

        # Select the members
        output.data_list = [member for member, sel in zip(output.data_list, selection) if sel]

        return output

    def ensemble_mean(self, select_datasets: list[bool] = None):
        """Returns a WindDataSet representing the ensemble mean, already supposes that the ensemble has a common grid
        select_datasets: list of booleans of the same length as the number of datasets in the ensemble, to decide which dataset to include
        """

        # Check if the ensemble has a common grid, raise an error if not
        if not self.check_one_grid():
            raise ValueError(f'Ensemble does not have a common grid, which is needed to compute the ensemble mean.')

        # Create an empty dataset to save the mean
        ensemble_mean = copy.deepcopy(self.data_list[0]) # lil hack to get something in the right format
        ensemble_mean.data['wind_speed'] = ensemble_mean.data.wind_speed - ensemble_mean.data.wind_speed
        ensemble_mean.data['wind_direction'] = ensemble_mean.data.wind_direction - ensemble_mean.data.wind_direction
        ensemble_mean.data = ensemble_mean.data.fillna(0) # replace nans with 0's to avoid problems in sum

        # Make the timestamps "uniform" to make sure they match for all members of the ensemble
        ensemble_mean.uniform_timestamps()

        # Use just the data not the full winddataset
        ensemble_mean = ensemble_mean.data

        # Set select_datasets to a list of only true if not given
        if select_datasets is None:
            select_datasets = np.repeat('True', len(self.data_list))
        elif all(~np.array(select_datasets)):
            raise ValueError('Not a single dataset is selected, impossible to compute the ensemble mean.')

        # Add each ensemble member to the mean
        for i, dataset in enumerate(self.data_list):
            if select_datasets[i]:
                # make the timestamps in the datasets uniform and follow a Gregorian calendar
                dataset.uniform_timestamps()
                # when this function does not work (two datasets can't be added to each other), it's probably because of a timestamp mismatch
                # the uniform_timestamps function solves the mismatches encountered up to now, but for new models
                # new exceptions might have to be made
                
                # Add this dataset to the ensemble mean
                ensemble_mean = ensemble_mean + dataset.data.fillna(0) # replace nans with 0's, so that nans don't influence the sum (x + nan = nan)

        # Number of non nans to compute the average based on the non nan values (otherise e.g. 5 values, 2 nans are ignored but still divided by 5)
        total_non_nans= 0
        for non_nans in [~np.isnan(d.data) for d, i in zip(self.data_list, select_datasets) if i == True]:
            total_non_nans = total_non_nans + non_nans

        # Normalise the mean with the number of datasets where there is no nan
        # if there are 0 non nans, this becomes normalisation by 0 and therefore results in nan, which is ideal if all values from the ensemble members are nan
        ensemble_mean = ensemble_mean/total_non_nans

        # Add common attributes between all ensemble members
        common_keys = reduce(operator.and_, (set(d) for d in [d.data.attrs for d in self.data_list]))
        common_elements = {k: self.data_list[0].data.attrs[k] for k in common_keys if all(self.data_list[0].data.attrs[k] == d.get(k) for d in [d.data.attrs for d in self.data_list])}
        ensemble_mean.attrs = common_elements
        
        # Construct a WindDataSet from this, depending on shape of dataset
        if isinstance(ensemble_mean.lat, (np.floating, float)):
            return PointWindDataSet(name = f'{self.name} ensemble mean {select_datasets}', data = ensemble_mean, remove_nan = False)
        else: return SpatialWindDataSet(name = f'{self.name} ensemble mean {select_datasets}', data = ensemble_mean)

    def mask_land_sea(self, sea: bool = True, n_buffer: int = 0):
        """ mask the land/sea from all WindDataSets in the Ensemble using a general, not model-specific mask. Returns a new Ensemble
        sea: should the sea or the land be masked? If True, the sea remains, if False, the land remains
        n_buffer: the number of neighbours used to buffer, by default no buffer. The land sea mask by default
        """

        # Copy the ensemble to avoid altering the original data
        output = copy.deepcopy(self) #deepcopy is needed since this is essentially a nested list: a list of WindDataSets which are lists of data
        
        # Mask all ensemble members:
        for i in range(len(output.data_list)):
            if isinstance(output.data_list[i], SpatialWindDataSet):
                output.data_list[i] = output.data_list[i].mask_land_sea(sea = sea, n_buffer = n_buffer)
            else:
                print(f'{self.data_list[i].name} is a single point dataset, so no mask is applied')

        # Mask the reference if there is one
        if output.ref is not None:
            if isinstance(output.ref, SpatialWindDataSet):
                output.ref = output.ref.mask_land_sea(sea = sea)
            else:
                print(f'{self.ref.name} is a single point dataset, so no mask is applied')
            
        # Edit name by adding land/sea to it
        if sea: name = 'sea'
        else: name = 'land'
        output.name =f'{output.name} {name}'
        
        return output

    def select_season(self, season: str):
        """Select the specified season, and return a new Ensemble containing only data from that season
        season: String winter, spring, summer, fall/autumn
        """

        output = copy.deepcopy(self) #copy the ensemble to not alter the original data, deepcopy is needed since this is essentialy
        # a nested list: a list of WindDataSets which are lists of data
        # select season from all ensemble members:
        for i in range(len(output.data_list)):
            output.data_list[i] = output.data_list[i].select_season(season)
        # and select the season from the reference if there is one
        if output.ref is not None:
            output.ref = output.ref.select_season(season)
        #add the season to the name
        output.name = f'{output.name} {season}'

        return output

    def plot_map_mean_wspd(self, plot_range: list = None, cmap: str = 'viridis'):
        """ plot maps of the wind speed for all datasets with the same range
        plot_range: the range used for the color scales
        cmap: the color map to use
        """
        
        # if no range is given, calculate the range by traversing all datasets in the ensemble
        #not very optimal: np.mean over time is computed twice for each dataset: when plotting and when calculating range
        # maybe adjust later
        if plot_range is None:
            mini = np.infty
            maxi = -np.infty
            for dataset in self.data_list:
                #calculate the quantity that will be plotted
                mean_wind_speed = dataset.data.wind_speed.mean(dim = 'time', skipna = True, dtype=np.float64)
                #adjust min and max if needed: if they are greater/smaller than previous min/max
                if mean_wind_speed.min().data < mini:
                    mini= mean_wind_speed.min().data
                if mean_wind_speed.max().data > maxi:
                    maxi= mean_wind_speed.max().data

            plot_range=[mini, maxi] #set range

        
        #plotting all after each other not in the same grid
        for dataset in self.data_list:
            # only for SpatialWindDataSets the map can be plotted
            if isinstance(dataset, SpatialWindDataSet):
                dataset.plot_map_mean_wspd(plot_range=plot_range, cmap = cmap)

        #also plot the reference if there is one
        if self.ref is not None:
            self.ref.plot_map_mean_wspd(plot_range = plot_range)

    def plot_wind_rose(self):
        """ plots basic wind roses for all datasets at once 
        """
        #possibility: plot these in some sort of raster, with same axes for each wind rose

        #run over list of datasets and plot the wind rose
        for dataset in self.data_list:
            dataset.plot_wind_rose()

        #also plot the reference if there is one
        if self.ref is not None:
            self.ref.plot_wind_rose()

    def plot_histogram_wspd(self, nbins: int):
        """ plot a histogram of the windspeeds for all ensemble members 
        nbins =  the number of bins to use or a strategy such as 'auto' (these take longer)
                see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        """
        
        #flatten the wind speed data to one-dimensional arrays (this is needed to apply the hist function)
        flattened_data = [dataset.data.wind_speed.data.flatten() for dataset in self.data_list] #do this for all ensemble members and put in a list
        labels = [dataset.name for dataset in self.data_list] #construct the labels for the plot by listing all dataset names
        
        #if there is a reference dataset, add it to the flattened data and labels
        if self.ref is not None:
            flattened_data.insert(0, self.ref.data.wind_speed.data.flatten()) #add reference on the first place of the list
            labels.insert(0, self.ref.name)

        #actual plotting
        plt.hist(flattened_data, bins=nbins, density=True, label=labels)
        plt.title(f'Wind speed distribution')  
        plt.ylabel('Frequency')
        plt.xlabel('Wind speed (m/s)')
        plt.legend()
        plt.show()

    def plot_density_wspd(self, nbins: str, ax = None, ylim: float = None, detailed: bool = True, cmap: str = 'viridis', alphas: list = [1,1], linewidths: list = [2,3], linestyles = ['solid', 'solid'], plot_mean: bool = False):
        """ plot estimated probability density function of the windspeeds for all ensemble members 
        nbins =  the number of bins to use or a strategy such as 'auto' (these take longer)
                see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        ax: optional, axis object to plot on
        ylim: optional, limit of the y axis
        detailed: option to plot lines without legend, all in one color and a bit fainter if False
        alphas, linewidths, linestyles: options for the undetailed lines and reference line
        plot_mean: should annotation with mean be added
        """

        # Use the globally defined colors
        global colors

        # Create figure if not given
        if ax is None:
            fig, ax = plt.subplots()

        # Loop over the ensemble, estimating the probability density function for each member and the reference
        for i in range(len(self.data_list)):
            dataset = self.data_list[i]
              
            # Prepare colormap for many similar lines (so different from globally defined colors)
            cmap = plt.get_cmap(cmap) 
            values = np.linspace(0, 1, len(self.data_list))
            colors_cmap = [cmap(value) for value in values]

            # Set the plotting options depending on how much detail we want to see
            if not detailed and dataset.name not in ['ERA5', 'CERRA']: 
                color = colors_cmap[i] #np.linspace(0,0.7, len(self.data_list)+1, endpoint = False)[i+1]
                if i == len(self.data_list)-1: label = self.name # only put a label for the last one
                else: label = '_nolegend_'
            else: 
                color = colors[i%len(colors)]
                label = dataset.name

            # Plot actual density with desired parameters        
            dataset.plot_density(nbins = nbins, label = label, color = color, linestyle = linestyles[0], linewidth = linewidths[0], alpha = alphas[0], plot_mean = plot_mean, ax = ax, ylim = ylim)   
        
        # Plot the reference
        if self.ref is not None:
            self.ref.plot_density(nbins = nbins, ax = ax, color = 'black', label = self.ref.name, linestyle = linestyles[1], linewidth = linewidths[1], alpha = alphas[1], ylim = ylim)

        # Set full plot title     
        ax.set_title(f'{self.name}')  

        plt.tight_layout()

    def plot_time_series(self, groupby: str = 'year', text_size: float = 8, annotate: bool = False, ax = None):
        """ plot yearly/... time series for all members of the ensemble and the reference
        groupby:  which unit of time to group by, year or month 
        text_size: set the size of the text elements in the plot
        annotate: should the name of the dataset be added to the lines?
        ax: optional, axis object to plot on
        """
        
        # Create the figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Set the font size 
        plt.rc('font', size = text_size)

        global colors # use the globally defined colors

        # Plot the reference if there is one
        if self.ref is not None:
            self.ref.plot_time_series_line(ax, groupby, annotate = annotate, linestyle='-',
                    color='black', label=f'{self.ref.name}')

        # Plot the ensemble members
        for i in range(len(self.data_list)):
            self.data_list[i].plot_time_series_line(ax, groupby, annotate = annotate, linestyle='-', 
                color=colors[i%len(colors)], label=f'{self.data_list[i].name}')

        # Add plot elements depending on which unit of time to group by
        if groupby == "year":
            ax.set_title(f'Yearly mean wind speed')
            ax.set_xlabel('Year')

        elif groupby == "month":
            ax.set_title(f'Monthly mean wind speed')
            ax.set_xlabel('Month')
            ax.set_xticks(labels=['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ticks=np.arange(1, 13))

        # Add plot generalities
        ax.set_ylabel('Wind speed (m/s)')
        ax.grid(axis = 'y') # only horizontal gridlines
        #plt.gca().set_ylim(bottom=0)
        ax.spines[['right', 'top']].set_visible(False) # remove right and top axis
        
        # Add legend if no annotations are added
        if not annotate:
            ax.legend(frameon = False)

        plt.tight_layout()
        #plt.show()

    def plot_time_series_grouped_high_low(self, groupby: str = 'year', ylim: float = None, ax = None):
        """ plot yearly time series for all members of the ensemble and the reference, using the same linestyle for members in the same group and only having groups in the legend
        specifically for plotting high/low res division: one model is one color, linestyle is high/low, does not work well with other types of group configurations
        groupby:  which unit of time to group by, year, month, or hour 
        ax: optional, axis object to plot on
        """
        # could be integrated in function above to avoid code duplication

        # Create the figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Construct the colors
        unique_groups, idx =np.unique(self.groups, return_index = True) #get the different groups, also return original indexes
        unique_groups = unique_groups[np.argsort(idx)] # and sort according to these, because np.unique does not preserve the ordering, which is important when labelling
        n_groups = len(unique_groups) #get the number of different groups
        global colors # use the globally defined colors
        colors_models = np.repeat(colors, n_groups) #repeat the colors for each group (this assumes input is sorted: similar members of different groups are together)
        #so for low/high res: low and high res version of different GCMs are together

        linestyles_options = ['solid', 'dotted', 'dashed'] #options for the linestyles
        linestyles = self.groups #copy the groups

        # now construct linestyle for all members to be plotted by using same linestyle for same group
        #(don't name the groupnames the same as the linestyle options above, this won't work)
        # loop over all groups
        for i in range(n_groups):
            group = unique_groups[i] #get the group name
            linestyles = [linestyles_options[i] if string == group else string for string in linestyles] #replace all groupnames in the list by the ith linestyle option

        # Plot the reference
        if self.ref is not None:
                self.ref.plot_time_series_line(ax, groupby, linestyle='-',
                     color='black', label=f'{self.ref.name}', annotate = True)
        # Plot the ensemble members
        for i in range(len(self.data_list)):
            color = colors_models[i%len(colors_models)] #set color
            self.data_list[i].plot_time_series_line(ax, groupby, linestyle=linestyles[i], 
                color=color, label=f'{self.data_list[i].name}', annotate = True)

        # Add plot elements depending on which unit of time to group by
        if groupby == "year":
            ax.set_title(f'Mean wind speed {self.name}')
            ax.set_xlabel('Year')
        elif groupby == "month":
            ax.set_title(f'Monthly mean wind speed')
            ax.set_xlabel('Month')
            ax.set_xticks(labels=['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ticks=np.arange(1, 13))

        # Add general plot elements
        ax.set_ylabel('Wind speed (m/s)')
        ax.grid(True)

        # Specify the ylim if needed
        if ylim is not None:
            plt.gca().set_ylim(bottom=0)
            plt.gca().set_ylim(top=ylim)

        # Add the name of the reference if there is one
        if self.ref is not None:
            unique_groups = np.insert(unique_groups, 0, self.ref.name) 
        
        # Add the legend
        ax.legend(unique_groups) #put the group names (and ref) as labels

        #plt.show()

    def plot_time_series_ensemble_mean(self, groupby: str = 'year', annotate: bool = False, ylim: float = None, save: bool = False, text_size: float = 8, ax = None):
        """ plot yearly time series for the ensemble mean and the reference, with individual models as lighter lines
        groupby:  which unit of time to group by, year, month, or hour 
        annotate: should text annotation be added to the lines
        ylim: upper limit on the y axis, optional
        legend: should the legend be plotted? if not, labels are added as annotations on the plot
        save: should the plot be saved?
        text_size: optional, size of the text elements in the plot
        ax: optional, axis object to plot on
        """
        print("WARNING: THIS DOES NOT WORK WITH DATASETS WITH NANS IN THEM, TO BE RESOLVED") 
        # Create the figure
        if ax is None:
            fig, ax =  plt.subplots(figsize=(10, 6))

        # Set the font size 
        plt.rc('font', size = text_size)

        # Get the unique groups
        unique_groups, idx = np.unique(self.groups, return_index = True) #get the different groups, also return original indexes
        unique_groups = unique_groups[np.argsort(idx)] # and sort according to these, because np.unique does not preserve the ordering, which is important when labelling
        n_groups = len(unique_groups) #get the number of different groups
        
        # Initialise the colors
        global colors #use the globally defined colors
        colors_models = self.groups #copy the groups to initialise the colors
        #now construct color for all members to be plotted by using same color for the same group
        #(don't name the groupnames the same as the color options above, this won't work)
        # loop over all groups
        for i in range(n_groups):
            group = unique_groups[i] #get the group name
            colors_models = [colors[i] if string == group else string for string in colors_models] #replace all groupnames in the list by the ith linestyle option

        # Initialise ensemble mean (to save the mean for each group)
        ensemble_mean = np.zeros(n_groups).tolist()

        # Plot the reference if there is one
        if self.ref is not None:
            self.ref.plot_time_series_line(ax, groupby, annotate = annotate, linestyle='-', linewidth = 3,
                    color='black', label=f'{self.ref.name}')

        # Plot the ensemble members
        for i in range(len(self.data_list)):
            ax, mean_member = self.data_list[i].plot_time_series_line(ax, groupby, annotate = False, return_values = True, linestyle='-', 
                alpha =0.2, color=colors_models[i], label='_nolegend_') # makes sure these lines don't appear in the legend
            index_gr = np.where(unique_groups == self.groups[i])[0][0] #get the index of the group name in unique_groups
            ensemble_mean[index_gr] += mean_member.fillna(0) # increase the ensemble mean of the correct group

        # Plot the ensemble means for each group
        for i in range(len(ensemble_mean)):
            group_mean = ensemble_mean[i] # get the ensemble mean for this group (not scaled yet) from the ensemble means list
            
            # Replace 0's with nan, these are the grid points where all members are zero (sum 0), which means that they were all nan
            group_mean = group_mean.where(group_mean != 0, np.nan)

            # Number of nans for the normalisation of the mean
            total_nans= 0
            for nans in [~np.isnan(d.data.wind_speed) for d in self.data_list]:
                total_nans = total_nans + nans
                # todo this should be summed over lon and lat

            # Scale the group mean
            print(group_mean)
            print(total_nans)
            group_mean = group_mean.wind_speed/total_nans
            group_size = np.sum(np.array(self.groups) == unique_groups[i]) # get the size of this particular group
            #print(total_nans)
            # depending on the groupby values, the variables have different names 
            # maybe make an ensemble mean function for the grouped ensemble means? then this could use the same plot_time_series_line function
            if groupby =='year':
                plt.plot(group_mean.year, group_mean.data, color = colors[i], linewidth = 3, label = unique_groups[i]) # plot the group mean, scaled by the size of the group
                if annotate:
                    plt.text(group_mean.year[-1]+0.3, (group_mean.wind_speed)[-1]-0.25,unique_groups[i], 
                        color = colors[i], weight = 'bold') # add the label in the plot
            elif groupby == 'month':
                plt.plot(group_mean.month, group_mean.data, color = colors[i], linewidth = 3, label = unique_groups[i]) # plot the group mean, scaled by the size of the group
                if annotate:
                    plt.text(group_mean.month[-1]+0.3, (group_mean.wind_speed)[-1],
                        unique_groups[i], color = colors[i], weight = 'bold')
        
        # Add plot elements depending on which unit of time to group by 
        match groupby:
            case "year":
                ax.set_title(f'Mean wind speed {self.name}')
                ax.set_xlabel('Year')
            case "month":
                ax.set_title(f'Monthly mean wind speed')
                ax.set_xlabel('Month')
                ax.set_xticks(labels=['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ticks=np.arange(1, 13))
                
        # Add plot generalities
        ax.set_ylabel('Wind speed (m/s)')
        ax.grid(axis = 'y')
        ax.spines[['right', 'top']].set_visible(False) # remove right and top axis

        # Specify the ylim if needed
        if ylim is not None:
            plt.gca().set_ylim(bottom=0)
            plt.gca().set_ylim(top=ylim)

        # Construct the legend if no annotations are added
        if not annotate:
            #for line in leg.get_lines(): #set the alpha to one for each line in the legend
            #    line.set_alpha(1)
            ax.legend(frameon = False)

        # Save the plot if needed
        if save:
            plt.savefig(f"{self.name}_{groupby}.pdf")

        # Show the plot
        #plt.show() 

class Comparator:
    """ Abstract parent class to unite methods used by both Validators and Projectors"""

    def _construct_label(self, wind_speed: bool, metric: str, period = None, relative = False):
        """Construct a label for the colorbar on a graph/map showing error metrics
        wind_speed: if True, the wind speed is used, if False the wind direction is used
        metric: the name of the metric
        period: optional, the period for variability metrics
        """
        # Construct the unit depending on the metric and the wind speed:
        if metric.lower() == 'relative bias': unit = ' (%)'
        elif metric.lower() == 'rcov': unit = ''
        elif wind_speed: unit =' (m/s)'
        else: unit = ' (°)'

        # Construct a part indicating if this is bias or not (for the variability metrics, this is not about relative bias)
        if relative: rel = ' bias '
        else: rel = ''

        # Construct the label as the metric plus the unit (and the period if the period was specified)
        label = metric + rel + unit

        # If the period was specified, add it to the label
        if period is not None:
            match period.lower(): #check which case the lowercase version of the period matches
                case 'season':
                    label = period + 'al ' + metric + rel + unit 
                case 'year'|'month':
                    label = period + 'ly ' + metric + rel + unit
                case 'none':
                    label =  metric + rel + unit

        # And return it
        return label

    def plot_map(self, data, plot_range: list = None, colorbar_label: str = None, title: str = None, cmap: str = 'viridis', hatch = None, text_size: float = 8, ax = None, 
        remove_cbar: bool = False, hatch_pattern: str = 'X', labels: list = ['left', 'bottom'], plot_mean: bool = False, offsets: list = [-2.2, -2]):
        """ create a map of the given error metric or variability data (similar to plotting function for winddataset)
        data: an xarray containing lat, lon and some sort of spatial data
        plot_range: the range to set for the colorbar on the plots
        colorbar_label: optional, the label to give the colorbar
        title: optional, the title to put on the plot
        hatch: array of the same shape as data, determining if any hatching over the map is needed
        hatch_pattern: the hatching pattern, options see https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
        text_size: size of the text on the plot
        ax: optional, ax to plot to
        remove_cbar: should the colorbar be plotted
        labels: which axis ticks should be plotted, False to have no labels
        plot_mean: optional, should a the mean of the data be added as an annotation on the map
        offsets: optional, offsets for the annotation [x,y], compared to max lat and max lon
        """

        # Create the map
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()

        # Set the font size 
        plt.rc('font', size = text_size)

        # If the range was not specified beforehand, just set the min and max as the range
        if plot_range is None:
            plot_range=[np.min(data), np.max(data)]

            # Symmetrize the range if the lower value is negative, and put the color scale to a divergent scale
            if plot_range[0] < 0:
                ext = np.max(np.abs(plot_range))
                plot_range = [-ext, ext]
                if cmap is None:
                    cmap = 'seismic'

        # Plot the data on the map (by default wind speed, not wind direction)
        mesh = ax.pcolormesh(data.lon, data.lat, data.data, transform=ccrs.PlateCarree(), 
            cmap=cmap, vmin=plot_range[0], vmax=plot_range[1])
        ax.set_extent([min(data.lon), max(data.lon), min(data.lat), max(data.lat)], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        gl = ax.gridlines(draw_labels=labels, dms=True, x_inline=False, y_inline=False, color = 'None')
        gl.xlines = False
        gl.ylines = False
        
        # Add the hatching if needed
        if hatch is not None:
            ax.pcolor(data.lon, data.lat, hatch, alpha = 0, transform=ccrs.PlateCarree(), hatch = hatch_pattern)    

        # Add a colorbar and colorbar label
        cbar = plt.colorbar(mesh, ax=ax, pad=0.08, aspect=40, shrink=0.6)
        cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")
        if remove_cbar: cbar.remove()

        # Add annotation mean if needed
        if plot_mean:
            ax.text(x = max(data.lon) + offsets[0], y = max(data.lat) + offsets[1], s = f"Mean:\n {data.mean(dtype=np.float64).data:.2f}", backgroundcolor = 'white')

        # Add a title
        ax.set_title(title)

        return mesh
    
    def plot_table(self, values: list, colnames: list, rownames: list, cmap: str = 'viridis', title: str = None, colorbar_label: str = None, textcolors=("black", "white"), 
        threshold: float =None, plot_range: list = None, filename: str = None, grid_color:str = 'w', ax =  None, cbar_kwargs: dict = None, **kwargs):
        """ Plotting the given errors in a colored table with one common colorbar, strongly based on #https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        errors: list of values to plot, elements of the list are merged and then transposed
            so if values is [a,b] with a= [1,2] and b = [3,4], then the table will be:
            1   3
            2   4
        colnames: list of strings, to label the columns
        rownames: list of strings, to label the rows
        title: optional, title of the plot
        colorbar_label: optional label for the colorbar
        textcolors: opitonal, colors of the text in the squares
        threshold: optional, value to switch between textcolors (for readability)
        plot_range: optional, range of the colors
        filename: name of the file when saving the plot, if None the plot is not saved
        grid_color: the color of the grid
        ax: optional, ax to add the plot to
        cbar_kwargs: arguments to hand to the colorbar, e.g. dict(pad=0.08, aspect=40, shrink=0.6)
        """

        # Set the plot range if not given to min and max of the data
        if plot_range is None:
            min_data = np.min(values)
            max_data = np.max(values)
            plot_range = [min_data, max_data]

        # If the minimum is negative or the color map is seismic (divergent), symmetrise the plot range and choose a diverging color map
        if (plot_range[0] < 0) | (cmap == 'seismic'):
            abs_max = np.max(np.abs(plot_range))
            plot_range = [-abs_max,abs_max]
            #change the colormap to a diverging one while we're at it
            if cmap == 'viridis': #default value of cmap
                cmap = 'seismic'
                # coolwarm: some sort of gray as 0, so not very clear, alternatives with white as 0: bwr, seismic, RdBu 
                #and alter the textcolors to suit this colormap
                textcolors =('black', 'white')
                
        # Construct the data to put in the table
        data_merged = np.array(values) #in a numpy array
        data_merged = data_merged.transpose() # transpose this array
        if len(data_merged.shape) == 1: data_merged = np.expand_dims(data_merged, axis = 1) # to make this work for one column as well

        # Create the heatmap
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(data_merged, cmap = cmap, vmin=plot_range[0], vmax=plot_range[1])

        # Show all ticks and label them with the respective list entries
        ax.set_yticks(range(len(rownames)), labels=rownames,
                    ha="right")
        ax.xaxis.tick_top() # put the x axis ticks on top
        ax.set_xticks(range(len(colnames)), labels=colnames, rotation=-30, ha="right", rotation_mode="anchor") #rotate the xaxis ticks (for readability in large tables)
        plt.tick_params(axis='y',which='both', right=False, left=False) # turn off the little axis ticks on the y-axis, look like minus signs

        # Create colorbar
        if cbar_kwargs is None: cbar_kwargs = dict(pad=0.08, aspect=40, shrink=0.6,) # set some default values for the size of the color bar
        cbar = ax.figure.colorbar(im, **cbar_kwargs)
        cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(data_merged.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data_merged.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color=grid_color, linestyle='-', linewidth=3)
        ax.tick_params(which="minor", top=False, left=False)

        # Set the threshold to switch textcolors to suit the images color range if not given 
        if threshold is None:
            threshold = plot_range[1]/2.    
        #originally, im.norm(threshold) was used to scale 
        # the color range is always between 0 and 1, so threshold was scaled
        # Add text to each square
        for i in range(data_merged.shape[0]):
            for j in range(data_merged.shape[1]):
                number = round(data_merged[i, j],2) # get the number to write in this particular cell
                if number == 0: number = 0 # to avoid things like -0.0
                text = im.axes.text(j, i, number, ha="center", va="center", 
                    color=textcolors[int(np.abs(data_merged[i, j]) > threshold)], **kwargs) # set color depending on value compared to threshold

        # Add a title
        plt.title(title)

        plt.tight_layout()
        
        # Save the plot if needed
        if filename is not None:
            plt.savefig(f'{filename}.pdf')

        # Show the figure
        #plt.show()

        return ax

    def _variability_one_ensemble(self, metric: str, ensemble: Ensemble, period: str = None, wind_speed: bool = True, relative: bool = False):
        """ calculate the variability (standard deviation) for each ensemble member 
        and substract the standard deviation of the reference from it
        metric: the variability metric to compute, std or rcov
        ensemble: ensemble to compute standard deviations for
        period: period over which the wind speeds should be averaged/taken the median (depends on metric), 
                options: year, month, season, none (so no averaging is done)
        wind_speed: should the wind speed or wind direction be saved? if True, wind speed, if False wind direction
        relative: should the absolute values be saved (False)? or the bias compared to the reference (True)? this is only possible if the ensemble has a reference
        """

        # Initialise empty array to save the variabilities
        values = []

        # Convert period name to lowercase (capital letters are useful when adding map/table labels)
        if period is not None:
            period = period.lower()

        # Convert metric name to lowercase (capital letters are useful when adding map/table labels)
        metric = metric.lower()

        # Initialise the groupby string depending on the period given
        match period:
            case ''|'ungrouped'|'none': groupby = 'time' #this causes each timestep to be a group, so in fact no groups are formed
            case 'month': groupby = 'time.month'
            case 'year': groupby = 'time.year'
            case 'season': groupby = 'time.season'

        # Calculate the desired metric
        match metric:
            case 'std'|'standard deviation'|'standard_deviation':
                # Calculate the standard deviation for the reference
                #first group the data with period specified above, take mean per group, then std over these means:
                if ensemble.ref is not None:
                    std_ref = ensemble.ref.data.groupby(groupby, squeeze = False).mean(dtype=np.float64).std(dtype = np.float64)

                    # If not relative, add the stds for the ref to the list, depending on whether wind speed or wind direction is needed
                    if not relative:
                        if wind_speed: values.append(std_ref.wind_speed.data)
                        else: values.append(std_ref.wind_direction.data)
                else:
                    if relative:
                        raise ValueError(f'Relative standard deviations can not be computed: no reference is given')

                # Loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the std for the ensemble member
                    std = winddataset.data.groupby(groupby, squeeze = False).mean(dtype=np.float64).std(dtype = np.float64)
                    # add the stds -std_ref to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        if relative: values.append(std.wind_speed.data - std_ref.wind_speed.data)
                        else: values.append(std.wind_speed.data)
                    else:
                        if relative: values.append(std.wind_direction.data - std_ref.wind_direction.data)
                        else:  values.append(std.wind_direction.data)

            case 'rcov':
                # Calculate the RCoV: robust coefficient of variance (see e.g. Gunturu 2012, Lee 2018)
                # somewhat of an own interpretation with the grouping to obtain rcov for different periods
                # Calculate rcov for the reference
                # Group the data according to the specified period and take medians (to get data for this frequency)
                if ensemble.ref is not None:
                    ref_grouped = ensemble.ref.data.groupby(groupby, squeeze = False).median()
                    mad = np.abs(ref_grouped - ref_grouped.median()).median() # calculate the mad = median absolute deviaion from the median
                    rcov_ref = mad/ref_grouped.median() # and then normalise this by the median

                    # Save the reference if not relative, depending on wind speed or wind direction
                    if not relative:
                        if wind_speed: values.append(rcov_ref.wind_speed.data)
                        else: values.append(rcov_ref.wind_direction.data)
                else:
                    if relative:
                        raise ValueError(f'Relative standard deviations can not be computed: no reference is given')

                # Loop over all members of the ensemble
                for winddataset in ensemble.data_list:

                    # calculate the rcov for the ensemble member
                    # Group the data according to the specified period and take medians (to get data for this frequency)
                    model_grouped = winddataset.data.groupby(groupby, squeeze = False).median()
                    mad = np.abs(model_grouped - model_grouped.median()).median() #calculate the mad = median absolute deviaion from the median
                    rcov = mad/model_grouped.median() # and then normalise this by the median

                    # add the rcovs - rcov_ref to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        if relative: values.append(rcov.wind_speed.data - rcov_ref.wind_speed.data)
                        else: values.append(rcov.wind_speed.data)
                    else:
                        if relative: values.append(rcov.wind_direction.data - rcov_ref.wind_direction.data)
                        else: values.append(rcov.wind_direction.data)

            case _ :
                # Throw an error in all other cases: the metric is not defined
                raise ValueError(f'{metric} is not a known variability metric')

        # Return the full list
        return values
    
class Validator(Comparator):
    """ Compares Ensembles of WindDataSets to a the given reference, by computing error metrics and plotting them"""

    def __init__(self, ensembles):
        """ initialisation of the Validator
        ensembles: the list of ensembles this validator should validate, should have a common lat-lon grid
        """

        # Check if the ensembles have a common grid, raise a warning if not (since the same grid is not necessarily needed for e.g. comparing to obs)
        for ens in ensembles:
            if not ens.check_one_grid():
                warnings.warn(f'Ensemble {ens.name} does not have a common grid, which is expected to create a Validator.')

        # initialise ensemble field with the ensemble on a uniform grid
        self.ensembles = ensembles

        print(f'Validator created with {len(ensembles)} ensembles')

    def __repr__(self):
        """ Representation of the Validator for printing
        """
        return f'Validator with ensembles: {self.ensembles}'

    def _error_one_ensemble(self, metric: str, ensemble: Ensemble, wind_speed: bool = True):
        """ Compute error metric, reducing spatial and time dimensions, for one ensemble
        metric: String indicating which error metric should be calculated, e.g. rmsd, bias, ...
        ensemble: the Ensemble to compute the error metric for
        """

        #initialise empty list to save errors for each member of the ensemble
        errors = []

        #convert metric name to lowercase (capital letters are useful when adding table labels)
        metric = metric.lower()

        #compute the metric
        match metric:

            case 'bias':
                # bias of the mean wind speed/direction (over time, full period, and space), ensemble member - reference
                # loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the bias
                    bias = winddataset.data.mean(dtype=np.float64) - ensemble.ref.data.mean(dtype=np.float64)
                    # add the errors to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        errors.append(bias.wind_speed.data)
                    else:
                        errors.append(bias.wind_direction.data)

            case 'bias_exact':
                # bias of the wind speed/direction, so comparing timestep by timestep and then taking the mean (time and space)
                # ensemble member - reference
                # so not really what we should be looking at for climate data! but interesting for comparison with bias above
                # loop over all members of the ensemble
                print('Warning: this metric will return NaNs if the model data is not saved at the same timesteps as the reference data.')
                for winddataset in ensemble.data_list:
                    # align the time to the reference dataset: only timesteps available in both datasets are kept
                    # so if the edges don't perfectly match these are ignored
                    # and weird timesteps from converting calendars are ignored
                    member, ref = xr.align(winddataset.data, ensemble.ref.data, join='inner')
                    # calculate the exact bias: difference between each timestep at each location, then mean over space and time
                    bias_exact = (member - ref).mean(dtype=np.float64)
                    # add the errors to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        errors.append(bias_exact.wind_speed.data)
                    else:
                        errors.append(bias_exact.wind_direction.data)

            case 'relative bias':
                # relative bias = (model -  ref)/ref
                # loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the rel bias
                    rel_bias = (winddataset.data.mean(dtype=np.float64) - ensemble.ref.data.mean(dtype=np.float64))/ensemble.ref.data.mean(dtype=np.float64)*100
                    # add the errors to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        errors.append(rel_bias.wind_speed.data) #*100 to express in percentages
                    else:
                        errors.append(rel_bias.wind_direction.data)

            case 'rmsd':
                # rmsd = root mean square difference, to evaluate annual cycle (Hahmann 2022)
                #loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the rmsd (rmse for monthly means)
                    rmsd = score.rmse(winddataset.data.groupby('time.month').mean(dtype=np.float64), 
                        ensemble.ref.data.groupby('time.month').mean(dtype=np.float64), dim = None, skipna = True) #skipna needed for land sea mask (contains nans)
                    # add the errors to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        errors.append(rmsd.wind_speed.data)
                    else:
                        errors.append(rmsd.wind_direction.data)
            
            case 'emd'|'wasserstein':
                # earth mover's distance emd: evaluate differences in shape between two distributions
                #loop over all members of the ensemble and calculate emd, depending on whether wind speed or wind direction is needed
                for winddataset in ensemble.data_list:
                    # split by wind speed and wind direction because this takes a lot of time
                    # the other metrics are much faster, so not worth splitting
                    if wind_speed:
                        #to use this function, the data needs to be flattened
                        wspd_flat = winddataset.data.wind_speed.values.flatten()
                        wspd_flat_ref = ensemble.ref.data.wind_speed.values.flatten()
                        #calculate wasserstein distance, filtering out the nans cause the function just returns nan otherwise
                        errors.append(wasserstein_distance(wspd_flat[~np.isnan(wspd_flat)], wspd_flat_ref[~np.isnan(wspd_flat_ref)]))
                    else:
                        wdir_flat = winddataset.data.wind_direction.values.flatten()
                        wdir_flat_ref = ensemble.ref.data.wind_direction.values.flatten()
                        errors.append(wasserstein_distance(wdir_flat[~np.isnan(wdir_flat)], wdir_flat_ref[~np.isnan(wdir_flat_ref)]))

            case _ :
                # Throw an error in all other cases: the metric is not defined
                raise ValueError(f'{metric} is not a known error metric')

        return errors
            
    def _spatial_error_one_ensemble(self, metric: str, ensemble: Ensemble, wind_speed: bool = True):
        """ Compute error metric for one ensemble, reducing time dimension and keeping the spatial dimension, can be used to make maps
        metric: String indicating which error metric should be calculated, e.g. rmsd
        wind_speed: if True, errors are calculated for the wind speed, if False, errors are calculated for the wind direction
        """
        
        #initialise empty list to store error for each ensemble member
        errors = [] 

        #convert metric name to lowercase (capital letters are useful when adding map labels)
        metric = metric.lower()

        #compute the metric
        match metric:
            case 'bias':
                # calculate the bias of the means over the time dimension (over full period)
                for winddataset in ensemble.data_list:
                    bias = winddataset.data.mean(dim='time', dtype=np.float64) - ensemble.ref.data.mean(dim='time', dtype=np.float64)
                    #alternative: score.me(winddataset.data.mean(dim='time', dtype=np.float64), self.ensemble.ref.data.mean(dim='time', dtype=np.float64), dim = [])
                    #gives the same result, might be faster?
                    if wind_speed:
                        errors.append(bias.wind_speed)
                    else:
                        errors.append(bias.wind_direction)
            
            case 'bias_exact':
                # bias of the wind speed/direction, so comparing timestep by timestep and then taking the mean (space)
                # ensemble member - reference
                # so not really what we should be looking at for climate data! but interesting for comparison with bias above
                # loop over all members of the ensemble
                print('Warning: this metric will return NaNs if the model data is not saved at the same timesteps as the reference data.')
                for winddataset in ensemble.data_list:
                    # align the time to the reference dataset: only timesteps available in both datasets are kept
                    # so if the edges don't perfectly match these are ignored
                    # and weird timesteps from converting calendars are ignored
                    member, ref = xr.align(winddataset.data, ensemble.ref.data, join='inner')
                    # calculate the exact bias: difference between each timestep at each location, then mean over time
                    exact_bias = (member - ref).mean(dim = 'time', dtype=np.float64)
                    if wind_speed:
                        errors.append(exact_bias.wind_speed)
                    else:
                        errors.append(exact_bias.wind_direction)

            case 'rmsd':
                # rmsd = root mean square difference, to evaluate annual cycle (Hahmann 2022)
                for winddataset in ensemble.data_list:
                    # calculate the rmsd (rmse for monthly means) for each ensemble member
                    rmsd = score.rmse(winddataset.data.groupby('time.month').mean(dtype=np.float64), 
                        ensemble.ref.data.groupby('time.month').mean(dtype=np.float64), dim = 'month')
                    if wind_speed:
                        errors.append(rmsd.wind_speed)
                    else:
                        errors.append(rsmd.wind_direction)

            case _ :
                # Throw an error in all other cases: the metric is not defined
                raise ValueError(f'{metric} is not a known error metric')

        return errors
    
    def _spatial_variability_one_ensemble(self, metric: str, ensemble: Ensemble, period: str = None, wind_speed: bool = True):
        """ calculate the spatial variability for each ensemble member 
        metric: the variability metric to compute, std or rcov
        ensemble: ensemble to compute standard deviations for
        period: period over which the wind speeds should be averaged/taken the median (depends on metric), 
                options: year, month, season, none (so no averaging is done)
        wind_speed: should the wind speed or wind direction be saved? if True, wind speed, if False wind direction
        """

        # Initialise empty array to save the variabilities
        values = []

        # Convert period name to lowercase (capital letters are useful when adding map/table labels)
        if period is not None:
            period = period.lower()

        # Convert metric name to lowercase (capital letters are useful when adding map/table labels)
        metric = metric.lower()

        # Initialise the dimension string (used to select the dimension to reduce, namely the time dimension after grouping)
        dim = period

        # Initialise the groupby string depending on the period given, to group the data. In case of no period, the dim is also altered
        match period:
            case 'none': groupby = 'time'; dim = 'time' #this causes each timestep to be a group, so in fact no groups are formed
            case 'month': groupby = 'time.month'
            case 'year': groupby = 'time.year'
            case 'season': groupby = 'time.season'

        # Calculate the desired metric
        match metric:
            case 'std'|'standard deviation'|'standard_deviation':
                # Calculate the standard deviation for the reference, reducing only the time dimension constructed above
                #first group the data with period specified above, take mean per group, then std over these means:
                std_ref = ensemble.ref.data.groupby(groupby).mean(dtype=np.float64).std(dim = dim, dtype = np.float64)
                if wind_speed: values.append(std_ref.wind_speed)
                else: values.append(std_ref.wind_direction)

                # Loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the std for the ensemble member
                    std = winddataset.data.groupby(groupby).mean(dtype=np.float64).std(dim = dim, dtype = np.float64)
                    # add the stds to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        values.append(std.wind_speed)
                    else:
                        values.append(std.wind_direction)

            case 'rcov':
                # Calculate the RCoV: robust coefficient of variance (see e.g. Gunturu 2012, Lee 2018)
                # somewhat of an own interpretation with the grouping to obtain rcov for different periods

                # Calculate rcov for the reference
                # Group the data according to the specified period and take medians (to get data for this frequency)
                ref_grouped = ensemble.ref.data.groupby(groupby).median()
                mad = np.abs(ref_grouped - ref_grouped.median(dim = dim)).median(dim = dim) # calculate the mad = median absolute deviation from the median, while only reducing the time dimension
                rcov_ref = mad/ref_grouped.median(dim = dim) # and then normalise this by the median

                # Add the reference rcov to the list, wind speed or wind direction
                if wind_speed: values.append(rcov_ref.wind_speed)
                else: values.append(rcov_ref.wind_direction)

                # Loop over all members of the ensemble
                for winddataset in ensemble.data_list:
                    # calculate the rcov for the ensemble member
                    # Group the data according to the specified period and take medians (to get data for this frequency)
                    model_grouped = winddataset.data.groupby(groupby).median()
                    mad = np.abs(model_grouped - model_grouped.median(dim = dim)).median(dim = dim) #calculate the mad = median absolute deviation from the median
                    rcov = mad/model_grouped.median(dim = dim) # and then normalise this by the median
                    
                    # add the rcovs to the list, depending on whether wind speed or wind direction is needed
                    if wind_speed:
                        values.append(rcov.wind_speed)
                    else:
                        values.append(rcov.wind_direction)

            case _ :
                # Throw an error in all other cases: the metric is not defined
                raise ValueError(f'{metric} is not a known variability metric')

        # Return the full list
        return values

    def calculate_metric(self, metric: str, period: str =  None, wind_speed = True, relative = None):
        """ calculate the error metric or variability metric for each ensemble member and the reference
        metric: the variability metric to compute, std or rcov
        period: optional, only needed for variability metrics, the period over which the wind speeds should be looked at
                options: year, month, season, none (no averaging is done)
        wind_speed: should the wind speed or wind direction be saved? if True, wind speed, if False wind direction
        relative: optional, should variabilities be compared to the ref (True)? or should they be absolute (False)? only needed for variabilities
        """
        # Initialise empty list
        output = []

        # Compute the metric for each ensemble and add to the list
        # if no period is given, the metric is an error metric
        if period is None:
            for ens in self.ensembles:
                error_ens = self._error_one_ensemble(metric = metric, ensemble=ens, wind_speed = wind_speed)
                output.append(error_ens)
        # if a period is given, the metric is a variability metric
        else:
            for ens in self.ensembles:
                error_ens = self._variability_one_ensemble(metric = metric, period =  period, ensemble=ens, 
                    wind_speed = wind_speed, relative = relative)
                output.append(error_ens)

        # Return output
        return output

    def calculate_spatial_metric(self, metric: str, period = None, wind_speed: bool = True):
        """Compute error metric, reducing temporal dimension
        metric: String indicating which error metric should be calculated, e.g. rmsd, bias, ...
        period: optional, only needed for variability metrics, the period over which the wind speeds should be looked at
                options: year, month, season, none (no averaging is done)
        wind_speed: should the wind speed or wind direction be saved? if True, wind speed, if False wind direction
        returns an array of arrays, containing errors for each ensemble member in each ensemble
        """

        # Initialise empty list
        output = []

        # Compute the error for each ensemble and add to the list, variability if period is given, otherwise error metric
        if period is None:
            for ens in self.ensembles:
                error_ens = self._spatial_error_one_ensemble(metric=metric, ensemble=ens, wind_speed = wind_speed)
                output.append(error_ens)
        else:
            for ens in self.ensembles:
                error_ens = self._spatial_variability_one_ensemble(metric=metric, period = period, ensemble=ens, wind_speed = wind_speed)
                output.append(error_ens)

        # Return output
        return output

    def plot_spatial_metric(self, metric: str, plot_range: list = None, period: str = None, wind_speed: bool = True, cmap: str ='viridis', colorbar_label: str = None, text_size: float = 7.5):
        """ plot an error metric or variability metric computed using the calculate_spatial_metric function 
        metric: the error metric to be plotted
        plot_range: optional, range to be used for the colorbar, if not specified the overall min and max of the ensemble are used
        period: optional, only needed for variability metrics, the period over which the wind speeds should be looked at
                options: year, month, season, none (no averaging is done)
        wind_speed: optional, if True the wind speed is plotted, if False the wind direction is plotted
        cmap: optional, the colormap to use 
        colorbar_label: optional, label to add to the colorbar
        text_size: optional, to change the text size on all plot elements
        """

        # Calculate the error metric for each ensemble member in each ensemble
        errors = self.calculate_spatial_metric(metric, period = period, wind_speed =  wind_speed)
            
        # Calculate the range as min, max of all errors across all ensembles
        plot_range = [np.min([np.min(ens_error) for ens_error in errors]), np.max([np.max(ens_error) for ens_error in errors])] #set range

        # symmetrize the range if the minimum is negative or if the color map is seismic (divergent color scale)
        if (plot_range[0]<0) | (cmap == 'seismic'):
            abs_max = np.max(np.abs(plot_range))
            plot_range = [-abs_max,abs_max]
            #change the colormap to a diverging one while we're at it
            if cmap == 'viridis': # default value of cmap (nothing is done when a colormap is specified)
                cmap = 'seismic'
                # coolwarm: some sort of gray as 0, so not very clear
                # alternatives with white as 0: bwr, seismic, RdBu 

        # Loop over the ensembles, constructing the plots
        for e in range(len(self.ensembles)): 

            # Get the ensemble
            ens = self.ensembles[e]

            # Print ensemble name
            print(ens.name)
            
            # Get the error for this ensemble 
            error_ens =  errors[e]

            # Check if the reference is included in the list (by comparing the number of errors to the number of ensemble members)
            if len(error_ens) == len(ens.data_list)+1:
                relative = False
            else: relative = True

            # Construct the label of the colorbar if not given
            if colorbar_label is None:
                colorbar_label = self._construct_label(wind_speed = wind_speed, metric = metric, period = period, relative = relative)

            # Plot the reference if included (so errors are not relative)
            if not relative:
                self.plot_map(error_ens[0], plot_range = plot_range, colorbar_label=colorbar_label, 
                    title = f'{metric} {ens.ref.name}', cmap = cmap)
                increment = 1 # set the increment to increase the index when looping over the errors below, when ref is present all members are shifted by 1 in the list of errors
            else: 
                increment = 0 # if the reference is not present, no shift is needed
            
            # Loop over the ensemble members' errors, plotting each of them in a map
            for i in range(len(ens.data_list)):

                # Select the error for this ensemble member from the selected ensemble
                error_member = error_ens[i + increment]
                
                # Plot the map
                self.plot_map(error_member, plot_range=plot_range, colorbar_label=colorbar_label,
                    title = f'{metric} {ens.data_list[i].name}', cmap = cmap, text_size = text_size)            
    
    def calculate_weighted_error_metric(self, metric: str, include_reanalysis: bool, periods: list = [None], wind_speed: bool = True, verbose: bool = False):
        """ Calculate the specified error metric, weighted according to the number of timesteps in the reference (returns also split out values)
            expects a list of ensembles each with the same members but different references (can be spatial and point)
        metric: the metric to plot
        include_reanalyis: bool, should spatial winddatasets (which are the reanalyses) be included?
        period: optional, the periods to be used if the metric is a variability metric
        wind_speed: should the metrics be calculated for the wind speed (True) or the wind direction (False)
        verbose: should the factors be printed
        """

        # Check if the list of ensembles is of the right shape: a list of ensembles with the same members but different references
        if len(np.unique([len(ens.data_list) for ens in self.ensembles])) != 1:
            raise ValueError(f'Ensembles do not all have the same number of members, which is expected to calculate weighted error metrics. The ensembles are supposed to have the same members but different references.')
        member_names = [[mem.name for mem in ens.data_list] for ens in self.ensembles]
        if not all(names == member_names[0] for names in member_names):
            raise ValueError(f'Ensembles do not all have the same members judging from the names, which is expected to calculate weighted error metrics. The ensembles are supposed to have the same members but different references.')
        
        # Compute the metrics, and add it to values for wind speed or wind direction, loop over periods if needed
        metrics_periods = []
        for p in periods:
            metrics_periods.append(self.calculate_metric(metric, wind_speed = wind_speed, period = p, relative = True)) # absolute weighted metrics does not make sense (absolute values from the models, but weighted according to ref?)
            # relative is true for variability metric (so we get relative to reference), does not make sense to compute weighted means otherwise

        # Get the number of non nan timesteps for each of the pointdatasets (i.e. lidar observations)
        total_timesteps = [sum(~np.isnan(ens.ref.data.wind_speed)).data for ens in self.ensembles if type(ens.ref) ==  PointWindDataSet]
        total_timesteps = sum(np.array(total_timesteps)) # calculate sum over all datasets

        # Add the timesteps for the spatial datasets if needed
        if include_reanalysis:
            # The spatial datasets are reanalysis datasets which don't contain nans except where the domain is clipped
            # so we just use the maximal number of non nan timesteps
            timesteps_spatial = [sum(~np.isnan(ens.ref.data.wind_speed)).max().data for ens in self.ensembles if type(ens.ref) ==  SpatialWindDataSet]

            # Calculate the total number of timesteps (not nans) of the lidars for the normalisation
            total_timesteps = sum(np.array(timesteps_spatial)) + total_timesteps
        if verbose: print(f"Total number of timesteps: {total_timesteps}")

        # Calculate the weighted sum of the metrics across the different references using the number of non nan timesteps as weights
        sums_periods = []
        for metrics in metrics_periods:
            weighted_sum = 0
            for i, ens in enumerate(self.ensembles):
                if verbose: print(ens.name)
                # calculate the weight factor depending on point or spatial dataset
                if type(ens.ref) == PointWindDataSet:
                    if verbose: print(f'Number of timesteps: {sum(~np.isnan(ens.ref.data.wind_speed)).data}')
                    factor = sum(~np.isnan(ens.ref.data.wind_speed)).data/total_timesteps
                else:
                    if include_reanalysis:
                        factor = (sum(~np.isnan(ens.ref.data.wind_speed))).max().data/total_timesteps
                        if verbose: print(f"Number of timesteps : {(sum(~np.isnan(ens.ref.data.wind_speed))).max().data}")
                    else: 
                        factor = 0 # set factor to zero if these datasets don't need to be included
                # Add the weighted metric to the sum
                weighted_sum += np.array(metrics[i]) * factor
                if verbose: print (f'Factor: {factor:.2f}')
            sums_periods.append(weighted_sum)

        return sums_periods, metrics_periods

    def plot_table_weighted_error_metric(self, metric: str, periods: list = [None], wind_speed: bool = True, include_split: bool = False, include_reanalysis: bool = False, save: bool = False, ax = None, cbar_kwargs: dict = None, verbose: bool = False, **kwargs):
        """ Plots a table of the specified error metric, weighted according to the number of timesteps in the reference
            expects a list of ensembles each with the same members but different references (can be spatial and point)
        metric: the metric to plot
        period: optional, the periods to be used if the metric is a variability metric
        wind_speed: should the metrics be calculated for the wind speed (True) or the wind direction (False)
        include_split: should the metrics split out by dataset also be plotted? only for non variability metrics with different periods
        include_reanalysis: should the reanalysis be included in the weighted mean? only for variability metrics with different periods, otherwise by default both are plotted
        save: should the plot be saved?
        ax: optional, ax to add the plot to
        cbar_kwargs: extra arguments handed to the colorbar
        kwargs: extra arguments handed to plot_table
        """

        # Check if there is no conflict of arguments
        if include_split and None not in periods:
            raise ValueError(f'Combination of including split out members for different periods of variability metric is not implemented (yet?).')
        if include_reanalysis and None in periods:
            print('By default both the weighted mean with and without reanalyses are plotted, so the include_reanalysis keyword has no effect here. It only applies for variability metrics with different periods.')

        # Calculate the default and weighted metrics, both with and without reanalyses (could be done a bit more efficiently by altering the function above to avoid repeated calculation of the metrics)
        weighted_metrics_lidars, metrics = self.calculate_weighted_error_metric(metric = metric, periods = periods, wind_speed = wind_speed, include_reanalysis = False, verbose = verbose)
        weighted_metrics_all, _ = self.calculate_weighted_error_metric(metric = metric, periods = periods, wind_speed = wind_speed, include_reanalysis = True, verbose = verbose)
        if include_split: 
            metrics[0].append([np.array(a) for a in weighted_metrics_lidars[0]]) #stick them together in the right shape
            metrics[0].append([np.array(a) for a in weighted_metrics_all[0]])

        # Set plot_range, over all ensembles and metrics
        plot_range = [np.min(metrics), np.max(metrics)]

        # Save the plot if needed by setting the filename to hand
        if save: filename = f'table_weighted_errors'
        else: filename = None

        # Set the row and column names
        if None not in periods: colnames = [self.ensembles[0].data_list[0].data.frequency.capitalize() if p.lower() == 'none' else p for p in periods] 
        else: 
            colnames = []
            if include_split:
                colnames = [ens.name for ens in self.ensembles]
            colnames.append('Weighted mean lidars')
            colnames.append('Weighted mean')

        member_names = [winddataset.name for winddataset in self.ensembles[0].data_list]
        
        # Set the data
        if None not in periods:
            if include_reanalysis: data = weighted_metrics_all
            else: data = weighted_metrics_lidars 
        else:
            if include_split: data = metrics[0]
            else: data = [[np.array(a) for a in weighted_metrics_lidars[0]], [np.array(a) for a in weighted_metrics_all[0]]]
        
        # Plot the table
        if None not in periods: relative = True # variability metrics are relative here
        else: relative =  False
        self.plot_table(data, colnames = colnames, rownames = member_names, plot_range = plot_range,
            title = '', filename = filename, colorbar_label = self._construct_label(metric = metric, wind_speed = wind_speed, relative = relative), cbar_kwargs = cbar_kwargs, ax = ax, **kwargs)

    def plot_table_error_metric(self, metrics: list, wind_speed: bool = True, means: bool = False, save: bool = False, merge: bool = False, ax = None, cbar_kwargs: dict = None, **kwargs):
        """ Plots a table of the specified error metrics
        metrics: list of Strings containing the names of the metrics to plot
        wind_speed: should the metrics be calculated for the wind speed (True) or the wind direction (False)
        means: should the ensemble means be added to the tables?
        save: should the plot be saved?
        merge: should there be multiple tables or one merged one? this is only works for just one metrics, and when all ensembles have the same members
        ax: optional, ax to add the plot to
        cbar_kwargs: extra arguments handed to the colorbar
        kwargs: extra arguments handed to plot_table
        """

        # Initialise empty array to save the metrics
        all_metrics = []
        
        # Compute the metrics, and add it to values for wind speed or wind direction
        for m in metrics:
            metric = self.calculate_metric(m, wind_speed=wind_speed)
            all_metrics.append(metric)

        # Set plot_range, over all ensembles and metrics
        plot_range = [np.min(all_metrics), np.max(all_metrics)]

        # Plot the table for each of the ensembles, looping over the ensembles
        for i in range(len(self.ensembles)):

            # Initialise an empty array to save the metrics for this ensemble
            values =[]

            # Add each metric for this ensemble to values
            for m in range(len(metrics)):
                metric_m_i = all_metrics[m][i] #index m selects the metric, index i selects the ensemble
                if means:
                    metric_m_i.append(np.mean(metric_m_i)) # add the ensemble mean for this metric if needed
                values.append(metric_m_i) # add this metric to the values for this ensemble

            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in self.ensembles[i].data_list]

            # Add the ensemble mean to the names if needed
            if means:
                member_names.append('$\\bf{Ensemble\ mean}$') # add the ensemble mean

            # Save the plot if needed by setting the filename to hand
            if save:
                filename = f'{self.ensembles[i].name}_table_errors'
            else: filename = None

            # Plot the table
            if not merge:
                self.plot_table(values, colnames = metrics, rownames = member_names, plot_range = plot_range,
                title = self.ensembles[i].name, filename = filename, cbar_kwargs = cbar_kwargs, **kwargs)
        
        if merge:
            if len(metrics)>1: print('Warning: the merge option when plotting is only meant for a single metric.')
            # squeeze: removes redundant dimension to allow plotting with the same method
            self.plot_table(np.squeeze(np.array(all_metrics)), colnames = [ens.name for ens in self.ensembles], rownames = member_names, plot_range = plot_range,
                title = '', filename = filename, colorbar_label= metrics[0], ax = ax, cbar_kwargs= cbar_kwargs, **kwargs)

    def plot_table_variabilities(self, metric, periods, wind_speed = True, relative = False, means = False, save = False, **kwargs):
        """ Plots a table of the variability metric for the specified periodicities for each ensemble
        metric: variability metric
        periods: list of Strings, containing the periodicities to plot
        relative: should bias metric compared to ref be computed (if true)? or absolute values (False)
        wind_speed: should the metrics be calculated for the wind speed (True) or the wind direction (False)
        means: should the ensemble means be added to the tables?
        save: should the plot be saved?
        """

        # Initialise empty array to save the standard deviations
        values = []
        
        # Compute the variability metric for each period, and add it to values
        for p in periods:
            std = self.calculate_metric(metric = metric, period = p, wind_speed = wind_speed, relative = relative)
            values.append(std)

        # Set the plot_range
        plot_range = [np.min(values), np.max(values)]

        # Construct the label of the color bar (no period since there are multiple)
        label = self._construct_label(metric = metric, wind_speed = wind_speed, relative = relative, period = None)

        # Loop over the ensembles to make the plots
        for i in range(len(self.ensembles)):

            # Get the ensemble from the list
            ens = self.ensembles[i]

            # Initialise an empty array to save the variability metrics for different periods for this ensemble
            var_ens =[]

            # Add each period variability for this ensemble to std_ens
            for p in range(len(periods)):
                var_ens_i_p = values[p][i] #index p selects the metric, index i selects the ensemble
                if means:
                    var_ens_i_p.append(np.mean(var_ens_i_p)) # add the ensemble mean for this period if needed
                var_ens.append(var_ens_i_p)

            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in ens.data_list]

            # Add the ensemble mean to the names if needed
            if means:
                member_names.append('$\\bf{Ensemble\ mean}$') # add the ensemble mean

            # Add the name of the reference dataset if non relative values are given
            if not relative:
                member_names.insert(0, ens.ref.name) #add reference name 

            # Save the plot if needed by setting the filename to hand
            if save:
                filename = f'{ens.name}_table_variabilities'
            else: filename = None

            # Plot the table
            colnames = [self.ensembles[0].data_list[0].data.frequency.capitalize() if p.lower() == 'none' else p for p in periods] # Assumes daily dat
            self.plot_table(var_ens, colnames = colnames, rownames = member_names, plot_range = plot_range, title = ens.name, 
                colorbar_label= label, filename = filename, **kwargs)

    def plot_table_seasons(self, metric, period = None, relative = False, wind_speed = True, **kwargs):
        """ Plot a table of the specified error/variability metric, split by season
        metric: error metric to be calculated
        period: only needed for variability metrics, determines over which period the variability is calculated
        relative: only needed for variability metrics, determines if absolute or values compared to reference are returned
        wind_speed: optional, should wind_speed or wind direction be used
        """

        # Initialise an empty array to save the metric
        values = []

        # Loop over the ensembles and calculate the metric for each season
        for ens in self.ensembles:
            # Select each season from the ensemble
            winter = ens.select_season('winter')
            spring = ens.select_season('spring')        
            summer = ens.select_season('summer')        
            autumn = ens.select_season('autumn')

            # Calculate the error metric for each season and add the results to values
            validator_seasons = Validator([winter, spring, summer, autumn]) #create a validator object for this season, no grid interpolation is done
            error = validator_seasons.calculate_metric(metric, period = period, wind_speed = wind_speed, relative=relative) # compute the error metric
            values.append(error)

        # Set the plot range as the min and max of the metric values over all ensembles
        plot_range = [np.min(values), np.max(values)]

        # Construct the label of the color bar
        label = self._construct_label(wind_speed = wind_speed, metric = metric, period = period, relative = relative)

        # Construct the column names as the seasons
        seasons = ['DJF', 'MAM','JJA', 'SON']

        # Loop over the ensembles to make the plots
        for i in range(len(self.ensembles)):

            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in self.ensembles[i].data_list]

            # Add the name of the reference dataset if non relative values are given for a variability metric (indicated by the fact that a period is given)
            if not relative and period is not None:
                member_names.insert(0, ens.ref.name) #add reference name 

            # Plot the table
            self.plot_table(values[i], colnames = seasons, rownames = member_names, plot_range = plot_range,
                title=self.ensembles[i].name, colorbar_label= label, **kwargs)

    def plot_table_land_sea(self, metric, period = None, relative = False, wind_speed = True, n_buffer = 0, **kwargs):
        """ Plot a table of the specified error metric, split by land/sea
        metric: error metric to be calculated
        period: only needed for variability metrics, determines over which period the variability is calculated
        relative: only needed for variability metrics, determines if absolute or values compared to reference are returned
        wind_speed: optional, should wind_speed or wind direction be used
        n_buffer: the number of neighbours used to buffer, by default no buffer. The land sea mask by default
        """

        # Initialise an empty array to save the metric
        values = []

        # Loop over the ensembles to calculate the metric for all ensembles for land and sea
        for ens in self.ensembles:

            # Select land and sea from ensemble
            land = ens.mask_land_sea(sea = False, n_buffer = n_buffer)    
            sea = ens.mask_land_sea(sea = True, n_buffer =  n_buffer)

            # Calculate the error metric for each season
            surface_validator = Validator([land, sea]) #create a validator object containing both land and sea
            error = surface_validator.calculate_metric(metric, period = period, wind_speed = wind_speed, relative = relative) # compute the error metric
            values.append(error) # save the results

        # Set the plot range as the minimum and maximum value of all values for the metric
        plot_range = [np.min(values), np.max(values)]

        # Construct the label of the color bar
        label = self._construct_label(wind_speed = wind_speed, metric = metric, period = period, relative = relative)

        # Construct the colnames as surfaces
        surfaces = ['Land', 'Sea']

        # Loop over the ensembles again, now plotting the table
        for i in range(len(self.ensembles)):
            
            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in self.ensembles[i].data_list]

            # Add the name of the reference dataset if non relative values are given for a variability metric (indicated by the fact that a period is given)
            if not relative and period is not None:
                member_names.insert(0, ens.ref.name) #add reference name 

            # Plot the table
            self.plot_table(values[i], colnames = surfaces, rownames = member_names, plot_range = plot_range, 
                colorbar_label= label, title = self.ensembles[i].name, **kwargs)

    def plot_table_locations(self, metric, lat, lon, period = None, relative = False, wind_speed = True, **kwargs):
        """ Plot a table of the specified error metric, at each of the specified locations (grid point closest to the locations is used, no interpolation)
        metric: error metric to be calculated
        lat: the latitudes of the locations
        lon: the longitudes of the locations
        period: only needed for variability metrics, determines over which period the variability is calculated
        relative: only needed for variability metrics, determines if absolute or values compared to reference are returned
        wind_speed: optional, should wind_speed or wind direction be used
        also plots a map with the selected locations
        """

        # First a map showing the locations is plotted

        # Create the map
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()

        # Plot the positions on the map, add borders and coastlines
        ax.scatter(lon, lat, color ='r', marker = 'x')
        ax.set_extent([longitude_min_domain, longitude_max_domain, latitude_min_domain, latitude_max_domain], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  # Country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Show the map with the positions
        plt.show()

        # Next the actual table is constructed

        # Initialise an empty array to save the metric values at each location
        values = []

        # Calculate the metric for each ensemble
        for ens in self.ensembles:

            # Initialise empty array to save the ensembles at each location
            ens_locations = []

            # Get the ensemble at each location (the closest grid point to the given coordinates)
            for i in range(len(lat)):
                ens_locations.append(ens.sel(dict(lat= lat[i], lon = lon[i]), method='nearest')) # for each member the grid point closest to the location
            
            # Calculate the metric at the locations
            v_location = Validator(ens_locations) #create a validator object for this season, no grid interpolation is done
            error = v_location.calculate_metric(metric, period = period, wind_speed = wind_speed, relative = relative) # compute the error metric
            values.append(error)

        # Set the plot range as the min and max of the metric values over all ensembles
        plot_range = [np.min(values), np.max(values)]

        # Construct column names: locations
        east_west = ['E' if lon_i > 0 else 'W' for lon_i in lon] #depending on the value of the longitude, the suffix should be E or W
        locations = [f'{lat[i]}°N,\n{lon[i]}°{east_west[i]}' for i in range(len(lat))]

        # Construct the label of the color bar
        label = self._construct_label(wind_speed = wind_speed, metric = metric, period = period, relative = relative)

        # Loop over the ensemble to make the plots
        for i in range(len(self.ensembles)):

            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in self.ensembles[i].data_list]

            # Add the name of the reference dataset if non relative values are given for a variability metric (indicated by the fact that a period is given)
            if not relative and period is not None:
                member_names.insert(0, ens.ref.name) #add reference name 

            # Plot the table
            self.plot_table(values[i], colnames = locations, rownames = member_names,
                title = self.ensembles[i].name, plot_range = plot_range, colorbar_label= label, **kwargs)

class Projector(Comparator):
    """ Compares a list of historical ensembles to its future counterparts """
    
    def __init__(self, ensembles: list, test_mode: bool = False, verbose = True):
        """ set up the projector object by taking a list of historical ensembles, constructing the equivalent future ensembles and regridding them
        ensembles: the list of historical ensembles, each should have a common grid among its ensemble members
        test_mode: should the winddatasets be limited in size for quick testing?
        verbose: should progress be printed?
        """

        # Check if the ensembles have a common grid, raise an error if not
        for ens in ensembles:
            if not ens.check_one_grid():
                raise ValueError(f'Ensemble {ens.name} does not have a common grid, which is expected to create a Projector.')

        # Initialise the list of historical ensembles
        self.hist = copy.deepcopy(ensembles)

        # Initialise empty list for future ensembles
        list_fut = []

        # Print a little warning
        print('Projector assumes the historical ensemble has a common grid and regrids the future ensemble to the same grid')

        # Construct the list of future ensembles
        for hist_ens in ensembles:

            # Set the reference to none in the historical ensembles
            hist_ens.ref = None

            # Construct the list of paths for the future ensemble
            paths_fut = []
            for path_hist in hist_ens.path_list:
                path_fut = path_hist.replace('/hist/', '/fut/') #switch folders
                if 'hist-1950' in path_fut: path_fut = path_fut.replace('hist-1950', 'highres-future') #edit the path to get path for the future simulation
                elif 'highresSST-present' in path_fut: path_fut = path_fut.replace('present', 'future')
                paths_fut.append(path_fut)

            # Create the future ensemble
            print(f'Loading future ensemble for {hist_ens.name}')
            names = [winddataset.name for winddataset in hist_ens.data_list] # get the names from the historical simulation
            fut_ens = Ensemble(name = 'Future', path_list = paths_fut, groups = hist_ens.groups, test_mode = test_mode)
            
            # assuming that every ensemble either contains only spatial datasets or only point datasets, apply regridding or selection of closest points
            if type(hist_ens.data_list[0]) == SpatialWindDataSet:
                print(f'Regridding ensemble {hist_ens.name}')
                # Regrid to the grid of the first historical simulation
                fut_ens = fut_ens.one_grid(hist_ens.data_list[0], verbose = verbose) 
                # Set gridpoints where full timeseries is nan in hist also to nan, which works for sea masks without having to redo the sea mask function
                for i in range(len(hist_ens.data_list)):
                    fut_ens.data_list[i].apply_nans_other(hist_ens.data_list[i])
            else:
                print(f'Selecting the closest point {hist_ens.name}')
                fut_ens = fut_ens.get_closest_point(hist_ens.data_list[0])
            list_fut.append(fut_ens)
        
        # Save the list of future ensembles
        self.fut = list_fut

    def __repr__(self):
        """ Representation of the Projector for printing
        """
        return f'Projector with ensembles: {self.hist} and {self.fut}'
    
    def select_members_ensembles(self, selection: list[bool]):
        """Select for each ensemble, future and historical, members according to the selection list.
        selection: list of booleans, determining whether specific members should be kept ot not
        """
        
        # Copy the original projector
        output = copy.deepcopy(self)
        
        # Select the members from each historical ensemble
        for i, ens in enumerate(self.hist):
            output.hist[i] = ens.select_members(selection)
        
        # Select the members from each future ensemble
        for i, ens in enumerate(self.fut):
            output.fut[i] = ens.select_members(selection)

        return output

    def select_season(self, season: str):
        """ Select a season in all ensembles and return a new projector of this season
        season: string indicating which season to select
        """

        # Copy the original projector
        output = copy.deepcopy(self)
        
        # Select the season from each historical ensemble
        for i, ens in enumerate(self.hist):
            output.hist[i] = ens.select_season(season)
        
        # Select the members from each future ensemble
        for i, ens in enumerate(self.fut):
            output.fut[i] = ens.select_season(season)

        return output

    def convert_pvalues_to_pandas(self, pvalues: np.array):
        """ Convert a numpy array of pvalues to a handier pandas dataframe format
        pvalues: the numpy array containing the pvalues
        ens_list: the ensemble list this list of pvalues was based on
        """

        # Convert to dataframe
        pvalues = pd.DataFrame(pvalues)

        # Set row and column names
        pvalues.columns = [mem.name for mem in self.hist[0].data_list] # assuming all of the ensembles have the same members
        row_names = dict(zip(np.arange(0,12), [ens.name for ens in self.hist])) # assuming the ensembles are named meaningfully
        pvalues = pvalues.rename(index = row_names)
        return pvalues

    def statistical_test(self, a, b, test: str, level: float = 0.05, spatial: bool = False):
        """perform a statistical test, returns the pvalue or true false depending on significance level
        a,b: the arrays of values to compare
        test: string specifying which type of test to perform
        level: optional,the significance level to use to convert to a significant/unsignificant mask, 
            when None the p-values themselves are returned
        spatial: should the pvalue be computed at each gridpoint or for all gridpoints aggregated
        """

        # Set the axis depending on whether we want spatial results or not
        if spatial: axis = 0
        else: axis = None

        # match the test name to decide which statistical test to perform
        match test.replace('-', '').lower():
            case 'mannwhitney'|'mannwhitneyu'|'mwu': # equal distribution, assumptions: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
                p = scipy.stats.mannwhitneyu(a, b, nan_policy = 'omit', axis = axis).pvalue 
            case 't'|'t-test'|'ttest': # equal mean, assumptions:  https://en.wikipedia.org/wiki/Student%27s_t-test 
                p = scipy.stats.ttest_ind(a, b, nan_policy = 'omit', axis = axis).pvalue
            case 'ks'|'kolmogorovsmirnov': # equal distribution, samples from same distribution assumptions: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
                p = scipy.stats.ks_2samp(a, b, nan_policy = 'omit', axis = axis).pvalue
            case 'bartlett': # equal variance, assumptions: https://en.wikipedia.org/wiki/Bartlett%27s_test, assumes normal distribution so Levene better alternative
                p = scipy.stats.bartlett(a, b, nan_policy = 'omit', axis = axis).pvalue
            case 'levene': # equal variance, assumptions: https://en.wikipedia.org/wiki/Levene%27s_test
                p = scipy.stats.levene(a, b, nan_policy = 'omit', axis = axis).pvalue          
            case _ :
                # Throw an error in all other cases: the test is not defined
                raise ValueError(f'{test} is not a known test statistic')

        if level is None:
            return p # return the pvalue themselves if no level is specified
        else:
            return np.where(np.isnan(p), np.nan, p < level) # compare the pvalues to the significance level and add to the list for this ensemble, keep nans as nans

    def statistical_test_hist_fut(self, test: str, level: float = 0.05, filename: str = None, spatial: bool = False):
        """perform a statistical test comparing historical and future period, returns result as list of arrays of pvalues per location 
        test: string specifying which type of test to perform
        level: optional,the significance level to use to convert to a significant/unsignificant mask, 
            when None the p-values themselves are returned
        filename: optional, filename to save the pvalues under
        spatial: should the pvalue be computed at each gridpoint or for all gridpoints aggregated
        """

        # Initialise an empty list to save the pvalues
        pvalues = []
        
        # Perform the test for each ensemble member, for each ensemble
        for i in range(len(self.hist)):
            pvalues_ens_i = [] # create an empty list to save pvalues for this ensemble
            
            # Loop over the ensemble members
            for j in range(len(self.hist[i].data_list)):
                hist_j = self.hist[i].data_list[j] # select the historical ensemble member
                fut_j = self.fut[i].data_list[j] # select the future ensemble member
                
                # Calculate the p-value comparing historical and future
                p = self.statistical_test(hist_j.data.wind_speed, fut_j.data.wind_speed, test, level, spatial = spatial)

                pvalues_ens_i.append(p) # save the pvalues for this ensemble member 

            pvalues.append(pvalues_ens_i) # add the pvalues to the full list

        # Save the pvalues if needed
        if filename is not None:
            np.save(filename, pvalues)

        # Return pvalues
        return pvalues

    def plot_map_change_mean_wspd(self, test: str = None, level: float = None, relative: bool = False, cmap: str = 'seismic', plot_mean: bool = False, offsets: list = [-2.2, -2], verbose: bool = False, text_size: float = 6):
        """ For each ensemble member of all ensembles, plot the change of the mean windspeed between historical and future period on a map
        test: optional, name of the statistical test that is performed to get the significance of the change, hatching on the plot
        level: the significance level to use for the statistical test
        cmap: the colormap
        relative: if True, relative bias is plotted, if False, bias is plotted
        plot_mean: optional, determines if the plot should have an annotation with the mean
        offsets: ofsets [x,y] to the annotation of plot mean
        verbose: print the differences for pointwinddatasets
        text_size: size of the text in the figure
        """

        # Create an empty list to save changes in
        diffs = []

        # Loop over the ensembles
        for j in range(len(self.hist)):
            
            ens_hist = self.hist[j]
            ens_fut = self.fut[j]

            # Create empty list to save changes in for this ensemble
            diffs_ens = []

            # Calculate and save the difference of the mean wind speed in the historical and the future period for each ensemble member
            for i in range(len(ens_hist.data_list)):
                fut_i = ens_fut.data_list[i]
                hist_i = ens_hist.data_list[i]
                # fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0
                # and if decrease in the future, we have fut < hist <=> fut - hist < 0
                diff_i = fut_i.data.wind_speed.mean(dim = 'time', dtype = np.float64) - hist_i.data.wind_speed.mean(dim = 'time', dtype = np.float64)
                if relative:
                    diff_i =  diff_i/hist_i.data.wind_speed.mean(dim = 'time', dtype = np.float64)*100 # convert to percentage if needed
                diffs_ens.append(diff_i)

            diffs.append(diffs_ens)

        print('Difference here is fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0 and if decrease in the future, we have fut < hist <=> fut - hist < 0')

        # Calculate the range as min, max of all diffs, need double loop to do this for list of list of xarrays
        max_diffs = []
        min_diffs = []
        for diff_ens in diffs:
            for diff_member in diff_ens:
                min_diffs.append(np.min(diff_member))
                max_diffs.append(np.max(diff_member))
        plot_range = [np.min(min_diffs), np.max(max_diffs)] #set range

        # Symmetrize the range
        abs_max = np.max(np.abs(plot_range))
        plot_range = [-abs_max,abs_max]

        # Construct the unit for the label for the colorbar
        if relative: unit = '(%)'
        else: unit = '(m/s)'

        # Perform the statistical test if needed and create a mask for each ensemble member
        if test is not None:
            mask = self.statistical_test_hist_fut(test, level, spatial = True)

        # Create the plots
        n_maps = sum([type(ens.data_list[0]) is SpatialWindDataSet for ens in self.hist]) * len(self.hist[0].data_list)
        nrows, ncols = almost_factors(n_maps)
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (12, 10))
        
        # Plot a difference map for each ensemble member
        plot_index = 0
        for j, diffs_ens in enumerate(diffs):
            if verbose: print(self.hist[j].name)
            for i, diff in enumerate(diffs_ens):
                # Plot map or print out difference depending on point or spatial winddataset
                if diff.size == 1:
                    if verbose: print(f"{diff.data}% change for {self.hist[j].data_list[i].name}")
                else:
                    # Choose the axis
                    ax = axes.reshape(-1)[plot_index]
                    ax.set_global()
                    # Set the hatching if needed
                    if test is not None:
                        hatch = mask[j][i] # hatching based on the statistical test
                        hatch = np.where(np.isnan(diff.data), np.nan, hatch) # mask nan values from original dataset to avoid hatching outside domain
                        hatch = np.where(hatch  == 1, hatch, np.nan) # put nans wherever the statistical test did not return true
                        test_label = '' #f'({test}, {level*100}%)' # set the label to be used in the plot title
                    else: hatch = None
                    # Set the ticks depending on where the plot is positioned
                    if ax.get_subplotspec().is_first_col() & ax.get_subplotspec().is_last_row(): labels = ['left', 'bottom']
                    elif ax.get_subplotspec().is_last_row(): labels = ['bottom']
                    elif ax.get_subplotspec().is_first_col(): labels = ['left']
                    else: labels = False
                    # Create the plot
                    im = self.plot_map(diff, title = f'{self.hist[j].data_list[i].name}', plot_range = plot_range, labels = labels,
                        cmap = cmap, hatch = hatch, ax = ax, remove_cbar = True, hatch_pattern = 'XXXXXX', text_size = text_size, plot_mean = plot_mean, offsets = offsets)
                    
                    plot_index = plot_index + 1
        
        # Remove unnecessary axes
        for i in range(1,ncols*nrows - n_maps+1):
            axes.reshape(-1)[-i].axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax = axes, fraction=0.03, pad=0.04)
        cbar.ax.set_ylabel(f'Change in mean wind speed {unit}', rotation=-90, va="bottom")

    def plot_map_median_change_mean_wspd(self, type: str = 'median', plot_range: list = None, cmap: str = 'seismic', relative: bool = False, agreement: float = 80.0, text_size: float = 8):
        """ plot the median (over the ensemble) change of the mean windspeed between historical and future period on a map
        type: either median (default) or mean
        plot_range: optional, to customize the range of the plot
        cmap: the colormap
        relative: if True, relative bias is plotted, if False, bias is plotted
        agreement: percentage of models needed to agree for hatching on plot, if None, no hatching is added
        text_size: size of the text elements on the plot
        """

        # Check if agreement is an acceptable value
        if agreement is not None:
            if (agreement <= 0) | (agreement >= 100): 
                raise ValueError('Agreement parameter should be a strictly positive number between 0 and 100 (not included).')

        # Create empty lists to save changes and agreement on signs in
        diffs = []
        signs = []

        # Loop over the ensembles
        for j in range(len(self.hist)):
            
            ens_hist = self.hist[j]
            ens_fut = self.fut[j]

            # Create empty list to save changes in for this ensemble
            diffs_ens = []

            # Calculate and save the difference of the mean wind speed in the historical and the future period for each ensemble member
            for i in range(len(ens_hist.data_list)):
                fut_i = ens_fut.data_list[i]
                hist_i = ens_hist.data_list[i]
                # fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0
                # and if decrease in the future, we have fut < hist <=> fut - hist < 0
                diff_i = fut_i.data.wind_speed.mean(dim = 'time', dtype = np.float64) - hist_i.data.wind_speed.mean(dim = 'time', dtype = np.float64)
                if relative:
                    diff_i =  diff_i/hist_i.data.wind_speed.mean(dim = 'time', dtype = np.float64)*100 # convert to percentage if needed
                diffs_ens.append(diff_i)
            print('Difference here is fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0 and if decrease in the future, we have fut < hist <=> fut - hist < 0')

            # Get the median (or mean) of the change across the members in the ensemble
            if type == 'median':
                median_change = np.median(diffs_ens, axis = 0)
            elif type == 'mean':
                median_change = np.mean(diffs_ens, axis = 0)

            # Convert median to xarray dataset with lat and lon
            median_xr = diffs_ens[0].copy() # copy the first xarray of the list, does not matter which as they have a common grid
            median_xr.data = median_change

            # Add this ensemble to the full list
            diffs.append(median_xr)

            # Calculate the sign agreement for the members of this ensemble
            sign_ens = np.sum(np.array(diffs_ens) > 0, axis = 0)/np.array(diffs_ens).shape[0]*100 #percentage of models which has a positive sign
            
            # Convert to xr dataset with lat and lon
            sign_ens_xr = diffs_ens[0].copy()
            sign_ens_xr.data = sign_ens

            # Add to list of sign agreements for all ensembles
            signs.append(sign_ens_xr)

        # Calculate the range as min, max of all diffs
        if plot_range is None:
            plot_range = [np.min([np.min(d) for d in diffs]), np.max([np.max(d) for d in diffs])] #set range

            # Symmetrize the range if the min and max have a different sign (or their product is negative)
            if plot_range[0]*plot_range[1]<0:
                abs_max = np.max(np.abs(plot_range))
                plot_range = [-abs_max,abs_max]

        # Construct the unit for the label for the colorbar
        if relative: unit = '(%)'
        else: unit = '(m/s)'

        # Plot a difference map for each ensemble
        for j, diff in enumerate(diffs):
            print(self.hist[j].name)
            #construct hatch !opposite thinking for mask: keeps the values that do not fulfill the requirements
            #both percentages of positive above the agreement threshold and percentages of negative are good, so second condition also needed
            if agreement is None: hatch = None # no hatching is needed in this case
            else:
                hatch = np.array((signs[j].data > agreement) | (signs[j].data < 100 - agreement))
                hatch = np.where(np.isnan(diff.data), np.nan, hatch) # mask nan values from original dataset to avoid hatching outside domain
                hatch = np.where(hatch  == 1, hatch, np.nan) # put nan wherever hatch condition is not fulfilled

            # Plot map or print out difference depending on point or spatial winddataset
            if diff.size == 1:
                print(f"{diff.data}, agreement: {100-signs[j].data if signs[j].data < 50 else signs[j].data}")
            else:
                #plot both the change with the hatching and the model agreement as percentages
                self.plot_map(diff, title = f'{self.hist[j].name}', plot_range = plot_range, hatch_pattern = 'XXXXXX',
                    cmap = cmap, colorbar_label = f'Median change in mean wind speed {unit}', hatch = hatch, text_size = text_size) #hist-fut
                if agreement is not None:
                    cmap = plt.cm.get_cmap('viridis', 10) # discrete colormap
                    self.plot_map(signs[j].where(~np.isnan(diff)), title = 'Model agreement on sign', plot_range = [0,100], 
                        cmap = cmap, colorbar_label=f'Percentage of models predicting increase (%)') # plot of the actual percentages of model agreement
    
    def plot_seasonality(self, alpha: float = 0.2, save: bool = False, uncertainty: str = 'std', color_hist: int = 0, color_fut: int = 1, text_size: float = 8):
        """ Plot a comparison of the seasonality for the future and historical period
        the ensemble mean is plotted, with uncertainty bands ranging from the ensemble min to max  or +- 1 std for each month
        alpha: the alpha for the uncertainty bands
        uncertainty: defines how the uncertainty bands have to be plotted, std mean + and - one standard deviation, minmax means min and maximum
        color_hist: index of the color for the historical curve and band
        color_fut: index of the color for the future curve and band
        text_size: size of the text elements in the plot
        """
        
        # Initialise the colors
        global colors #use the globally defined colors

        # Set the font size 
        plt.rc('font', size = text_size)

        # Initialise lists to save the ensemble mean monthly mean wind speed
        hist_mean = np.zeros(len(self.hist)).tolist()
        fut_mean = np.zeros(len(self.fut)).tolist()

        # Initialise lists to save the min and max of the uncertainty bands
        hist_min = np.zeros(len(self.hist)).tolist()
        fut_min = np.zeros(len(self.fut)).tolist()
        hist_max = np.zeros(len(self.hist)).tolist()
        fut_max = np.zeros(len(self.fut)).tolist()

        # Loop over the ensembles and ensemble members to calculate the ensemble mean of the mean wind speed for each month
        for j in range(len(self.hist)):
            # collect the historical data from the ensemble members and get the mean
            # data is first grouped by month, then monthly means are calculated, then mean over the full domain is taken and wind speed is selected
            hist_data =  [(member_i.data.groupby("time.month").mean(dtype=np.float64)).mean(dim =['lat','lon'], dtype=np.float64).wind_speed.data if type(member_i) == SpatialWindDataSet
                else (member_i.data.groupby("time.month").mean(dtype=np.float64)).wind_speed.data
                for member_i in self.hist[j].data_list]
            hist_mean[j] = np.mean(hist_data,axis = 0)
            
            # collect the future data from the ensemble members and get the mean
            fut_data = [(member_i.data.groupby("time.month").mean(dtype=np.float64)).mean(dim =['lat','lon'], dtype=np.float64).wind_speed.data if type(member_i) == SpatialWindDataSet
                else (member_i.data.groupby("time.month").mean(dtype=np.float64)).wind_speed.data
                for member_i in self.fut[j].data_list]
            fut_mean[j] = np.mean(fut_data,axis = 0)

            # Now set the min and max of the bands, depending on the method
            if uncertainty == 'std':
                # calculate the std across the ensemble members for historical and future data
                std_hist = np.std(hist_data, axis = 0)
                std_fut = np.std(fut_data, axis = 0) 
                # set the band max and min
                hist_min[j] = hist_mean[j]-std_hist
                hist_max[j] = hist_mean[j]+std_hist 
                fut_min[j] = fut_mean[j]-std_fut
                fut_max[j] = fut_mean[j]+std_fut
            elif uncertainty == 'minmax':
                hist_min[j] = np.min(hist_data, axis = 0) 
                hist_max[j] = np.max(hist_data, axis = 0)
                fut_min[j] = np.min(fut_data, axis = 0)
                fut_max[j] = np.max(fut_data, axis = 0)
       
        # Calculate the plot range to use for the y-axis for all plots, list comprehension to loop over all ensembles and all members
        plot_range = [np.min([np.min([hist_min[j],fut_min[j]]) for j in range(len(hist_min))]), 
            np.max([np.max([hist_max[j], fut_max[j]]) for j in range(len(hist_min))])]

        # Loop over the ensembles again, now creating the plots
        for j in range(len(self.hist)):    
            # Create the figure
            fig, ax = plt.subplots(figsize=(8, 4))

            # Plot the historical and future ensemble mean 
            plt.plot(np.arange(1,13), hist_mean[j], label = 'Historical', linewidth = 2, color = colors[color_hist])
            plt.plot(np.arange(1,13), fut_mean[j], label = 'Future', linewidth = 2, color= colors[color_fut]) 

            # Plot the uncertainty band 
            plt.fill_between(np.arange(1,13), hist_min[j], hist_max[j], color = colors[color_hist], alpha = alpha)
            plt.fill_between(np.arange(1,13), fut_min[j], fut_max[j], color = colors[color_fut], alpha = alpha)  
            
            #Add plot details
            plt.title(f'{self.hist[j].name}')
            #plt.xlabel('Month')
            plt.xticks(labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ticks=np.arange(1, 13))
            plt.ylabel('Wind speed (m/s)')
            plt.grid(axis = 'y')
            plt.legend(loc = 'lower left', frameon = False)
            ax.spines[['right', 'top']].set_visible(False) # remove right and top axis

            # Set the scale on the y-axis
            plt.gca().set_ylim(bottom=plot_range[0]-0.5)
            plt.gca().set_ylim(top=plot_range[1]+0.5)

            plt.tight_layout()

            print('Differences in each month (ensemble mean):')
            print('hist:')
            print(hist_mean)
            print('fut')
            print(fut_mean)
            diff_ens_means = (np.array(fut_mean) - np.array(hist_mean))[0]
            print(f"Jan: {diff_ens_means[0]:.2f}, Feb:{diff_ens_means[1]:.2f}, Mar:{diff_ens_means[2]:.2f}, Apr:{diff_ens_means[3]:.2f} May:{diff_ens_means[4]:.2f} Jun:{diff_ens_means[5]:.2f} Jul:{diff_ens_means[6]:.2f} Aug:{diff_ens_means[7]:.2f} Sep:{diff_ens_means[8]:.2f} Oct:{diff_ens_means[9]:.2f} Nov:{diff_ens_means[10]:.2f} Dec:{diff_ens_means[11]:.2f}")
            
            print('Differences in each month (ensemble min):')
            print('hist')
            print(hist_min)
            print('fut')
            print(fut_min)
            diff_ens_min = (np.array(fut_min) - np.array(hist_min))[0]
            print(f"Jan: {diff_ens_min[0]:.2f}, Feb:{diff_ens_min[1]:.2f}, Mar:{diff_ens_min[2]:.2f}, Apr:{diff_ens_min[3]:.2f} May:{diff_ens_min[4]:.2f} Jun:{diff_ens_min[5]:.2f} Jul:{diff_ens_min[6]:.2f} Aug:{diff_ens_min[7]:.2f} Sep:{diff_ens_min[8]:.2f} Oct:{diff_ens_min[9]:.2f} Nov:{diff_ens_min[10]:.2f} Dec:{diff_ens_min[11]:.2f}")
            
            print('Differences in each month (ensemble max):')
            print('hist')
            print(hist_max)
            print('fut')
            print(fut_max)
            diff_ens_max = (np.array(fut_max) - np.array(hist_max))[0]
            print(f"Jan: {diff_ens_max[0]:.2f}, Feb:{diff_ens_max[1]:.2f}, Mar:{diff_ens_max[2]:.2f}, Apr:{diff_ens_max[3]:.2f} May:{diff_ens_max[4]:.2f} Jun:{diff_ens_max[5]:.2f} Jul:{diff_ens_max[6]:.2f} Aug:{diff_ens_max[7]:.2f} Sep:{diff_ens_max[8]:.2f} Oct:{diff_ens_max[9]:.2f} Nov:{diff_ens_max[10]:.2f} Dec:{diff_ens_max[11]:.2f}")
                    
            # Save the plot if needed
            if save:
                plt.savefig(f'{self.hist[j].name}_seasonality.png')

    def compare_densities(self, nbins: int = 100, path_pvalues: str = None, sign_level: float = 0.05, ylim: float = None, nrows: int = None, ncols: int = None, plot_mean: bool = False, 
        color_hist: int = 0, color_fut: int = 1, offsets_x: list = [-2, 0.5], offsets_y: list = [0.01, 0.02]):
        """ Plot a comparison of probability densities between the future and historical simulations 
        nbins: number of bins used to estimate the density curve, for quick testing set this to a low number
        path_pvalues: optional, path to pvalues (actual values not true/false), if the pvalues indicates a significant difference the plot background is set to silver
        sign_level: the significance level to apply when evaluating the pvalues
        ylim: optional, common limit for the y-axis
        nrows: optional, to choose number of rows 
        ncols: optional, to choose number of columns
        plot_mean: optional, to plot a vertical line to indicate the mean
        """
        # Create the grid of axes
        n_ens = len(self.hist)
        n_members = len(self.hist[0].data_list)
        if (nrows is None) & (ncols is None): nrows = n_ens; ncols =  n_members
        elif (nrows is None) | (ncols is None): raise ValueError("Either both ncols and nrows should be given as an argument, or both should be left as None to be chosen automatically.")
        fig, axes = plt.subplots(ncols = ncols, nrows = nrows, sharex = True, sharey = True, figsize = (3.625*ncols, 3.3333*nrows))

        # Load the pvalues
        if path_pvalues is not None:
            pvalues = np.load(path_pvalues)

        # Now plot hist and fut probability curves together for each ens member in the proj
        plot_index = 0 # index of the plot, to enable any shape we want
        for j in range(n_ens):
            for i in range(n_members):
                ax = axes.reshape(-1)[plot_index] # Select the ax in the grid
                # Construct the offsets of the labels (lower value to the left, higher on the right)
                if self.hist[j].data_list[i].data.wind_speed.mean(dtype = np.float64) > self.fut[j].data_list[i].data.wind_speed.mean(dtype = np.float64):
                    offset_hist = np.max(np.array(offsets_x)) 
                    offset_fut = np.min(np.array(offsets_x))
                else:
                    offset_hist = np.min(np.array(offsets_x)) 
                    offset_fut = np.max(np.array(offsets_x)) 
                self.hist[j].data_list[i].plot_density(nbins = nbins, ax = ax, color = colors[color_hist], label = '$\\bf{Historical}$', 
                    plot_mean = plot_mean, offset_y = offsets_y[0], offset_x = offset_hist, text_kwargs = dict(weight = 'bold'))
                self.fut[j].data_list[i].plot_density(nbins = nbins, ax = ax, color = colors[color_fut], label = 'Future', 
                    plot_mean = plot_mean, offset_y = offsets_y[1], offset_x = offset_fut, linestyle = 'dashed')
                if ylim is not None: ax.set_ylim([0,ylim])

                # Remove axes labels, are added globally
                ax.set_xlabel('')
                ax.set_ylabel('')

                # Set a background color if the pvalue is significant
                if path_pvalues is not None:
                    if pvalues[j,i] < sign_level:
                        ax.set_facecolor("silver")
                
                # set a title if the number of columns and rows was custom and doesn't align with number of ens and members
                if nrows != n_ens: ax.set_title(self.hist[j].data_list[i].name)

                plot_index = plot_index + 1

        # Label rows and columns, only if the number of rows and columns are automatically chosen
        if nrows == n_ens:
            if n_ens > 1:
                if n_members == 1:
                    for ax, row in zip(axes, [ens.name for ens in self.hist]): ax.set_ylabel(row, size = 'large', rotation = 90)
                else:
                    for ax, row in zip(axes[:,0], [ens.name for ens in self.hist]): ax.set_ylabel(row, size = 'large', rotation = 90)
            if n_members > 1:
                if n_ens == 1:
                    for ax, col in zip(axes, [mem.name for mem in self.hist[0].data_list]):ax.set_title(col)
                else:
                    for ax, col in zip(axes[0], [mem.name for mem in self.hist[0].data_list]):ax.set_title(col)
                
        # Add full figure x and y labels
        fig.supxlabel('Wind speed (m/s)')
        fig.supylabel('Frequency')

        # Remove unnecessary axes, which can happen for custom nrows and ncol
        for i in range(1,nrows*ncols - n_ens*n_members +1):
            axes.reshape(-1)[-i].axis('off')

        plt.tight_layout()

    def calculate_variability_change(self, metric: str, period: str, wind_speed: bool = True):
        """ Calculate the change in variability between the historical and future period
        metric: the metric to calculate: std or rcov
        period: over which period the metric should be calculated, options: year, month, season, none (no averaging is done)
        """

        # Initialise empty list to save the changes
        changes = []

        # Loop over the list of historical and future ensembles, and calculate the differences between 
        # the historical and future period variabilities
        for i in range(len(self.hist)):
            var_hist = self._variability_one_ensemble(metric = metric, ensemble = self.hist[i], period = period, relative = False, wind_speed = wind_speed)
            var_fut = self._variability_one_ensemble(metric, self.fut[i], period, relative = False, wind_speed = wind_speed)
            change_i = np.array(var_fut)-np.array(var_hist) #convert to np arrays to compute difference
            changes.append(change_i) # add to list

        return changes

    def plot_variability(self, metric: str, periods: list, wind_speed: bool = True, means: bool = False, save = False, ax = None,  **kwargs):
        """ Plot the differences of the variabilities for the future and historical period
        metric: the metric to plot, std or rcov
        periods: list of periods to calculate the variability for
        means: should the ensemble means be added to the tables?
        ax: optional, ax to add the plot to
        save: should the plot be saved?
        """

        # Initialise empty array to save the changes in variability
        changes = []
        
        # Compute the variability metric for each period, and add it to values
        for p in periods:
            change = self.calculate_variability_change(metric, p, wind_speed)
            changes.append(change)

        # Set the plot_range
        plot_range = [np.min(changes), np.max(changes)]

        # Construct the label of the color bar
        label = 'Change ' + self._construct_label(wind_speed = wind_speed, relative = False, metric = metric)
        print('Difference here is fut - hist since if increase in the future, we have fut > hist <=> fut - hist > 0 and if decrease in the future, we have fut < hist <=> fut - hist < 0')

        # Loop over the historical ensembles to make the plots
        for i in range(len(self.hist)):

            # Get the historical ensemble from the list (just for the model name, so does not matter if hist or fut is used)
            ens = self.hist[i]

            # Initialise an empty array to save the variability metrics for different periods for this ensemble
            change_ens =[]

            # Add each period variability for this ensemble to std_ens
            for p in range(len(periods)):
                var_p_i = changes[p][i] #index p selects the period, index i selects the ensemble
                if means:
                    var_p_i= np.concatenate([var_p_i, [np.mean(var_p_i)]]) # add the ensemble mean for this metric if needed
                change_ens.append(var_p_i) # add this metric to the values for this ensemble

            # Construct the rownames as the names of the ensemble members
            member_names = [winddataset.name for winddataset in ens.data_list]

            # Add the ensemble mean to the names if needed
            if means:
                member_names.append('$\\bf{Ensemble\ mean}$') # add the ensemble mean

            # Save the plot if needed by setting the filename to hand
            if save:
                filename = f'{ens.name}_table_variability_change'
            else: filename = None

            # Plot the table
            colnames = [self.hist[0].data_list[0].data.frequency.capitalize() if p.lower() == 'none' else p for p in periods] # Assumes daily data
            self.plot_table(change_ens, colnames = colnames, rownames = member_names, plot_range = plot_range, title = ens.name, ax = ax,  
                colorbar_label= label, filename = filename, **kwargs)