# Class to calculate wind power from wind speed data using a power curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

class PowerCalculator():
    """ Class to calculate wind power from wind speed data, in a basic way using a power curve
    so wakes, blockage, ... are not taken into account
    """

    def __init__(self, wind_turbine: str):
        """ Initialise a PowerCalculator with a power curve and a rated wind speed linked to the name of the wind turbine 
        wind_turbine: name of the wind turbine to use, the rated power and path of the csv with the power curve should be hardcoded here
        """
        match wind_turbine:
            case "IEA10MW":
                self.name = 'IEA10MW'
                self.rated_speed = 11 #m/s
                self.cut_in  = 4
                self.cut_out = 25
                self.power_curve = pd.read_csv('.../IEA_Reference_10MW_198.csv') # TODO ADAPT PATH
            case "IEA15MW":
                self.name = 'IEA15MW'
                self.rated_speed = 10.6 #m/s
                self.cut_in = 3
                self.cut_out = 25
                self.power_curve = pd.read_csv('.../IEA_Reference_15MW_240.csv') # TODO ADAPT PATH
            case _:
                raise ValueError(f'Wind turbine {wind_turbine}  not found. Either it still has to be implemented or it\'s a typo')
        
        # make sure these calculation are only executed once by saving them in the object
        self.rated_power = self.get_rated_power()
        self.speed_to_power = self.estimate_below_rated()
        print(self)

    def __repr__(self):
        """ Representation of the PowerCalculator for printing
        """
        return f'Powercalculator for turbine {self.name}:\ncut-in wind speed {self.cut_in} m/s\ncut-out wind speed {self.cut_out} m/s\nrated wind speed {self.rated_speed} m/s'

    def get_rated_power(self):
        """ Get the rated power from the csv file containing the power curve"""
        closest = (self.power_curve['Wind Speed [m/s]']-self.rated_speed).abs().argsort()[0]
        return self.power_curve.iloc[closest]['Power [kW]'].item()

    def estimate_below_rated(self):
        """ Using the data in the power curve csv, interpolate a wind speed to power relation below the rated wind speed 
        """
        # Create a mask to only consider values below rated wind speeds
        below_rated = self.power_curve['Wind Speed [m/s]'] < self.rated_speed

        # Perform cubic interpolation from wind speed to power for the below rated conditions
        speed_to_power = np.poly1d(np.polyfit(self.power_curve['Wind Speed [m/s]'][below_rated], self.power_curve['Power [kW]'][below_rated], deg = 3))

        return speed_to_power

    def calculate_power_single(self, wind_speed: float):
        """ Given a wind speed, estimate the power  the power that would be produced by this wind turbine from the power curve
        wind_speed: the wind speed to calculate the power from
        """
        # get the power depending on the regime we are in 
        if np.isnan(wind_speed):
            return np.nan
        elif wind_speed < self.cut_in or wind_speed > self.cut_out: 
            return 0
        elif wind_speed >= self.rated_speed: 
            return self.rated_power
        else: 
            power = self.speed_to_power(wind_speed)
            # little check to avoid effect of interpolation at the edge, which otherwise can cause capacity factors of > 100%
            if power < self.rated_power: return power
            else: return self.rated_power

    def calculate_power(self, wind_speeds: np.array):
        """ Given an array of wind speeds, calculate the power that would be produced by this wind turbine
        wind_speeds: the numpy array of wind speeds
        """
        calculate_power_array = np.vectorize(self.calculate_power_single, otypes=[float]) # vectorize the previous function to work on arrays, specify the type to avoid it thinking it's int from a few calls
        return calculate_power_array(wind_speeds)

    def kW_to_MW(self, powers: np.array):
        """Given an array of powers in kW, convert to MW """
        return powers/1000

    def plot_power_curve(self, ax = None, color: str = None, capacity_factor: bool = False, add_points: bool = False, MW: bool = False, figsize = (6,6)):
        """ Plot the both the discrete power curve points from the csv and the power calculated here
        ax: optional, plot to add this power curve to
        capacity_factor: set this to true to plot the capacity factor instead of the power, so divide by rated powe
        add_points: if true, empirical points from which the curve is derived are also plotted 
        """

        # Create a figure if none is given
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('Wind Speed (m/s)')
            if capacity_factor:
                ax.set_title(f'Normalised power curve (capacity factor)')
                ax.set_ylabel('Capacity factor (%)')
            else:
                ax.set_title(f'Power curve')
                if MW: ax.set_ylabel('Power (MW)')
                else: ax.set_ylabel('Power (kW)')

        # Actually create the plot
        speeds = np.linspace(self.cut_in-1, self.cut_out +1, 70) # make up some wind speeds to plot
        if color is None: color = 'green'
        if capacity_factor:
            plt.plot(speeds, self.calculate_capacity_factor(speeds), c =color, label = self.name)
        else:
            if MW: powers = self.kW_to_MW(self.calculate_power(speeds))
            else: powers = self.calculate_power(speeds)
            plt.plot(speeds, powers, c =color, label = self.name)
            if add_points:
                if MW: powers = self.kW_to_MW(self.power_curve['Power [kW]'])
                else: powers = self.power_curve['Power [kW]']
                plt.scatter(x = self.power_curve['Wind Speed [m/s]'], y = powers, c =color)
        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)

        return ax 

    def calculate_capacity_factor(self, wind_speeds: np.array):
        """ Given an array of wind speeds, calculate the power that would be produced by this wind turbine
        wind_speeds: the numpy array of wind speeds
        """
        # First calculate the power from the given wind speeds
        powers = self.calculate_power(wind_speeds)

        # Create a function to calculate the capacity factor and vectorize it
        power_to_cf = lambda power: power/self.rated_power*100
        power_to_cf_array = np.vectorize(power_to_cf, otypes=[float])

        # Apply this function to the powers calculated earlier
        return power_to_cf_array(powers)

    def winddataset_to_power(self, ds):
        """ Given a winddataset of wind speeds, convert it to a winddataset of power
        ds: the winddataset
        """
        # Copy the dataset, put the power in place and clean it up
        power = copy.deepcopy(ds)
        power.data = power.data.drop_vars(['wind_direction'], errors = 'ignore') # if wind direction is not present, don't raise an error
        power.data.wind_speed.data = self.calculate_power(ds.data.wind_speed) 
        power.data = power.data.rename_vars({'wind_speed':'power'})
        power.data.power.attrs['long_name'] = 'power'
        power.data.power.attrs['units'] = 'kW'

        return power

    def winddataset_to_capacity_factor(self, ds):
        """ Given a winddataset of wind speeds, convert it to a winddataset of power
        ds: the winddataset
        """
        # Copy the dataset, put the power in place and clean it up
        power = copy.deepcopy(ds)
        power.data = power.data.drop_vars(['wind_direction'], errors = 'ignore')
        power.data.wind_speed.data = self.calculate_capacity_factor(ds.data.wind_speed) 
        power.data = power.data.rename_vars({'wind_speed':'capacity_factor'})
        power.data.capacity_factor.attrs['long_name'] = 'capacity factor'
        power.data.capacity_factor.attrs['units'] = '%'

        return power

    def ensemble_to_power(self, ens):
        """ Given an ensemble of wind speeds, convert it to an ensemble of power
        ens: the ensemble
        """
        # Copy the ensemble and convert the members and reference to power
        output = copy.deepcopy(ens)
        output.data_list = [self.winddataset_to_power(ds) for ds in output.data_list]
        if ens.ref is not None:
            output.ref = self.winddataset_to_power(output.ref)

        return output

    def ensemble_to_capacity_factor(self, ens):
        """ Given an ensemble of wind speeds, convert it to an ensemble of capacity factor
        ens: the ensemble
        """
        # Copy the ensemble and convert the members and reference to power
        output = copy.deepcopy(ens)
        output.data_list = [self.winddataset_to_capacity_factor(ds) for ds in output.data_list]
        if ens.ref is not None:
            output.ref = self.winddataset_to_capacity_factor(output.ref)

        return output