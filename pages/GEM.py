import streamlit as st
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import requests
import pandas as pd
import json
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from io import BytesIO
import base64


# Set matplotlib to use a simple backend
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better compatibility


class GEMModelForecast:
    """A class to handle weather forecast data with GEM model visualization"""
    
    def __init__(self, data_dir="gem_data"):
        self.data_dir = data_dir
        self.dataset = None
        self.ensemble_datasets = []
        self.api_base_url = "https://api.open-meteo.com/v1/forecast"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_forecast(self, latitude, longitude, variable=None, forecast_days=7):

        # Map the GEM variables to Open-Meteo variables
        variable_mapping = {
            "TMP": "temperature_2m",
            "PRES": "surface_pressure",
            "WIND": "wind_speed_10m",
            "PCPN": "precipitation",
            None: None  # Handle the case where variable is None
        }
        
        # Default weather variables to request
        weather_vars = [
            "temperature_2m", 
            "surface_pressure", 
            "wind_speed_10m", 
            "precipitation",
            "relative_humidity_2m"
        ]
        
        # If a specific variable is requested, focus on that
        open_meteo_var = variable_mapping.get(variable)
        if open_meteo_var:
            weather_vars = [open_meteo_var]
        
        # Format date and create filename
        now = datetime.now()  
        date_str = now.strftime("%Y%m%d%H")
        
        # after you compute date_str
        var_tag = variable if variable else "all"
        filename = f"Open-Meteo_forecast_{latitude}_{longitude}_{var_tag}_{date_str}.nc"
        filepath = os.path.join(self.data_dir, filename)

        # Check if recent data already exists (less than 3 hours old)
        if os.path.exists(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (now - file_time).total_seconds() < 10800:  # 3 hours in seconds
                st.write(f"Using recent data file: {filepath}")
                return filepath
        
        st.write(f"Downloading forecast data for coordinates: {latitude}, {longitude}")
        
        # Build API request parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(weather_vars),
            "forecast_days": forecast_days,
            "timezone": "auto"
        }
        
        try:
            # Make API request
            response = requests.get(self.api_base_url, params=params)
            response.raise_for_status()  # Raise exception for bad responses
            data = response.json()
            
            # Convert to xarray dataset
            ds = self._convert_json_to_xarray(data)
            
            # Save to netCDF
            ds.to_netcdf(filepath)
            st.write(f"Downloaded and saved forecast data to: {filepath}")
            
            return filepath
            
        except Exception as e:
            st.error(f"Error downloading forecast data: {e}")
            # If download fails, create synthetic data as fallback
            st.write("Falling back to synthetic data generation...")
            self._create_synthetic_data(filepath, variable if variable else "TMP", now, 0)
            return filepath
    
    def _convert_json_to_xarray(self, json_data):
        """Convert Open-Meteo JSON response to xarray Dataset
        
        Args:
            json_data (dict): Open-Meteo API response
            
        Returns:
            xarray.Dataset: Dataset containing the forecast data
        """
        # Extract hourly data
        hourly_data = json_data.get("hourly", {})
        
        # Get time values
        time_strings = hourly_data.get("time", [])
        times = np.array([np.datetime64(ts) for ts in time_strings])
        
        # Create data dictionary for xarray
        data_dict = {}
        attrs_dict = {}
        
        # Variable mappings and metadata
        var_metadata = {
            "temperature_2m": {"long_name": "Air Temperature at 2m", "units": "°C", "standard_name": "TMP"},
            "surface_pressure": {"long_name": "Surface Pressure", "units": "hPa", "standard_name": "PRES"},
            "wind_speed_10m": {"long_name": "Wind Speed at 10m", "units": "m/s", "standard_name": "WIND"},
            "precipitation": {"long_name": "Total Precipitation", "units": "mm", "standard_name": "PCPN"},
            "relative_humidity_2m": {"long_name": "Relative Humidity at 2m", "units": "%", "standard_name": "RH"}
        }
        
        # Add data variables from hourly data
        for var_name, values in hourly_data.items():
            if var_name != "time":  # Skip time as it's used for coordinates
                # Convert to numeric values
                numeric_values = np.array(values, dtype=float)
                
                # Add to data dictionary with standard name if available
                if var_name in var_metadata:
                    standard_name = var_metadata[var_name]["standard_name"]
                    data_dict[standard_name] = (["time"], numeric_values)
                    attrs_dict[standard_name] = var_metadata[var_name]
                else:
                    data_dict[var_name] = (["time"], numeric_values)
                    attrs_dict[var_name] = {"long_name": var_name, "units": "unknown"}
        
        # Create a dataset with proper coordinates and dimensions
        ds = xr.Dataset(
            data_vars=data_dict,
            coords={"time": times, 
                    "lat": json_data.get("latitude", 0),
                    "lon": json_data.get("longitude", 0)}
        )
        
        # Add attributes
        for var_name, attrs in attrs_dict.items():
            for attr_name, attr_value in attrs.items():
                ds[var_name].attrs[attr_name] = attr_value
        
        # Add global attributes
        ds.attrs["Conventions"] = "CF-1.6"
        ds.attrs["title"] = "Weather Forecast"
        ds.attrs["institution"] = "Open-Meteo API"
        ds.attrs["source"] = "Open-Meteo Weather Forecast API"
        ds.attrs["forecast_days"] = json_data.get("forecast_days", 7)
        
        return ds
    
    def _create_synthetic_data(self, filepath, variable, forecast_date, forecast_hour):
        """Create synthetic netCDF data as fallback when API fails"""
        st.write("Creating synthetic data for demonstration...")
        
        # Define time dimension
        now = datetime.datetime.now()
        times = pd.date_range(start=now, periods=168, freq='1H')  # 7 days hourly
        time_values = np.array(times, dtype='datetime64[ns]')
        
        # Generate synthetic data based on variable type
        if variable == "TMP":  # Temperature
            # Create temperature field with realistic patterns
            # Generate temperatures with daily cycle - warmer during day, cooler at night
            base_temp = 15  # Base temperature
            day_cycle = 10 * np.sin(np.pi * (np.arange(168) % 24) / 12)  # Daily cycle
            trend = np.linspace(-1, 1, 168)  # Slight trend over the week
            data = base_temp + day_cycle + trend
            # Add some noise
            data += np.random.normal(0, 1, size=data.shape)
            long_name = "Air Temperature at 2m"
            units = "°C"
            
        elif variable == "PRES":  # Pressure
            # Create pressure field with realistic patterns
            base_pressure = 1013  # Base pressure in hPa
            # Pressure varies slightly over time
            pressure_trend = 10 * np.sin(np.pi * np.arange(168) / 84)
            data = base_pressure + pressure_trend
            # Add some random variations
            data += np.random.normal(0, 2, size=data.shape)
            long_name = "Surface Pressure"
            units = "hPa"
            
        elif variable == "WIND":  # Wind speed
            # Wind speed with daily pattern
            base_wind = 5  # Base wind speed
            daily_pattern = 3 * np.sin(np.pi * (np.arange(168) % 24) / 8)
            data = base_wind + daily_pattern
            # Add some randomness
            data += np.random.normal(0, 1, size=data.shape)
            data = np.abs(data)  # Make sure wind speeds are positive
            long_name = "Wind Speed at 10m"
            units = "m/s"
            
        elif variable == "PCPN":  # Precipitation
            # Create precipitation data - mostly zeros with occasional rain
            data = np.zeros(168)
            # Add some random precipitation events
            rain_periods = np.random.randint(0, 168, size=5)  # 5 random rain periods
            for period in rain_periods:
                duration = np.random.randint(1, 6)  # Duration of 1-5 hours
                intensity = np.random.uniform(0.5, 5)  # Rain intensity
                for i in range(duration):
                    if period + i < 168:
                        data[period + i] = intensity * (1 - i/duration)  # Decreasing intensity
            long_name = "Total Precipitation"
            units = "mm"
        
        else:
            # Default random data
            data = np.random.normal(0, 1, size=168)
            long_name = f"Variable {variable}"
            units = "unknown"
        
        # Create a dataset
        ds = xr.Dataset(
            data_vars={
                variable: (["time"], data),
            },
            coords={
                "time": time_values,
                "lat": 90.0,  # Default latitude
                "lon": -100.0   # Default longitude
            }
        )
        
        # Add attributes
        ds[variable].attrs["long_name"] = long_name
        ds[variable].attrs["units"] = units
        ds.attrs["Conventions"] = "CF-1.6"
        ds.attrs["title"] = f"Synthetic Forecast - {long_name}"
        ds.attrs["institution"] = "Synthetic Data Generator"
        ds.attrs["source"] = "Synthetic Weather Model"
        
        # Save to netCDF
        ds.to_netcdf(filepath)
        st.write(f"Created synthetic data file: {filepath}")
    
    def load_forecast(self, filepath):
        """Load a forecast file
        
        Args:
            filepath (str): Path to netCDF file
            
        Returns:
            xarray.Dataset: Loaded dataset
        """
        try:
            self.dataset = xr.open_dataset(filepath)
            st.write(f"Loaded dataset: {filepath}")
            st.write(f"Variables: {list(self.dataset.data_vars)}")
            return self.dataset
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    
    def generate_ensemble(self, latitude, longitude, variable=None, n_members=10):
        """Generate an ensemble of forecasts by perturbing the base forecast
        
        Args:
            latitude (float): Latitude of the location
            longitude (float): Longitude of the location
            variable (str): Weather variable for ensemble
            n_members (int): Number of ensemble members
            
        Returns:
            list: List of ensemble member datasets
        """
        st.write(f"Generating {n_members} ensemble members...")
        self.ensemble_datasets = []
        
        # Download base forecast if needed
        base_file = self.download_forecast(latitude, longitude, variable)
        base_ds = xr.open_dataset(base_file)
        
        # Determine which variable to create ensemble for if not specified
        if variable is None or variable not in base_ds.data_vars:
            if "TMP" in base_ds.data_vars:
                variable = "TMP"
            else:
                variable = list(base_ds.data_vars)[0]
        
        # Create ensemble by adding perturbations
        for i in range(n_members):
            # Create a copy of the dataset
            ens_ds = base_ds.copy(deep=True)
            
            # Get the variable data
            var_data = ens_ds[variable].values
            
            # Add random perturbations (scaled appropriately for the variable)
            if variable == "TMP":
                # Temperature perturbations (°C)
                perturbation = np.random.normal(0, 0.5, size=var_data.shape)
            elif variable == "PRES":
                # Pressure perturbations (hPa)
                perturbation = np.random.normal(0, 1.0, size=var_data.shape)
            elif variable == "WIND":
                # Wind speed perturbations (m/s)
                perturbation = np.random.normal(0, 0.8, size=var_data.shape)
            elif variable == "PCPN":
                # Precipitation perturbations (mm)
                perturbation = np.random.normal(0, 0.5, size=var_data.shape)
                # Ensure non-negative precipitation
                var_data = np.maximum(0, var_data + perturbation)
                ens_ds[variable].values = var_data
                self.ensemble_datasets.append(ens_ds)
                continue
            else:
                # Default perturbation
                perturbation = np.random.normal(0, 0.05 * np.std(var_data), size=var_data.shape)
            
            # Apply perturbation
            ens_ds[variable].values = var_data + perturbation
            
            # Add to ensemble list
            self.ensemble_datasets.append(ens_ds)
            
        st.write(f"Generated {len(self.ensemble_datasets)} ensemble members")
        return self.ensemble_datasets
    
    def calculate_ensemble_statistics(self, variable):
        """Calculate ensemble statistics for the given variable
        
        Args:
            variable (str): Variable name to calculate statistics for
            
        Returns:
            dict: Dictionary containing ensemble mean, spread, min, max
        """
        if not self.ensemble_datasets:
            st.error("No ensemble data available. Generate ensemble first.")
            return None
        
        # Stack all ensemble member data
        all_data = np.stack([ds[variable].values for ds in self.ensemble_datasets])
        
        # Calculate statistics
        ensemble_mean = np.mean(all_data, axis=0)
        ensemble_spread = np.std(all_data, axis=0)
        ensemble_min = np.min(all_data, axis=0)
        ensemble_max = np.max(all_data, axis=0)
        
        # Create a new dataset with the statistics
        stats_ds = self.ensemble_datasets[0].copy(deep=True)
        stats_ds[f"{variable}_mean"] = (stats_ds[variable].dims, ensemble_mean)
        stats_ds[f"{variable}_spread"] = (stats_ds[variable].dims, ensemble_spread)
        stats_ds[f"{variable}_min"] = (stats_ds[variable].dims, ensemble_min)
        stats_ds[f"{variable}_max"] = (stats_ds[variable].dims, ensemble_max)
        
        # Add attributes
        stats_ds[f"{variable}_mean"].attrs = stats_ds[variable].attrs.copy()
        stats_ds[f"{variable}_mean"].attrs["long_name"] = f"Ensemble Mean of {stats_ds[variable].attrs.get('long_name', variable)}"
        
        stats_ds[f"{variable}_spread"].attrs = stats_ds[variable].attrs.copy()
        stats_ds[f"{variable}_spread"].attrs["long_name"] = f"Ensemble Spread of {stats_ds[variable].attrs.get('long_name', variable)}"
        stats_ds[f"{variable}_spread"].attrs["units"] = stats_ds[variable].attrs.get("units", "unknown")
        
        stats = {
            "mean": ensemble_mean,
            "spread": ensemble_spread,
            "min": ensemble_min,
            "max": ensemble_max,
            "dataset": stats_ds
        }
        
        return stats
    
    def plot_forecast(self, variable=None, ax=None, cmap=None, title=None):
        """Plot the forecast data as a time series
        
        Args:
            variable (str): Variable to plot (must be in dataset)
            ax (matplotlib.axes): Existing axes to plot on
            cmap (str): Colormap name
            title (str): Custom title for the plot
            
        Returns:
            matplotlib.figure.Figure: Figure containing the plot
        """
        if self.dataset is None:
            st.error("No dataset loaded. Load a forecast first.")
            return None
        
        # If variable not specified, use the first data variable
        if variable is None:
            variable = list(self.dataset.data_vars)[0]
        
        if variable not in self.dataset.data_vars:
            st.error(f"Variable {variable} not found in dataset.")
            st.write(f"Available variables: {list(self.dataset.data_vars)}")
            return None
        
        # Create figure and axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()
        
        # Get the data
        da = self.dataset[variable]
        
        # Get variable metadata
        long_name = da.attrs.get('long_name', variable)
        units = da.attrs.get('units', '')
        
        # Choose color based on variable
        color_mapping = {
            "TMP": "tab:red",
            "PRES": "tab:purple",
            "WIND": "tab:blue",
            "PCPN": "tab:green"
        }
        color = color_mapping.get(variable, "tab:blue")
        
        # Plot the time series
        try:
            # Handle time coordinate
            if 'time' in da.coords:
                x = da.time.values
                y = da.values
                ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=3)
                
                # Format x-axis as dates
                fig.autofmt_xdate()
                
            else:
                # If there's no time coordinate, use a simple index
                ax.plot(da.values, color=color, linewidth=2, marker='o', markersize=3)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add labels
            ax.set_xlabel('Time')
            ax.set_ylabel(f"{long_name} ({units})")
            
            # Set title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Weather Forecast: {long_name}")
            
            # Add light grey shading for every other day to distinguish days
            if 'time' in da.coords:
                times = pd.to_datetime(da.time.values)
                days = (times - times[0]).days
                unique_days = np.unique(days)
                
                for day in unique_days[::2]:  # Every other day
                    day_start = times[days == day].min()
                    day_end = times[days == day].max()
                    ax.axvspan(day_start, day_end, alpha=0.1, color='gray')
            
            return fig
        except Exception as e:
            st.error(f"Error plotting data: {e}")
            return None
    
    def plot_multiday_forecast(self):
        """Plot multiple variables for a multi-day forecast"""
        if self.dataset is None:
            st.error("No dataset loaded. Load a forecast first.")
            return None
            
        try:
            # Always plot these 4 variables if they exist
            variables = ["TMP", "PRES", "WIND", "PCPN"]
            existing_vars = [var for var in variables if var in self.dataset.data_vars]
            
            # Create subplots
            fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
            colors = ["tab:red", "tab:purple", "tab:blue", "tab:green"]
            
            # Plot each variable
            for i, var in enumerate(variables):
                if var in self.dataset.data_vars:
                    da = self.dataset[var]
                    axs[i].plot(da.time.values, da.values, color=colors[i], linewidth=2)
                    axs[i].set_ylabel(f"{da.attrs.get('long_name', var)} ({da.attrs.get('units', '')})")
                    axs[i].grid(True, linestyle='--', alpha=0.6)
                    if var == "PCPN":
                        axs[i].set_ylim(0, max(1, da.max().item() * 1.1))
                else:
                    # Hide axes for missing variables
                    axs[i].axis('off')

            plt.tight_layout()
            fig.suptitle("Multi-Day Weather Forecast", fontsize=16)
            return fig
            
        except Exception as e:
            st.error(f"Error plotting multi-day forecast: {e}")
            return None
    
    def plot_ensemble_forecast(self, variable):
        """Plot ensemble forecast time series with statistics
        
        Args:
            variable (str): Variable to plot
            
        Returns:
            matplotlib.figure.Figure: Figure containing the plot
        """
        if not self.ensemble_datasets:
            st.error("No ensemble data available. Generate ensemble first.")
            return None
        
        # Calculate ensemble statistics
        stats = self.calculate_ensemble_statistics(variable)
        
        if stats is None:
            return None
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get reference dataset for coordinates
            ref_ds = self.ensemble_datasets[0]
            times = ref_ds.time.values
            
            # Get variable metadata
            long_name = ref_ds[variable].attrs.get('long_name', variable)
            units = ref_ds[variable].attrs.get('units', '')
            
            # Plot each ensemble member as a light line
            for ds in self.ensemble_datasets:
                ax.plot(times, ds[variable].values, color='lightgray', linewidth=0.5, alpha=0.3)
            
            # Plot ensemble statistics
            ax.plot(times, stats["mean"], 'b-', linewidth=2, label='Ensemble Mean')
            
            # Plot ensemble spread (mean +/- 1 standard deviation)
            ax.fill_between(
                times, 
                stats["mean"] - stats["spread"], 
                stats["mean"] + stats["spread"],
                color='blue', alpha=0.2, label='Ensemble Spread (±1σ)'
            )
            
            # Plot ensemble min/max
            ax.plot(times, stats["min"], 'g--', linewidth=1, alpha=0.7, label='Ensemble Min')
            ax.plot(times, stats["max"], 'r--', linewidth=1, alpha=0.7, label='Ensemble Max')
            
            # Format x-axis as dates
            fig.autofmt_xdate()
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')
            
            # Add labels
            ax.set_xlabel('Time')
            ax.set_ylabel(f"{long_name} ({units})")
            
            # Set title
            ax.set_title(f"Ensemble Forecast: {long_name}")
            
            # Add light grey shading for every other day to distinguish days
            times_pd = pd.to_datetime(times)
            days = (times_pd - times_pd[0]).days
            unique_days = np.unique(days)
            
            for day in unique_days[::2]:  # Every other day
                day_start = times_pd[days == day].min()
                day_end = times_pd[days == day].max()
                ax.axvspan(day_start, day_end, alpha=0.1, color='gray')
            
            return fig
        except Exception as e:
            st.error(f"Error plotting ensemble forecast: {e}")
            return None
    
    def calculate_probability(self, variable, threshold, comparison="gt"):
        """Calculate probability of exceeding/falling below a threshold from ensemble
        
        Args:
            variable (str): Variable name
            threshold (float): Threshold value
            comparison (str): Comparison type ('gt' for greater than, 'lt' for less than)
            
        Returns:
            numpy.ndarray: Probability field (0-100%)
        """
        if not self.ensemble_datasets:
            st.error("No ensemble data available. Generate ensemble first.")
            return None
        
        # Stack all ensemble member data
        all_data = np.stack([ds[variable].values for ds in self.ensemble_datasets])
        n_members = len(self.ensemble_datasets)
        
        # Calculate exceedance probability
        if comparison == "gt":
            # Greater than threshold
            exceed_count = np.sum(all_data > threshold, axis=0)
        elif comparison == "lt":
            # Less than threshold
            exceed_count = np.sum(all_data < threshold, axis=0)
        else:
            st.error(f"Invalid comparison type: {comparison}. Use 'gt' or 'lt'.")
            return None
        
        # Convert to probability (%)
        probability = 100 * exceed_count / n_members
        
        return probability
    
    def plot_probability(self, variable, threshold, comparison="gt"):

        # Calculate probability
        probability = self.calculate_probability(variable, threshold, comparison)
        

        if probability is None:
            return None
        
        try:
            # Get reference dataset for coordinates
            ref_ds = self.ensemble_datasets[0]
            times = ref_ds.time.values
            
            # Get variable metadata
            da = ref_ds[variable]
            long_name = da.attrs.get('long_name', variable)
            units = da.attrs.get('units', '')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define comparison text
            comp_text = "exceeding" if comparison == "gt" else "below"
            
            # Plot probability time series
            ax.plot(times, probability, 'r-', linewidth=2)
            
            # Add shading based on probability values
            # Red for high probability, blue for low probability
            ax.fill_between(times, 0, probability, color='tomato', alpha=0.3)
            
            # Add reference lines
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% probability')
            ax.axhline(y=25, color='blue', linestyle=':', alpha=0.5, label='25% probability')
            ax.axhline(y=75, color='red', linestyle=':', alpha=0.5, label='75% probability')
            
            # Format x-axis as dates
            fig.autofmt_xdate()
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')
            
            # Add labels
            ax.set_xlabel('Time')
            ax.set_ylabel('Probability (%)')
            ax.set_ylim(0, 100)
            
            # Title
            ax.set_title(f"Probability of {long_name} {comp_text} {threshold} {units}")
            
            # Add light grey shading for every other day to distinguish days
            times_pd = pd.to_datetime(times)
            days = (times_pd - times_pd[0]).days
            unique_days = np.unique(days)
            
            for day in unique_days[::2]:  # Every other day
                day_start = times_pd[days == day].min()
                day_end = times_pd[days == day].max()
                ax.axvspan(day_start, day_end, alpha=0.1, color='gray')
            
            return fig
        except Exception as e:
            st.error(f"Error plotting probability: {e}")
            return None



    def get_forecast_summary(self, variable=None):
        """Generate a text summary of the forecast
        
        Args:
            variable (str): Variable to summarize (defaults to temperature)
            
        Returns:
            str: Text summary of forecast
        """
        if self.dataset is None:
            return "No forecast data loaded."
        
        # Default to temperature if variable not specified
        if variable is None:
            if "TMP" in self.dataset.data_vars:
                variable = "TMP"
            else:
                variable = list(self.dataset.data_vars)[0]
        
        if variable not in self.dataset.data_vars:
            return f"Variable {variable} not found in forecast data."
        
        try:
            # Get the data and metadata
            da = self.dataset[variable]
            long_name = da.attrs.get('long_name', variable)
            units = da.attrs.get('units', '')
            
            # Convert time to pandas DatetimeIndex for easier handling
            if 'time' in da.coords:
                times = pd.to_datetime(da.time.values)
                values = da.values
                
                # Group by day
                day_groups = {}
                for i, t in enumerate(times):
                    day = t.strftime('%Y-%m-%d')
                    if day not in day_groups:
                        day_groups[day] = []
                    day_groups[day].append(values[i])
                
                # Create summary for each day
                summary = f"Forecast Summary for {long_name}:\n\n"
                
                for day, vals in day_groups.items():
                    day_date = pd.to_datetime(day)
                    day_name = day_date.strftime('%A')  # Day of week
                    
                    min_val = min(vals)
                    max_val = max(vals)
                    mean_val = sum(vals) / len(vals)
                    
                    # Format the date nicely
                    formatted_date = day_date.strftime('%B %d, %Y')
                    
                    summary += f"{day_name}, {formatted_date}:\n"
                    summary += f"  Min: {min_val:.1f} {units}\n"
                    summary += f"  Max: {max_val:.1f} {units}\n"
                    summary += f"  Avg: {mean_val:.1f} {units}\n"
                    
                    # Add specific interpretations based on variable
                    if variable == "TMP":
                        if max_val > 30:
                            summary += "  Very hot conditions expected.\n"
                        elif max_val > 25:
                            summary += "  Warm conditions expected.\n"
                        elif max_val < 0:
                            summary += "  Freezing conditions expected.\n"
                        elif max_val < 10:
                            summary += "  Cool conditions expected.\n"
                    elif variable == "PCPN":
                        if max(vals) > 0:
                            total_precip = sum(vals)
                            if total_precip > 10:
                                summary += f"  Heavy precipitation expected (total: {total_precip:.1f} mm).\n"
                            else:
                                summary += f"  Light precipitation expected (total: {total_precip:.1f} mm).\n"
                    elif variable == "WIND":
                        if max(vals) > 10:
                            summary += "  Windy conditions expected.\n"
                        elif max(vals) > 5:
                            summary += "  Breezy conditions expected.\n"
                    
                    summary += "\n"
                
                return summary
            else:
                # If no time coordinate, just return summary stats
                min_val = float(da.min())
                max_val = float(da.max())
                mean_val = float(da.mean())
                
                summary = f"Forecast Summary for {long_name}:\n"
                summary += f"  Min: {min_val:.1f} {units}\n"
                summary += f"  Max: {max_val:.1f} {units}\n"
                summary += f"  Avg: {mean_val:.1f} {units}\n"
                
                return summary
                
        except Exception as e:
            return f"Error generating forecast summary: {e}"

    def get_image_download_link(self, fig, filename, text):
        """Generates a download link for a matplotlib figure"""
        try:
            # Save figure to a BytesIO buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Encode the image buffer to base64
            img_str = base64.b64encode(buf.read()).decode()
            
            # Create a download link
            href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
            return href
        except Exception as e:
            st.error(f"Error generating download link: {e}")
            return None

# Streamlit app logic
if __name__ == "__main__":
    gem = GEMModelForecast()
    
    # User inputs for location and forecast settings
    st.title("GEM Weather Forecast App")
    
    lat = st.number_input("Enter latitude (e.g., 45.0): ", -90.0, 90.0, 45.0, step=0.1)
    lon = st.number_input("Enter longitude (e.g., -75.0): ", -180.0, 180.0, -75.0, step=0.1)
    variable = st.selectbox("Select weather variable to focus on: ", ["TMP", "PRES", "WIND", "PCPN"])
    forecast_hour = st.number_input("Enter forecast hour (0-240): ", 0, 240, 0)
    n_members = st.number_input("Enter number of ensemble members: ", 1, 50, 10)
    threshold = st.number_input("Enter threshold value for probability calculation: ", 0.0, 100.0, 20.0)
    comparison = st.selectbox("Select comparison type ('gt' for greater than, 'lt' for less than): ", ["gt", "lt"])
    use_current = st.selectbox("Use current date/time for forecast? (y/n): ", ["y", "n"])
    
    if use_current == 'n':
        date_str = st.text_input("Enter date/time for forecast in 'YYYY-MM-DD HH:MM' format: ")
        try:
            forecast_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        except ValueError:
            st.error("Invalid date format, defaulting to current date/time.")
            forecast_date = datetime.now()
    else:
        forecast_date = datetime.now()
    
    if st.button("Generate Forecast"):
        # Step 1: Always download and load all variables for the multi-day forecast
        full_fp = gem.download_forecast(lat, lon, variable=None, forecast_days=8)
        gem.load_forecast(full_fp)
        
        # Plot 1: Multi-Day Forecast
        st.subheader("Multi-Day Forecast")
        fig1 = gem.plot_multiday_forecast()
        if fig1:
            st.pyplot(fig1)
            download_link1 = gem.get_image_download_link(fig1, "multi_day_forecast.png", "Download Multi-Day Forecast")
            st.markdown(download_link1, unsafe_allow_html=True)

        # Step 2: Download only the selected variable for ensemble and probability plots
        var_fp = gem.download_forecast(lat, lon, variable=variable, forecast_days=8)
        gem.load_forecast(var_fp)
        
        # Generate ensemble
        gem.generate_ensemble(lat, lon, variable=variable, n_members=n_members)

        # Plot 2: Ensemble Forecast for Selected Variable
        st.subheader(f"Ensemble Forecast for {variable}")
        fig2 = gem.plot_ensemble_forecast(variable)
        if fig2:
            st.pyplot(fig2)
            download_link2 = gem.get_image_download_link(fig2, f"ensemble_forecast_{variable}.png", f"Download Ensemble Forecast for {variable}")
            st.markdown(download_link2, unsafe_allow_html=True)

        # Plot 3: Ensemble Forecast for Temperature (TMP)
        st.subheader("Ensemble Forecast for Temperature")
        gem.generate_ensemble(lat, lon, variable="TMP", n_members=n_members)
        fig3 = gem.plot_ensemble_forecast("TMP")
        if fig3:
            st.pyplot(fig3)
            download_link3 = gem.get_image_download_link(fig3, "ensemble_forecast_TMP.png", "Download Ensemble Forecast for Temperature")
            st.markdown(download_link3, unsafe_allow_html=True)

        # Plot 4: Probability Forecast for Temperature
        st.subheader("Probability Forecast for Temperature")
        fig4 = gem.plot_probability("TMP", threshold, comparison)
        if fig4:
            st.pyplot(fig4)
            download_link4 = gem.get_image_download_link(fig4, "probability_forecast_TMP.png", "Download Probability Forecast for Temperature")
            st.markdown(download_link4, unsafe_allow_html=True)

        # Step 3: Forecast Summary
        st.subheader("Forecast Summary")
        summary = gem.get_forecast_summary()
        st.text_area("", summary, height=200)
