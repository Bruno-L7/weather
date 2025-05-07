import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import datetime as dt
import argparse
import streamlit as st

try:
    from wrf import (getvar, to_np, get_cartopy, latlon_coords, 
                     interplevel, ALL_TIMES, extract_times)
except ImportError:
    st.error("ERROR: wrf-python package not found. Install with: pip install wrf-python")
    sys.exit(1)

class WRFSimulation:
    """Class to handle WRF model simulation and analysis"""
    def __init__(self, work_dir=None):
        self.work_dir = work_dir or os.getcwd()
        self.ncfile = None
        self.output_file = None

        if 'WRF_DIR' not in os.environ:
            os.environ['WRF_DIR'] = '/opt/wrf'

    def prepare_input_data(self, start_date, end_date,
                           domain_center, domain_size, resolution,
                           download_gfs=True):
        # Convert date strings
        start = dt.datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S")
        end = dt.datetime.strptime(end_date, "%Y-%m-%d_%H:%M:%S")
        self._create_namelist_wps(start, end, domain_center, domain_size, resolution)
        if download_gfs:
            self._download_gfs_data(start, end)

        """Prepare input data for WRF simulation
        
        Args:
            start_date: Start date of simulation (YYYY-MM-DD_HH:MM:SS)
            end_date: End date of simulation (YYYY-MM-DD_HH:MM:SS)
            domain_center: Center of domain as (lat, lon)
            domain_size: Size of domain in km (x, y)
            resolution: Grid resolution in km
            download_gfs: Whether to download GFS data for initial/boundary conditions
        """
        print(f"Preparing input data for WRF simulation")
        print(f"Time period: {start_date} to {end_date}")
        print(f"Domain center: {domain_center}")
        print(f"Domain size: {domain_size[0]}km x {domain_size[1]}km")
        print(f"Resolution: {resolution}km")
        
        # Convert dates to datetime objects
        start = dt.datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S")
        end = dt.datetime.strptime(end_date, "%Y-%m-%d_%H:%M:%S")
        
        # Create namelist.wps file for domain generation
        self._create_namelist_wps(start, end, domain_center, domain_size, resolution)
        
        # Download GFS data if requested
        if download_gfs:
            self._download_gfs_data(start, end)
            
        print("Input data preparation complete")
        
    def _create_namelist_wps(self, start, end, domain_center, domain_size, resolution):
        """Create namelist.wps file for WPS (WRF Preprocessing System)"""
        # Calculate domain grid dimensions based on domain size and resolution
        nx = int(domain_size[0] / resolution)
        ny = int(domain_size[1] / resolution)
        
        # Get WRF_DIR environment variable
        wrf_dir = os.environ.get('WRF_DIR')
        
        # Write namelist.wps file
        with open(os.path.join(self.work_dir, "namelist.wps"), "w") as f:
            f.write("&share\n")
            f.write(f" wrf_core = 'ARW',\n")
            f.write(f" max_dom = 1,\n")
            f.write(f" start_date = '{start.strftime('%Y-%m-%d_%H:%M:%S')}',\n")
            f.write(f" end_date = '{end.strftime('%Y-%m-%d_%H:%M:%S')}',\n")
            f.write(f" interval_seconds = 21600,\n")
            f.write("/\n\n")
            
            f.write("&geogrid\n")
            f.write(f" parent_id = 1,\n")
            f.write(f" parent_grid_ratio = 1,\n")
            f.write(f" i_parent_start = 1,\n")
            f.write(f" j_parent_start = 1,\n")
            f.write(f" e_we = {nx},\n")
            f.write(f" e_sn = {ny},\n")
            f.write(f" dx = {resolution * 1000},\n")  # Convert to meters
            f.write(f" dy = {resolution * 1000},\n")  # Convert to meters
            f.write(f" map_proj = 'lambert',\n")
            f.write(f" ref_lat = {domain_center[0]},\n")
            f.write(f" ref_lon = {domain_center[1]},\n")
            f.write(f" truelat1 = {domain_center[0] - 5},\n")
            f.write(f" truelat2 = {domain_center[0] + 5},\n")
            f.write(f" stand_lon = {domain_center[1]},\n")
            f.write(f" geog_data_path = '{wrf_dir}/WPS_GEOG/',\n")  # FIXED LINE
            f.write("/\n\n")
            
            f.write("&ungrib\n")
            f.write(f" out_format = 'WPS',\n")
            f.write(f" prefix = 'FILE',\n")
            f.write("/\n\n")
            
            f.write("&metgrid\n")
            f.write(f" fg_name = 'FILE',\n")
            f.write(f" io_form_metgrid = 2,\n")
            f.write("/\n")
            
        print(f"Created namelist.wps in {self.work_dir}")
            
    def _download_gfs_data(self, start_date, end_date):
        """Download GFS data for the given time period"""
        print("Downloading GFS data...")
        # In a real implementation, you would use a library like wget or requests
        # to download GFS data from NOMADS or other sources
        print("Note: In a real implementation, this would download actual GFS data")
        print("      For this example, we're simulating the download process")
        
        # Create a placeholder file to simulate downloaded data
        with open(os.path.join(self.work_dir, "gfs_data_downloaded.txt"), "w") as f:
            f.write(f"GFS data for {start_date} to {end_date}\n")
            f.write("This is a placeholder file simulating downloaded GFS data\n")
            
    def configure_model(self, physics_suite="CONUS", 
                        timestep=60, 
                        num_domains=1,
                        vertical_levels=50):
        """Configure the WRF model parameters
        
        Args:
            physics_suite: Physics parameterization suite (CONUS, tropical, etc.)
            timestep: Model timestep in seconds
            num_domains: Number of nested domains
            vertical_levels: Number of vertical levels
        """
        print(f"Configuring WRF model parameters")
        print(f"Physics suite: {physics_suite}")
        print(f"Timestep: {timestep} seconds")
        print(f"Number of domains: {num_domains}")
        print(f"Vertical levels: {vertical_levels}")
        
        # Create namelist.input file
        self._create_namelist_input(physics_suite, timestep, num_domains, vertical_levels)
        
        print("Model configuration complete")
        
    def _create_namelist_input(self, physics_suite, timestep, num_domains, vertical_levels):
        """Create namelist.input file for WRF simulation"""
        with open(os.path.join(self.work_dir, "namelist.input"), "w") as f:
            f.write("&time_control\n")
            f.write(" run_days                            = 3,\n")
            f.write(" run_hours                           = 0,\n")
            f.write(" run_minutes                         = 0,\n")
            f.write(" run_seconds                         = 0,\n")
            f.write(" start_year                          = 2023,\n")
            f.write(" start_month                         = 08,\n")
            f.write(" start_day                           = 01,\n")
            f.write(" start_hour                          = 00,\n")
            f.write(" end_year                            = 2023,\n")
            f.write(" end_month                           = 08,\n")
            f.write(" end_day                             = 04,\n")
            f.write(" end_hour                            = 00,\n")
            f.write(" interval_seconds                    = 21600,\n")
            f.write(" input_from_file                     = .true.,\n")
            f.write(" history_interval                    = 180,\n")
            f.write(" frames_per_outfile                  = 1,\n")
            f.write(" restart                             = .false.,\n")
            f.write(" restart_interval                    = 7200,\n")
            f.write(" io_form_history                     = 2,\n")
            f.write(" io_form_restart                     = 2,\n")
            f.write(" io_form_input                       = 2,\n")
            f.write(" io_form_boundary                    = 2,\n")
            f.write("/\n\n")
            
            f.write("&domains\n")
            f.write(f" time_step                           = {timestep},\n")
            f.write(f" max_dom                             = {num_domains},\n")
            f.write(" e_we                                = 100,\n")
            f.write(" e_sn                                = 100,\n")
            f.write(f" e_vert                              = {vertical_levels},\n")
            f.write(" p_top_requested                     = 5000,\n")
            f.write(" num_metgrid_levels                  = 32,\n")
            f.write(" num_metgrid_soil_levels             = 4,\n")
            f.write(" dx                                  = 12000,\n")
            f.write(" dy                                  = 12000,\n")
            f.write("/\n\n")
            
            f.write("&physics\n")
            f.write(f" physics_suite                       = '{physics_suite}',\n")
            f.write(" mp_physics                          = 8,\n")
            f.write(" ra_lw_physics                       = 4,\n")
            f.write(" ra_sw_physics                       = 4,\n")
            f.write(" radt                                = 30,\n")
            f.write(" sf_sfclay_physics                   = 1,\n")
            f.write(" sf_surface_physics                  = 2,\n")
            f.write(" bl_pbl_physics                      = 1,\n")
            f.write(" bldt                                = 0,\n")
            f.write(" cu_physics                          = 1,\n")
            f.write(" cudt                                = 5,\n")
            f.write(" isfflx                              = 1,\n")
            f.write(" ifsnow                              = 1,\n")
            f.write(" icloud                              = 1,\n")
            f.write(" surface_input_source                = 3,\n")
            f.write(" num_soil_layers                     = 4,\n")
            f.write(" num_land_cat                        = 21,\n")
            f.write("/\n\n")
            
            f.write("&dynamics\n")
            f.write(" w_damping                           = 1,\n")
            f.write(" diff_opt                            = 1,\n")
            f.write(" km_opt                              = 4,\n")
            f.write(" diff_6th_opt                        = 0,\n")
            f.write(" diff_6th_factor                     = 0.12,\n")
            f.write(" base_temp                           = 290.0,\n")
            f.write(" damp_opt                            = 0,\n")
            f.write(" zdamp                               = 5000.,\n")
            f.write(" dampcoef                            = 0.2,\n")
            f.write(" khdif                               = 0,\n")
            f.write(" kvdif                               = 0,\n")
            f.write(" non_hydrostatic                     = .true.,\n")
            f.write(" moist_adv_opt                       = 1,\n")
            f.write(" scalar_adv_opt                      = 1,\n")
            f.write("/\n\n")
            
            f.write("&bdy_control\n")
            f.write(" spec_bdy_width                      = 5,\n")
            f.write(" spec_zone                           = 1,\n")
            f.write(" relax_zone                          = 4,\n")
            f.write(" specified                           = .true.,\n")
            f.write(" nested                              = .false.,\n")
            f.write("/\n\n")
            
            f.write("&namelist_quilt\n")
            f.write(" nio_tasks_per_group                 = 0,\n")
            f.write(" nio_groups                          = 1,\n")
            f.write("/\n")
            
        print(f"Created namelist.input in {self.work_dir}")
        
    def run_simulation(self, num_processors=4, use_mpi=True):
        """Run the WRF simulation
        
        Args:
            num_processors: Number of processors to use for parallel computation
            use_mpi: Whether to use MPI for parallel computing
        """
        print(f"Starting WRF simulation with {num_processors} processors")
        print("Note: In a real implementation, this would run the actual WRF model")
        print("      For this example, we're simulating the execution process")
        
        # In a real implementation, this would execute the WRF model binaries
        # using subprocess or similar
        if use_mpi:
            cmd = f"mpirun -np {num_processors} ./wrf.exe"
        else:
            cmd = "./wrf.exe"
            
        print(f"Simulating execution of: {cmd}")
        
        # Create a placeholder output file to simulate WRF output
        self._create_sample_output()
        
        print("WRF simulation completed")
    
    def _create_sample_output(self):
        """Create a sample output file to simulate WRF output"""

        output_dir = os.path.join(self.work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to output file
        self.output_file = os.path.join(output_dir, "wrfout_d01_2023-08-01_00:00:00")
        
        try:
            import netCDF4 as nc
            
            # Create netCDF file
            root_grp = nc.Dataset(self.output_file, 'w', format='NETCDF4')
            
            # Create dimensions
            root_grp.createDimension('Time', None)  # unlimited axis (WRF-Python expects 'Time')
            root_grp.createDimension('DateStrLen', 19)  # Length of the date string
            root_grp.createDimension('west_east', 100)  # WRF-Python expects these dimension names
            root_grp.createDimension('south_north', 100)
            root_grp.createDimension('bottom_top', 50)
            
            # Create time variable
            times = root_grp.createVariable('Times', 'S1', ('Time', 'DateStrLen'))
            
            # Create coordinate variables
            xlat = root_grp.createVariable('XLAT', 'f4', ('Time', 'south_north', 'west_east'))
            xlong = root_grp.createVariable('XLONG', 'f4', ('Time', 'south_north', 'west_east'))
            
            # 3D pressure and height variables needed by WRF-Python
            ph = root_grp.createVariable('PH', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            phb = root_grp.createVariable('PHB', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            p = root_grp.createVariable('P', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            pb = root_grp.createVariable('PB', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            
            # Temperature variables
            t = root_grp.createVariable('T', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            t2 = root_grp.createVariable('T2', 'f4', ('Time', 'south_north', 'west_east'))
            
            # Surface pressure
            psfc = root_grp.createVariable('PSFC', 'f4', ('Time', 'south_north', 'west_east'))
            slp = root_grp.createVariable('slp', 'f4', ('Time', 'south_north', 'west_east'))
            
            # Wind components
            u = root_grp.createVariable('U', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            v = root_grp.createVariable('V', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
            u10 = root_grp.createVariable('U10', 'f4', ('Time', 'south_north', 'west_east'))
            v10 = root_grp.createVariable('V10', 'f4', ('Time', 'south_north', 'west_east'))
            
            # Wind speed and direction
            wspd = root_grp.createVariable('wspd', 'f4', ('Time', 'south_north', 'west_east'))
            wdir = root_grp.createVariable('wdir', 'f4', ('Time', 'south_north', 'west_east'))
            
            # Precipitation
            rainc = root_grp.createVariable('RAINC', 'f4', ('Time', 'south_north', 'west_east'))
            rainnc = root_grp.createVariable('RAINNC', 'f4', ('Time', 'south_north', 'west_east'))
            
            # Create some map projection attributes required by WRF-Python
            root_grp.MAP_PROJ = 1  # Lambert Conformal
            root_grp.TRUELAT1 = 35.0
            root_grp.TRUELAT2 = 45.0
            root_grp.STAND_LON = -105.0
            root_grp.CEN_LAT = 40.0
            root_grp.CEN_LON = -105.0
            root_grp.DX = 12000.0
            root_grp.DY = 12000.0
            
            # Populate data
            # Time
            time_str = "2023-08-01_00:00:00"
            times[0, :] = list(time_str)
            
            # Create lat/lon grid
            lons = np.linspace(-115, -95, 100)
            lats = np.linspace(35, 45, 100)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            xlong[0] = lon_grid
            xlat[0] = lat_grid
            
            # Create grid indices
            x, y = np.indices((100, 100))
            center_x, center_y = 50, 50
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Surface temperature (K)
            t2[0] = 298.0 - 0.05 * dist
            
            # Create 3D temperature field
            for k in range(50):
                height_factor = 1.0 - (k / 50.0) * 0.5  # Temperature decreases with height
                t[0, k, :, :] = (298.0 - 0.05 * dist) * height_factor
            
            # Create 3D pressure fields (Pa)
            for k in range(50):
                height_factor = 1.0 - (k / 50.0) * 0.7  # Pressure decreases with height
                base_pressure = 101300.0 * height_factor
                p[0, k, :, :] = np.zeros((100, 100))  # Perturbation pressure
                pb[0, k, :, :] = base_pressure - 10.0 * dist  # Base state pressure
            
            # Create 3D geopotential heights
            for k in range(50):
                height = k * 500.0  # 500m per level
                ph[0, k, :, :] = np.zeros((100, 100))  # Perturbation geopotential
                phb[0, k, :, :] = 9.81 * height * np.ones((100, 100))  # Base geopotential
            
            # Surface pressure (Pa)
            psfc[0] = 101300.0 - 10.0 * dist
            slp[0] = 101300.0 - 10.0 * dist
            
            # Wind components (m/s)
            u10[0] = 5.0 * np.ones((100, 100))
            v10[0] = 2.0 * np.ones((100, 100))
            
            # 3D wind fields
            for k in range(50):
                height_factor = 1.0 + (k / 50.0)  # Wind increases with height
                u[0, k, :, :] = 5.0 * height_factor * np.ones((100, 100))
                v[0, k, :, :] = 2.0 * height_factor * np.ones((100, 100))
            
            # Wind speed and direction
            wspd[0] = np.sqrt(u10[0]**2 + v10[0]**2)
            wdir[0] = np.degrees(np.arctan2(-u10[0], -v10[0]))
            wdir[0] = np.where(wdir[0] < 0, wdir[0] + 360, wdir[0])
            
            # Precipitation (mm)
            rain_center_x, rain_center_y = 30, 70
            rain_dist = np.sqrt((x - rain_center_x)**2 + (y - rain_center_y)**2)
            rainc[0] = np.maximum(10.0 - 0.25 * rain_dist, 0)
            rainnc[0] = np.maximum(10.0 - 0.25 * rain_dist, 0)
            
            # Close file
            root_grp.close()
            
            print(f"Created sample WRF output file: {self.output_file}")
            
        except ImportError:
            print("WARNING: netCDF4 package not available, could not create sample output")
            with open(os.path.join(output_dir, "wrf_output.txt"), "w") as f:
                f.write("This is a placeholder for WRF output\n")
                f.write("In a real scenario, this would be a netCDF file\n")
        return self.output_file
            
    def load_results(self, output_file=None):
        if output_file:
            self.output_file = output_file
        if not self.output_file:
            st.error("No WRF output to load")
            return False
        try:
            self.ncfile = Dataset(self.output_file)
            return True
        except Exception as e:
            st.error(f"Failed to load results: {e}")
            return False
            
    def analyze_results(self):
        """Analyze WRF simulation results with error handling for missing variables"""
        if not self.ncfile:
            print("ERROR: No results loaded")
            return
            
        print("Analyzing WRF simulation results")
        
        # Extract basic variables
        try:
            # Get the simulation time
            times = extract_times(self.ncfile, ALL_TIMES)
            print(f"Simulation times: {times}")
            
            # Get key variables with error handling
            try:
                t2 = getvar(self.ncfile, "T2", timeidx=0)
                print(f"Temperature (2m) - Min: {to_np(t2).min():.2f}K, Max: {to_np(t2).max():.2f}K")
            except Exception as e:
                print(f"Could not extract T2: {e}")
            
            try:
                slp = getvar(self.ncfile, "slp", timeidx=0)
                print(f"Sea Level Pressure - Min: {to_np(slp).min():.2f}hPa, Max: {to_np(slp).max():.2f}hPa")
            except Exception as e:
                print(f"Could not extract slp: {e}")
            
            try:
                u10 = getvar(self.ncfile, "U10", timeidx=0)
                v10 = getvar(self.ncfile, "V10", timeidx=0)
                wind_speed = np.sqrt(to_np(u10)**2 + to_np(v10)**2)
                print(f"Wind Speed (10m) - Min: {wind_speed.min():.2f}m/s, Max: {wind_speed.max():.2f}m/s")
            except Exception as e:
                print(f"Could not extract wind: {e}")
            
        except Exception as e:
            print(f"ERROR: Failed to analyze results: {e}")
            
    def visualize_results(self, variable="T2", level=None, timeidx=0):
        if not self.ncfile:
            st.error("No results loaded. Run simulation first.")
            return
        # Determine and plot variable
        fig = plt.figure(figsize=(10, 8))
        try:
            # Special case for wind
            if variable.lower() == "wspd_wdir10":
                u10 = getvar(self.ncfile, "U10", timeidx=timeidx)
                v10 = getvar(self.ncfile, "V10", timeidx=timeidx)
                wspd = np.sqrt(to_np(u10)**2 + to_np(v10)**2)
                lats, lons = latlon_coords(u10)
                cs = plt.contourf(to_np(lons), to_np(lats), wspd)
                plt.barbs(to_np(lons)[::5, ::5], to_np(lats)[::5, ::5],
                          to_np(u10)[::5, ::5], to_np(v10)[::5, ::5], length=5)
                plt.title("10m Wind Speed and Direction")
                plt.colorbar(label='Wind Speed (m/s)')
            else:
                var_names = [v.lower() for v in self.ncfile.variables.keys()]
                if variable.lower() not in var_names:
                    st.error(f"Variable '{variable}' not found.")
                    return
                var = getvar(self.ncfile, variable, timeidx=timeidx)
                title = variable
                if level:
                    p = getvar(self.ncfile, "pressure", timeidx=timeidx)
                    var = interplevel(var, p, level)
                    title += f" at {level} hPa"
                lats, lons = latlon_coords(var)
                cs = plt.contourf(to_np(lons), to_np(lats), to_np(var))
                plt.title(title)
                plt.colorbar(ax=plt.gca(), shrink=0.7)
        except Exception as e:
            st.error(f"Plotting error: {e}")
            return
        # Display figure in Streamlit
        st.pyplot(fig)
        plt.close(fig)


import streamlit as st

def main():
    st.title("WRF Simulation Demo")

    # Sidebar inputs
    start_date = st.sidebar.text_input("Start date (YYYY-MM-DD_HH:MM:SS)", "2023-08-01_00:00:00")
    end_date = st.sidebar.text_input("End date (YYYY-MM-DD_HH:MM:SS)", "2023-08-04_00:00:00")
    lat = st.sidebar.number_input("Domain center latitude", value=40.0, format="%.4f")
    lon = st.sidebar.number_input("Domain center longitude", value=-105.0, format="%.4f")
    domain_x = st.sidebar.number_input("Domain size X (km)", value=1200.0)
    domain_y = st.sidebar.number_input("Domain size Y (km)", value=1200.0)
    resolution = st.sidebar.number_input("Grid resolution (km)", value=12.0)
    physics_suite = st.sidebar.selectbox("Physics Suite", ["CONUS", "tropical"] )
    timestep = st.sidebar.number_input("Time step (s)", value=60)
    num_domains = st.sidebar.number_input("Number of domains", value=1, min_value=1)
    vertical_levels = st.sidebar.number_input("Vertical levels", value=50, min_value=1)
    num_procs = st.sidebar.number_input("Processors", value=4, min_value=1)
    use_mpi = st.sidebar.checkbox("Use MPI", value=True)

    # Initialize or retrieve simulation
    if "sim" not in st.session_state:
        st.session_state.sim = WRFSimulation()
    sim = st.session_state.sim

    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running WRF simulation..."):
            sim.prepare_input_data(start_date, end_date, (lat, lon), (domain_x, domain_y), resolution)
            sim.configure_model(physics_suite, timestep, num_domains, vertical_levels)
            sim.run_simulation(num_procs, use_mpi)
            if sim.load_results():
                st.success("Simulation completed and results loaded.")
            else:
                st.error("Simulation finished but failed to load results.")

    # Visualization controls
    if sim.ncfile:
        variable = st.selectbox("Select variable to visualize", ["T2", "slp", "wspd_wdir10"])
        level = None
        if variable not in ["T2", "slp", "wspd_wdir10"]:
            level = st.number_input("Pressure level (hPa)", value=500)
        if st.button("Show Plot"):
            sim.visualize_results(variable, level)

if __name__ == "__main__":
    main()