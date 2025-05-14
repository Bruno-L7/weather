import numpy as np
import matplotlib
matplotlib.use('Agg') 
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf 
from scipy.fftpack import dct, idct
import streamlit as st
st.set_page_config(layout="wide")

class MM5Model:
    def __init__(self, grid_size=(50, 50, 10), dx=10.0, dy=10.0, dz=1.0, dt=0.1):
        self.nx, self.ny, self.nz = grid_size
        self.dx, self.dy, self.dz, self.dt = dx, dy, dz, dt
        
        # Atmospheric variables initialization (unchanged)
        self.temperature = np.zeros(grid_size)
        self.pressure = np.zeros(grid_size)
        self.wind_u = np.zeros(grid_size)
        self.wind_v = np.zeros(grid_size)
        self.wind_w = np.zeros(grid_size)
        self.humidity = np.zeros(grid_size)
        self.cloud_water = np.zeros(grid_size)
        self.precipitation = np.zeros((self.nx, self.ny))
        
        # Constants (unchanged)
        self.gravity = 9.81
        self.R = 287.0
        self.Cp = 1004.0
        self.latent_heat = 2.5e6
        self.k_diff = 25.0  # Diffusion coefficient (m²/s)
        
        # Initialize Chebyshev differentiation matrix and implicit matrix
        self.D2 = self._cheb_diff_matrix(self.nz)
        self.H = (self.nz - 1) * self.dz * 1000  # Total height in meters
        self.z_scale = (2.0 / self.H) ** 2  # Scaling factor for Chebyshev derivatives
        self.implicit_matrix = np.eye(self.nz) - self.dt * self.k_diff * self.z_scale * self.D2
        
        # Initialize topography and atmosphere (unchanged)
        self._initialize_topography()
        self._initialize_atmosphere()

    def _validate_grid_parameters(self):
        """Ensure CFL condition is satisfied"""
        max_wind = np.max([np.abs(self.wind_u), np.abs(self.wind_v), np.abs(self.wind_w)])
        cfl = max_wind * self.dt / min(self.dx, self.dy, self.dz)
        if cfl > 0.8:
            raise ValueError(f"CFL condition violated (CFL = {cfl:.2f} > 0.8). Reduce dt!")
        
    def _cheb_diff_matrix(self, n):
        """Compute Chebyshev 2nd derivative matrix"""
        x = np.cos(np.pi * np.arange(n) / (n-1))
        c = np.array([2.] + [1.]*(n-2) + [2.]) * (-1)**np.arange(n)
        X = np.tile(x, (n,1))
        dX = X.T - X
        D = np.outer(c, 1./c) / (dX + np.eye(n))
        D -= np.diag(D.sum(axis=1))
        return np.dot(D, D)

    def _initialize_topography(self):
        """Set up model topography with a simple terrain feature"""
        self.terrain = np.zeros((self.nx, self.ny))
        
        # Create a Gaussian hill in the middle of the domain
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)
        xx, yy = np.meshgrid(x, y)
        
        x0, y0 = self.nx // 2, self.ny // 2
        sigma = min(self.nx, self.ny) // 6
        
        self.terrain = 2.0 * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
        
    def _initialize_atmosphere(self):
        """Initialize the atmospheric state with realistic values"""
        np.random.seed(42)
        # Set up vertical profiles
        for k in range(self.nz):
            # Decrease temperature with height (lapse rate)
            self.temperature[:, :, k] = 288.0 - 6.5 * k * self.dz
            
            # Decrease pressure with height (hydrostatic approximation)
            if k == 0:
                self.pressure[:, :, k] = 101300.0  # Surface pressure (Pa)
            else:
                # Simple hydrostatic pressure calculation
                dz_m = self.dz * 1000.0  # Convert km to m
                self.pressure[:, :, k] = self.pressure[:, :, k-1] * np.exp(
                    -self.gravity * dz_m / (self.R * self.temperature[:, :, k]))
        
        # Add some random perturbations to make it interesting
        self.temperature += np.random.normal(0, 2.5, size=self.temperature.shape)
        
        # Initialize humidity with higher values near surface
        for k in range(self.nz):
            self.humidity[:, :, k] = 0.7 * np.exp(-k * self.dz / 3)
        
        # Add some wind field
        # Westerly jet in upper levels
        for k in range(self.nz):
            jet_strength = 10.0 * np.sin(np.pi * k / self.nz)
            self.wind_u[:, :, k] = jet_strength
            
            # Add some random components
            self.wind_u[:, :, k] += np.random.normal(0, 2, size=(self.nx, self.ny))
            self.wind_w = np.random.normal(0, 0.1, size=(self.nx, self.ny, self.nz))
            self.wind_v[:, :, k] = np.random.normal(0, 2, size=(self.nx, self.ny))
        

    def advection(self):
        """Vectorized WENO5 implementation with boundary handling"""
        # Pad temperature and wind fields with 2 ghost cells on all sides
        temp_padded = np.pad(self.temperature, ((2,2), (2,2), (2,2)), mode='edge')
        wind_u_padded = np.pad(self.wind_u, ((2,2), (2,2), (2,2)), mode='edge')
        wind_v_padded = np.pad(self.wind_v, ((2,2), (2,2), (2,2)), mode='edge')
        wind_w_padded = np.pad(self.wind_w, ((2,2), (2,2), (2,2)), mode='edge')
        
        # Calculate terms for each direction
        # X-direction
        slices_x = np.stack([temp_padded[i-2:i+3, :, :] 
                            for i in range(2, self.nx+2)], axis=0)  # Stack along axis=0
        flux_x  = self._vector_weno(slices_x, wind_u_padded, self.dx, axis=1)
        terms_x = flux_x[:,   2:-2,   2:-2]      
        # Y-direction
        slices_y = np.stack([temp_padded[:, j-2:j+3, :] 
                            for j in range(2, self.ny+2)], axis=0)
        flux_y  = self._vector_weno(slices_y, wind_v_padded, self.dy, axis=2)
        tmp_y   = flux_y[:,   2:-2,   2:-2]        # → (ny, nx, nz)
        terms_y = np.transpose(tmp_y, (1, 0, 2))      
        # Z-direction
        slices_z = np.stack([temp_padded[:, :, k-2:k+3] 
                            for k in range(2, self.nz+2)], axis=0)
        flux_z  = self._vector_weno(slices_z, wind_w_padded, self.dz, axis=3)
        tmp_z   = flux_z[:,   2:-2,   2:-2]        # → (nz, nx, ny)
        terms_z = np.transpose(tmp_z, (1, 2, 0))       
        # Update temperature
        self.temperature -= self.dt * (terms_x + terms_y + terms_z)
        self._enforce_bounds()
        st.write(terms_x.shape, terms_y.shape, terms_z.shape, self.temperature.shape)

    def _vector_weno(self, stencils, vel_field, delta, axis):
        """
        Vectorized WENO5 along the 'axis'-th dimension of stencils.
        - stencils.shape = (n_slices, 5, Ny_p, Nz_p) for x-direction (axis=1),
                        or (5, n_slices, Ny_p, Nz_p) for y (axis=2), etc.
        - vel_field is the padded velocity array of shape (Nx_p, Ny_p, Nz_p).
        """
        epsilon = 1e-40

        # 1) figure out which axis in vel_field corresponds to this direction
        vel_axis = axis - 1

        # 2) slice out the same range of vel_field that stencils covers
        n = stencils.shape[0]
        slicer = [slice(None)] * vel_field.ndim
        slicer[vel_axis] = slice(2, 2 + n)
        vel_slice = vel_field[tuple(slicer)]               # shape e.g. (50,54,14)
        vel_slice = np.moveaxis(vel_slice, vel_axis, 0)    # bring x-index to front → (50,54,14)

        # 3) build the mask of positive‐wind cells and expand it to stencil shape
        pos = (vel_slice > 0)
        # we want a boolean array shaped exactly like stencils:
        mask = np.expand_dims(pos, axis=axis)              # e.g. (50,1,54,14)
        mask = np.broadcast_to(mask, stencils.shape)       # (50,5,54,14)

        # 4) flip stencils where wind is negative
        st = np.where(mask, stencils, np.flip(stencils, axis=axis))

        # 5) move the 5‐point stencil axis to the end so we can do WENO on st[..., i]
        st_last = np.moveaxis(st, axis, -1)                # now: (50,54,14,5)

        # 6) compute smoothness indicators along that last axis
        β0 = (13/12)*(st_last[...,0] - 2*st_last[...,1] + st_last[...,2])**2 \
        +       (st_last[...,0] -   4*st_last[...,1] + 3*st_last[...,2])**2

        β1 = (13/12)*(st_last[...,1] - 2*st_last[...,2] + st_last[...,3])**2 \
        +       (st_last[...,1] -   st_last[...,3])**2

        β2 = (13/12)*(st_last[...,2] - 2*st_last[...,3] + st_last[...,4])**2 \
        +       (3*st_last[...,2] - 4*st_last[...,3] + st_last[...,4])**2

        # 7) nonlinear weights
        α = np.array([0.1, 0.6, 0.3]) / (epsilon + np.stack([β0,β1,β2], axis=-1))**2
        w = α / np.sum(α, axis=-1, keepdims=True)          # shape (50,54,14,3)

        # 8) candidate reconstructions
        q0 = ( 2*st_last[...,0] -  7*st_last[...,1] + 11*st_last[...,2]) / 6
        q1 = (-1*st_last[...,1] +  5*st_last[...,2] +  2*st_last[...,3]) / 6
        q2 = ( 2*st_last[...,2] +  5*st_last[...,3] -  1*st_last[...,4]) / 6

        # 9) final WENO flux (without dividing by delta yet)
        flux = w[...,0]*q0 + w[...,1]*q1 + w[...,2]*q2  # shape (50,54,14)

        # 10) scale by |velocity|/delta
        return (np.abs(vel_slice) / delta) * flux        # shape (50,54,14)


    def vertical_diffusion(self):
        """Hybrid implicit-spectral vertical diffusion with proper scaling and damping."""
        # 1. Convert dz to meters and compute total height H
        H = (self.nz - 1) * self.dz * 1000.0  # dz in km → H in meters
        z_scale = (2.0 / H) ** 2  # Scaling for Chebyshev derivatives in physical space

        # 2. Apply spectral damping (optional but included per your request)
        temp_spectral = dct(self.temperature, type=1, norm='ortho', axis=2)
        damping = np.exp(-0.01 * (np.arange(self.nz)**2))  # Damp high frequencies
        temp_spectral *= damping
        self.temperature = idct(temp_spectral, type=1, norm='ortho', axis=2)

        # 3. Implicit solve: (I - dt * k_diff * z_scale * D2) T_new = T_old
        implicit_matrix = np.eye(self.nz) - self.dt * self.k_diff * z_scale * self.D2
        for i in range(self.nx):
            for j in range(self.ny):
                self.temperature[i, j, :] = np.linalg.solve(implicit_matrix, self.temperature[i, j, :])

        self._enforce_bounds()

    def _enforce_bounds(self):
        """Ensure physical values with smooth clipping"""
        self.temperature = np.clip(self.temperature, 200, 350)
        self.humidity = erf(self.humidity)  # Smooth constraint to [0,1]
        self.cloud_water = np.maximum(0, self.cloud_water)
    
    def radiation(self):
        """Simplified radiation scheme"""
        # Cooling rate decreases with height (stronger cooling near surface)
        cooling_rate = np.zeros_like(self.temperature)
        for k in range(self.nz):
            cooling_rate[:, :, k] = 0.5 * np.exp(-k * 0.2)
        
        # Apply cooling
        self.temperature -= cooling_rate * self.dt
    
    def microphysics(self):
        """
        Simplified microphysics scheme to handle phase changes of water and precipitation.
        This is a highly simplified version of what MM5 would use.
        """
        # Threshold for cloud formation
        rh_threshold = 0.6
        
        for i in range(self.nx):
            for j in range(self.ny):
                column_precip = 0.0
                
                for k in range(self.nz-1, -1, -1):  # Process from top to bottom
                    # Cloud formation when humidity exceeds threshold
                    if self.humidity[i, j, k] > rh_threshold:
                        # Convert excess vapor to cloud water
                        excess = (self.humidity[i, j, k] - rh_threshold) * 0.1
                        self.humidity[i, j, k] = rh_threshold
                        self.cloud_water[i, j, k] += excess
                        
                        # Release latent heat from condensation
                        self.temperature[i, j, k] += excess * self.latent_heat / self.Cp
                    
                    # Cloud water to precipitation conversion (simplified)
                    if self.cloud_water[i, j, k] > 0.001:
                        precip_rate = 0.2 * self.cloud_water[i, j, k]
                        self.cloud_water[i, j, k] -= precip_rate

                        # convert to mm: precip_rate [kg/kg] * ρ [kg/m³] * Δz [m] → kg/m² → mm
                        dz_m = self.dz * 1000.0
                        rho = self.pressure[i, j, k] / (self.R * self.temperature[i, j, k])
                        column_precip += precip_rate * rho * dz_m

                # after looping all k
                self.precipitation[i, j] += column_precip
    
    def step(self):
        """Run one time step of the model"""
        # Main model processes in sequence
        self._validate_grid_parameters()  # ADD THIS LINE
        self.advection()
        self.vertical_diffusion()
        self.radiation()
        self.microphysics()
        
        # Ensure physical bounds on variables
        self.temperature = np.maximum(200.0, self.temperature)  # Prevent unphysically cold temperatures
        self.humidity = np.clip(self.humidity, 0.0, 1.0)  # RH between 0-100%
        self.cloud_water = np.maximum(0.0, self.cloud_water)
    
    def run_simulation(self, num_steps=48):
        """Run the model for multiple time steps"""
        for _ in range(num_steps):
            self.step()
    
    def plot_results(self):
        """Visualize model results"""
        plt.close('all')  # ← ADD THIS LINE to clear previous figures
        fig = plt.figure(figsize=(18, 12))  # Restore original size
        
        # Plot surface temperature
        ax1 = fig.add_subplot(221)
        surf_temp = self.temperature[:, :, 0] - 273.15
        
        # Use fixed color range instead of data-dependent vmin/vmax
        vmin = surf_temp.min() - 5  # ← Add buffer
        vmax = surf_temp.max() + 5
        im1 = ax1.imshow(surf_temp, cmap='RdBu_r', 
                        vmin=surf_temp.min(), 
                        vmax=surf_temp.max()) 
        st.write("Surface temperature range (°C):", surf_temp.min(), surf_temp.max())
        st.write("Accumulated precipitation range (mm):", self.precipitation.min(), self.precipitation.max())
        ax1.set_title('Surface Temperature (°C)')
        plt.colorbar(im1, ax=ax1)
        
        # Plot precipitation
        ax2 = fig.add_subplot(222)
        precip_max = max(0.01, self.precipitation.max())  # Ensure non-zero max
        im2 = ax2.imshow(self.precipitation, cmap='Blues', vmin=0, vmax=precip_max)
        ax2.set_title('Accumulated Precipitation (mm)')
        plt.colorbar(im2, ax=ax2)
        
        # Plot surface wind
        ax3 = fig.add_subplot(223)
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)
        X, Y = np.meshgrid(x, y)
        # Subsample wind field for clearer visualization
        skip = 3
        ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                   self.wind_u[::skip, ::skip, 0], self.wind_v[::skip, ::skip, 0])
        ax3.set_title('Surface Wind Field')
        
        # Plot 3D terrain with surface temperature
        ax4 = fig.add_subplot(224, projection='3d')
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)
        X, Y = np.meshgrid(x, y)
        surf = ax4.plot_surface(X, Y, self.terrain, cmap=cm.terrain, alpha=0.8)
        ax4.set_title('Model Terrain')
        ax4.set_zlabel('Elevation (km)')
        
        plt.tight_layout()
        return fig


def run_air_quality_simulation():
    """
    Example use case: Air quality modeling using MM5
    """
    st.write("Running MM5 for Air Quality Modeling...")
    
    # Initialize MM5 model
    model = MM5Model(grid_size=(50, 50, 15), dx=5.0, dy=5.0, dt=0.02)    
    # Add pollutant concentration field (simplified)
    pollutant = np.zeros((model.nx, model.ny, model.nz))
    
    # Add a pollution source in the center of the domain at surface level
    source_i, source_j = model.nx // 2, model.ny // 2
    pollutant[source_i-2:source_i+3, source_j-2:source_j+3, 0] = 100.0
    
    # Run weather simulation
    model.run_simulation(48)
    
    # Simulate pollutant dispersion based on wind field (simplified)
    for _ in range(48):
        # Simple advection of pollutant
        pollutant_new = np.copy(pollutant)
        
        for i in range(1, model.nx-1):
            for j in range(1, model.ny-1):
                for k in range(model.nz):
                    # x-direction advection
                    if model.wind_u[i, j, k] > 0:
                        dx_term = model.wind_u[i, j, k] * (pollutant[i, j, k] - pollutant[i-1, j, k]) / model.dx
                    else:
                        dx_term = model.wind_u[i, j, k] * (pollutant[i+1, j, k] - pollutant[i, j, k]) / model.dx
                    
                    # y-direction advection
                    if model.wind_v[i, j, k] > 0:
                        dy_term = model.wind_v[i, j, k] * (pollutant[i, j, k] - pollutant[i, j-1, k]) / model.dy
                    else:
                        dy_term = model.wind_v[i, j, k] * (pollutant[i, j+1, k] - pollutant[i, j, k]) / model.dy
                    
                    # Update pollutant
                    pollutant_new[i, j, k] = pollutant[i, j, k] - model.dt * (dx_term + dy_term)
        
        pollutant = pollutant_new
        
        # Add some vertical mixing
        for i in range(model.nx):
            for j in range(model.ny):
                for k in range(1, model.nz):
                    mix_rate = 0.1
                    pollutant[i, j, k] += mix_rate * (pollutant[i, j, k-1] - pollutant[i, j, k])
    
    st.write("→ Surface T (°C): min, max =", 
      np.min(model.temperature[:, :, 0] - 273.15), 
      np.max(model.temperature[:, :, 0] - 273.15))
    st.write("→ Total precipitation (mm): min, max =",
      np.min(model.precipitation), 
      np.max(model.precipitation))

    # Visualize results
    fig = plt.figure(figsize=(15, 10))
    
    # Plot surface pollutant concentration
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(pollutant[:, :, 0], cmap='YlOrRd')
    ax1.set_title('Surface Pollutant Concentration')
    plt.colorbar(im1, ax=ax1)
    
    # Plot wind field at surface level
    ax2 = fig.add_subplot(222)
    x = np.arange(0, model.nx)
    y = np.arange(0, model.ny)
    X, Y = np.meshgrid(x, y)
    skip = 3
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               model.wind_u[::skip, ::skip, 0], model.wind_v[::skip, ::skip, 0])
    ax2.set_title('Surface Wind Field')
    
    # Plot vertical cross-section of pollutant
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(pollutant[:, model.ny//2, :].T, cmap='YlOrRd', origin='lower',
                    aspect='auto', extent=[0, model.nx, 0, model.nz])
    ax3.set_title('Vertical Cross-section of Pollutant')
    ax3.set_xlabel('X Grid Point')
    ax3.set_ylabel('Height Level')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 3D visualization of pollutant plume
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Create a mask for areas with significant pollution
    threshold = 5.0
    x, y, z = np.where(pollutant > threshold)
    pollution_values = pollutant[x, y, z]
    
    # Normalize for color mapping
    norm = plt.Normalize(pollution_values.min(), pollution_values.max())
    colors = cm.YlOrRd(norm(pollution_values))
    
    # 3D scatter plot of pollution
    scatter = ax4.scatter(x, y, z, c=colors, s=20, alpha=0.5)
    ax4.set_title('3D Pollutant Distribution')
    ax4.set_xlabel('X Grid')
    ax4.set_ylabel('Y Grid')
    ax4.set_zlabel('Z Level')
    
    plt.tight_layout()
    return fig


def run_climate_research_simulation():
    """
    Example use case: Climate research using MM5
    """
    st.write("Running MM5 for Climate Research...")
    
    # Initialize MM5 model with larger domain for climate research
    model = MM5Model(grid_size=(80, 80, 20), dx=25.0, dy=25.0, dt=0.02)
    # Store monthly average temperature at 2m height
    num_months = 12
    monthly_temps = np.zeros((model.nx, model.ny, num_months))
    baseline_temp = model.temperature.copy()

    # Run simulation for each "month"
    for month in range(num_months):
        model.temperature = baseline_temp.copy()
        st.write(f"Simulating month {month+1}...")
        
        # Adjust seasonal forcing 
        season_factor = np.sin(2 * np.pi * month / 12)
        # DIAGNOSTIC: what *should* the surface shift be?
        st.write(f"Month {month+1}: seasonal_adjustment at surface = "
              f"{15.0 * season_factor:.2f} K")
        
        # Apply seasonal temperature adjustment
        for k in range(model.nz):
            seasonal_adjustment = 15.0 * season_factor * np.exp(-k * 0.2)
            model.temperature[:, :, k] += seasonal_adjustment
        
        # Run model for this month (4 weeks)
        model.run_simulation(28)
        
        # Store monthly average surface temperature
        monthly_temps[:, :, month] = model.temperature[:, :, 0] - 273.15  # K to °C
    
    # Calculate annual mean and seasonal amplitude
    annual_mean = np.mean(monthly_temps, axis=2)
    seasonal_amplitude = np.max(monthly_temps, axis=2) - np.min(monthly_temps, axis=2)
    
    # Visualize results
    fig = plt.figure(figsize=(16, 12))
    
    # Plot annual mean temperature
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(annual_mean, cmap='RdBu_r')
    ax1.set_title('Annual Mean Temperature (°C)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot seasonal temperature amplitude
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(seasonal_amplitude, cmap='viridis')
    ax2.set_title('Seasonal Temperature Range (°C)')
    plt.colorbar(im2, ax=ax2)
    
    # Plot seasonal cycle for a specific point
    ax3 = fig.add_subplot(223)
    point_x, point_y = model.nx // 2, model.ny // 2
    months = np.arange(1, 13)
    baseline_C = baseline_temp[point_x, point_y, 0] - 273.15
    expected = baseline_C + 15.0 * np.sin(2 * np.pi * (months-1) / 12)

    ax3.plot(months, expected, 's--', label='Analytic forcing')
    ax3.plot(months, monthly_temps[point_x, point_y, :], 'o-', linewidth=2, label='Model Output')
    ax3.legend()
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title(f'Seasonal Temperature Cycle at Point ({point_x}, {point_y})')
    ax3.grid(True)
    
    # Plot terrain with annual mean temperature
    ax4 = fig.add_subplot(224, projection='3d')
    x = np.arange(0, model.nx)
    y = np.arange(0, model.ny)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain surface
    surf = ax4.plot_surface(X, Y, model.terrain, cmap=cm.terrain, alpha=0.7)
    
    # Add temperature as color on top of terrain
    # (using subsampling for clearer visualization)
    skip = 4
    scatter = ax4.scatter(X[::skip, ::skip], Y[::skip, ::skip], 
                          model.terrain[::skip, ::skip] + 0.1,  # Slightly above terrain
                          c=annual_mean[::skip, ::skip], 
                          cmap='RdBu_r', s=40, alpha=1.0)
    plt.colorbar(scatter, ax=ax4, label='Annual Mean Temperature (°C)')
    
    ax4.set_title('Terrain with Annual Mean Temperature')
    
    plt.tight_layout()
    return fig


def main():
    st.title("MM5 Mesoscale Model Simulation")
    st.markdown("### Fifth-Generation Penn State/NCAR Mesoscale Model Demonstration")
    
    # Add parameter controls in sidebar
    with st.sidebar:
        st.header("Simulation Parameters")
        dt = st.slider("Time step (dt)", 0.01, 0.5, 0.02)
        num_steps = st.number_input("Number of steps", 24, 240, 24)

    # Basic model demonstration
    model = MM5Model(dt=dt)
    st.write(f"Initialized MM5 model with grid size: {(model.nx, model.ny, model.nz)}")
    
    if st.button("Run Basic Simulation"):
        with st.spinner("Running simulation..."):
            model.run_simulation(num_steps)
            fig = model.plot_results()
            st.pyplot(fig)

    # Air quality demo
    if st.checkbox("Show Air Quality Simulation"):
        fig = run_air_quality_simulation()
        st.pyplot(fig)

    # Climate research demo
    if st.checkbox("Show Climate Research Simulation"):
        fig = run_climate_research_simulation()
        st.pyplot(fig)


if __name__ == "__main__":
    main()