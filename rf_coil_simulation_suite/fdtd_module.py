import torch
import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
C0 = 299792458.0  # Speed of light in vacuum (m/s)
EPSILON0 = 8.854187817e-12  # Permittivity of free space (F/m)
MU0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Sample Material Database
MATERIAL_DATABASE = {
    0: {'name': 'Air', 'epsilon_r': 1.0, 'sigma': 0.0, 'mu_r': 1.0, 'rho': 1.225},
    1: {'name': 'Copper', 'epsilon_r': 1.0, 'sigma': 5.8e7, 'mu_r': 1.0, 'rho': 8960.0},
    2: {'name': 'Muscle', 'epsilon_r': 57.0, 'sigma': 0.76, 'mu_r': 1.0, 'rho': 1090.0},
    3: {'name': 'Fat', 'epsilon_r': 5.1, 'sigma': 0.04, 'mu_r': 1.0, 'rho': 920.0},
    4: {'name': 'Brain (Grey Matter)', 'epsilon_r': 58.0, 'sigma': 0.55, 'mu_r': 1.0, 'rho': 1040.0}
}

class FDTDSimulator:
    """
    A class for performing 3D Finite-Difference Time-Domain (FDTD) simulations.

    This simulator assumes a uniform Yee grid and uses PyTorch for calculations.
    It initializes the FDTD grid, field components (E and H), and material
    properties (epsilon_r, mu_r, sigma).
    """

    def __init__(self, 
                 dimensions: tuple[int, int, int], 
                 cell_size: float, 
                 pml_layers: int = 0, # Added PML layers parameter
                 dtype: torch.dtype = torch.float32, 
                 device: torch.device = torch.device('cpu')):
        """
        Initializes the FDTDSimulator.

        Args:
            dimensions (tuple[int, int, int]): The number of cells in each
                                               dimension (Nx, Ny, Nz).
            cell_size (float): The size of each cubic cell (dx = dy = dz) in meters.
            pml_layers (int, optional): Number of PML layers on each side.
                                        Defaults to 0 (no PML).
            dtype (torch.dtype, optional): The data type for PyTorch tensors.
                                           Defaults to torch.float32.
            device (torch.device, optional): The device for PyTorch tensors (e.g., 'cpu', 'cuda').
                                             Defaults to torch.device('cpu').
        """
        self.dimensions = dimensions
        self.cell_size = cell_size
        self.pml_layers = pml_layers
        self.dtype = dtype
        self.device = device

        self.Nx, self.Ny, self.Nz = dimensions

        self.initialize_fdtd_grid()
        if self.pml_layers > 0:
            self.initialize_pml()

    def initialize_fdtd_grid(self):
        """
        Initializes the FDTD grid, field arrays, and material property arrays.

        Calculates the time step `dt` based on the Courant stability criterion.
        Initializes Ex, Ey, Ez, Hx, Hy, Hz fields to zeros.
        Initializes relative permittivity (epsilon_r), relative permeability (mu_r),
        and electrical conductivity (sigma) for each cell.
        For simplicity, all field and material arrays are currently sized
        identically to `self.dimensions = (Nx, Ny, Nz)`. This implies that
        E, H, and material properties are co-located at cell centers or that
        the specific staggered locations within the Yee cell are implicitly
        handled by the update equations.
        """
        # Calculate Time Step dt using Courant stability criterion
        courant_factor = 0.9  # Stability factor, should be <= 1
        # For dx = dy = dz = cell_size, dt <= (cell_size / (C0 * sqrt(3)))
        self.dt = (courant_factor * self.cell_size) / (C0 * torch.sqrt(torch.tensor(3.0, dtype=self.dtype, device=self.device)))

        # Initialize Field Arrays (all zeros)
        # Assuming dimensions (Nx, Ny, Nz) for all fields for simplicity.
        # Specific Yee cell staggering will be handled by indexing in update equations.
        self.Ex = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.Ey = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.Ez = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        
        self.Hx = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.Hy = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.Hz = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

        # Initialize Material Property Arrays (default to free space)
        self.epsilon_r = torch.ones(self.dimensions, dtype=self.dtype, device=self.device)
        self.mu_r = torch.ones(self.dimensions, dtype=self.dtype, device=self.device)
        self.sigma = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.rho = 1000.0 * torch.ones(self.dimensions, dtype=self.dtype, device=self.device) # Default to density of water

        print(f"FDTD grid initialized: Dimensions=({self.Nx}, {self.Ny}, {self.Nz}), Cell Size={self.cell_size:.2e} m, dt={self.dt:.2e} s")
        print(f"Field arrays (Ex, Ey, Ez, Hx, Hy, Hz) created with shape {self.dimensions}.")
        print(f"Material arrays (epsilon_r, mu_r, sigma, rho) created with shape {self.dimensions}.")

    def load_material_properties(self, phantom_data: torch.Tensor, material_database: dict):
        """
        Populates the material property arrays (epsilon_r, sigma, mu_r)
        based on a phantom data map and a material database.

        Args:
            phantom_data (torch.Tensor): A 3D tensor with the same spatial
                                         dimensions as the grid (self.dimensions).
                                         Each element is an integer index
                                         corresponding to a key in material_database.
                                         Must be on the same device as the simulator
                                         and have an integer dtype.
            material_database (dict): A dictionary where keys are integer material
                                      indices and values are dictionaries with
                                      'epsilon_r', 'sigma', and 'mu_r'.
        Raises:
            ValueError: If phantom_data dimensions do not match grid dimensions.
            ValueError: If phantom_data contains indices not found in material_database.
        """
        if phantom_data.shape != self.dimensions:
            raise ValueError(
                f"phantom_data dimensions {phantom_data.shape} "
                f"do not match grid dimensions {self.dimensions}."
            )
        if phantom_data.device != self.device:
            # Or try to move it: phantom_data = phantom_data.to(self.device)
            raise ValueError(
                f"phantom_data device ({phantom_data.device}) "
                f"does not match simulator device ({self.device})."
            )
        if not (phantom_data.dtype == torch.int or phantom_data.dtype == torch.long):
             # Or try to cast it: phantom_data = phantom_data.long()
            raise ValueError(
                f"phantom_data dtype ({phantom_data.dtype}) must be torch.int or torch.long."
            )

        unique_material_indices = torch.unique(phantom_data)

        for material_idx in unique_material_indices:
            idx_val = material_idx.item() # Convert tensor to Python scalar
            if idx_val not in material_database:
                # Option: Assign default (air) properties and print warning
                # print(f"Warning: Material index {idx_val} not found in database. Assigning Air properties.")
                # material_props = material_database.get(0, {'epsilon_r': 1.0, 'sigma': 0.0, 'mu_r': 1.0})
                raise ValueError(f"Material index {idx_val} from phantom_data not found in material_database.")
            
            material_props = material_database[idx_val]
            mask = (phantom_data == material_idx)
            
            self.epsilon_r[mask] = material_props['epsilon_r']
            self.sigma[mask] = material_props['sigma']
            self.mu_r[mask] = material_props['mu_r']
            if 'rho' in material_props: # Check if rho is defined for this material
                self.rho[mask] = material_props['rho']
            # Else, self.rho keeps its default value (e.g., 1000 kg/m^3) for this material
        
        print(f"Material properties loaded from phantom_data. Updated epsilon_r, sigma, mu_r, rho based on {len(unique_material_indices)} unique materials.")

    def apply_source(self, 
                     time: float, 
                     location_indices: tuple[int,int,int], 
                     field_component: str, 
                     waveform_type: str, 
                     frequency: float, 
                     amplitude: float, 
                     waveform_params: dict = None):
        """
        Applies a source term to a specified field component at a given location.

        This method calculates a source value based on the specified waveform
        and adds it to the field component at the given grid cell.

        Args:
            time (float): The current simulation time in seconds.
            location_indices (tuple[int,int,int]): (ix, iy, iz) indices of the cell.
            field_component (str): 'Ex', 'Ey', or 'Ez'.
            waveform_type (str): 'gaussian_pulse' or 'sine_burst'.
            frequency (float): Frequency in Hz (center for Gaussian, actual for sine).
            amplitude (float): Peak amplitude of the waveform (e.g., V/m).
            waveform_params (dict, optional): Additional parameters for the waveform.
                For 'gaussian_pulse': {'tau': pulse_width_param, 't0': time_offset}
                    tau:  Related to pulse width (e.g., standard deviation).
                    t0: Time of the pulse peak.
                Defaults are provided if not specified.

        Raises:
            ValueError: If field_component is invalid, location is out of bounds,
                        or waveform_type is unknown.
        """
        ix, iy, iz = location_indices
        
        # Validate location_indices
        if not (0 <= ix < self.Nx and 0 <= iy < self.Ny and 0 <= iz < self.Nz):
            raise ValueError(f"Source location {location_indices} is out of grid bounds ({self.Nx}, {self.Ny}, {self.Nz}).")

        # Validate field_component
        if field_component not in ['Ex', 'Ey', 'Ez']:
            raise ValueError(f"Invalid field_component '{field_component}'. Must be 'Ex', 'Ey', or 'Ez'.")

        source_value = 0.0
        time_tensor = torch.tensor(time, dtype=self.dtype, device=self.device) # Ensure time is a tensor for torch functions

        if waveform_type == 'gaussian_pulse':
            if waveform_params is None: waveform_params = {}
            # Default t0: peak of pulse at 3*tau. Default tau: related to frequency.
            tau_default = 1.0 / (frequency * 0.4 * torch.pi) # Makes pulse reasonably well-contained
            t0_default = 3.0 * tau_default 
            
            tau = waveform_params.get('tau', tau_default)
            t0 = waveform_params.get('t0', t0_default)
            
            # Modulated Gaussian pulse: A * exp(-((t-t0)/tau)^2) * cos(2*pi*f*(t-t0))
            exponent = -((time_tensor - t0) / tau)**2
            cosine_term = torch.cos(2 * torch.pi * frequency * (time_tensor - t0))
            source_value = amplitude * torch.exp(exponent) * cosine_term
        
        elif waveform_type == 'sine_burst':
            # For now, a continuous sine wave. Burst parameters could be added later.
            # if waveform_params is None: waveform_params = {}
            # start_time = waveform_params.get('start_time', 0.0)
            # end_time = waveform_params.get('end_time', float('inf'))
            # if start_time <= time < end_time:
            source_value = amplitude * torch.sin(2 * torch.pi * frequency * time_tensor)
            # else:
            #     source_value = 0.0
        
        else:
            raise ValueError(f"Unknown waveform_type '{waveform_type}'. Must be 'gaussian_pulse' or 'sine_burst'.")

        # Add to the specified field component (soft source)
        if field_component == 'Ex':
            self.Ex[ix, iy, iz] += source_value.item() # Ensure scalar addition if source_value is tensor
        elif field_component == 'Ey':
            self.Ey[ix, iy, iz] += source_value.item()
        elif field_component == 'Ez':
            self.Ez[ix, iy, iz] += source_value.item()
        
        # print(f"Time: {time:.2e}, Applied source to {field_component}[{ix},{iy},{iz}]: {source_value.item():.2e}")

    def initialize_pml(self):
        """
        Initializes PML parameters and auxiliary conductivity arrays.
        This implementation uses a simplified approach where effective PML
        conductivities are stored for E and H fields.
        A polynomial grading (m=3) is used for sigma profiles.
        """
        if self.pml_layers == 0:
            print("PML layers set to 0, skipping PML initialization.")
            return

        m = 3  # Polynomial grading order
        # Target reflection error R0 (e.g., 1e-5 for -100dB reflection)
        # For simplicity, we'll use a fixed sigma_max_pml for now,
        # rather than calculating from R0, as R0 depends on precise PML formulation.
        sigma_max_pml = 0.8 * (m + 1) / (150 * np.pi * self.cell_size * self.pml_layers) # Heuristic
        # A more common formula for sigma_max related to reflection error R0:
        # sigma_max = - (m + 1) * EPSILON0 * C0 * np.log(R0) / (2 * self.pml_layers * self.cell_size)
        # For R0 = 1e-5, m=3, this can be large. The heuristic above is often used.
        # Let's use a more controllable sigma_max for now.
        # sigma_max_pml = 1e5 # Example value, often tuned. Let's stick to the heuristic.

        print(f"Initializing PML with {self.pml_layers} layers. Max PML sigma: {sigma_max_pml:.2e} S/m")

        # Create 1D PML conductivity profiles (ramping up from 0)
        # Profile for electric field PML conductivity (sigma_e)
        # Profile for magnetic field PML conductivity (sigma_m, for magnetic losses in PML)
        # For standard PML, sigma_m = sigma_e * (MU0 / EPSILON0)
        
        depth_profile = torch.arange(self.pml_layers, dtype=self.dtype, device=self.device) / self.pml_layers
        sigma_profile_e = sigma_max_pml * (depth_profile ** m)
        sigma_profile_m = sigma_profile_e * (MU0 / EPSILON0) # Magnetic equivalent

        # Initialize 3D PML conductivity arrays (for E and H fields)
        # These will store sigma_x, sigma_y, sigma_z components for E and H fields
        # Example: self.pml_sigma_Ex will be sigma_y and sigma_z in PML regions for Ex updates
        # For simplicity, we'll create combined sigma arrays for E and H.
        # These represent the sigma that each field component (Ex, Ey, Ez, Hx, Hy, Hz) sees along its propagation direction.
        
        self.pml_sigma_Ex = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.pml_sigma_Ey = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.pml_sigma_Ez = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        
        self.pml_sigma_Hx = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.pml_sigma_Hy = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        self.pml_sigma_Hz = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

        # Populate for Ex (affected by sigma_y and sigma_z in PML)
        # Y-PML regions for Ex (sigma_y component)
        self.pml_sigma_Ex[:, :self.pml_layers, :] += sigma_profile_e.flip(dims=[0]).reshape(1, -1, 1)
        self.pml_sigma_Ex[:, self.Ny-self.pml_layers:, :] += sigma_profile_e.reshape(1, -1, 1)
        # Z-PML regions for Ex (sigma_z component)
        self.pml_sigma_Ex[:, :, :self.pml_layers] += sigma_profile_e.flip(dims=[0]).reshape(1, 1, -1)
        self.pml_sigma_Ex[:, :, self.Nz-self.pml_layers:] += sigma_profile_e.reshape(1, 1, -1)

        # Populate for Ey (affected by sigma_x and sigma_z in PML)
        # X-PML regions for Ey (sigma_x component)
        self.pml_sigma_Ey[:self.pml_layers, :, :] += sigma_profile_e.flip(dims=[0]).reshape(-1, 1, 1)
        self.pml_sigma_Ey[self.Nx-self.pml_layers:, :, :] += sigma_profile_e.reshape(-1, 1, 1)
        # Z-PML regions for Ey (sigma_z component)
        self.pml_sigma_Ey[:, :, :self.pml_layers] += sigma_profile_e.flip(dims=[0]).reshape(1, 1, -1)
        self.pml_sigma_Ey[:, :, self.Nz-self.pml_layers:] += sigma_profile_e.reshape(1, 1, -1)

        # Populate for Ez (affected by sigma_x and sigma_y in PML)
        # X-PML regions for Ez (sigma_x component)
        self.pml_sigma_Ez[:self.pml_layers, :, :] += sigma_profile_e.flip(dims=[0]).reshape(-1, 1, 1)
        self.pml_sigma_Ez[self.Nx-self.pml_layers:, :, :] += sigma_profile_e.reshape(-1, 1, 1)
        # Y-PML regions for Ez (sigma_y component)
        self.pml_sigma_Ez[:, :self.pml_layers, :] += sigma_profile_e.flip(dims=[0]).reshape(1, -1, 1)
        self.pml_sigma_Ez[:, self.Ny-self.pml_layers:, :] += sigma_profile_e.reshape(1, -1, 1)

        # Populate for Hx (affected by sigma_m_y and sigma_m_z in PML)
        # Y-PML regions for Hx
        self.pml_sigma_Hx[:, :self.pml_layers, :] += sigma_profile_m.flip(dims=[0]).reshape(1, -1, 1)
        self.pml_sigma_Hx[:, self.Ny-self.pml_layers:, :] += sigma_profile_m.reshape(1, -1, 1)
        # Z-PML regions for Hx
        self.pml_sigma_Hx[:, :, :self.pml_layers] += sigma_profile_m.flip(dims=[0]).reshape(1, 1, -1)
        self.pml_sigma_Hx[:, :, self.Nz-self.pml_layers:] += sigma_profile_m.reshape(1, 1, -1)
        
        # Populate for Hy (affected by sigma_m_x and sigma_m_z in PML)
        # X-PML regions for Hy
        self.pml_sigma_Hy[:self.pml_layers, :, :] += sigma_profile_m.flip(dims=[0]).reshape(-1, 1, 1)
        self.pml_sigma_Hy[self.Nx-self.pml_layers:, :, :] += sigma_profile_m.reshape(-1, 1, 1)
        # Z-PML regions for Hy
        self.pml_sigma_Hy[:, :, :self.pml_layers] += sigma_profile_m.flip(dims=[0]).reshape(1, 1, -1)
        self.pml_sigma_Hy[:, :, self.Nz-self.pml_layers:] += sigma_profile_m.reshape(1, 1, -1)

        # Populate for Hz (affected by sigma_m_x and sigma_m_y in PML)
        # X-PML regions for Hz
        self.pml_sigma_Hz[:self.pml_layers, :, :] += sigma_profile_m.flip(dims=[0]).reshape(-1, 1, 1)
        self.pml_sigma_Hz[self.Nx-self.pml_layers:, :, :] += sigma_profile_m.reshape(-1, 1, 1)
        # Y-PML regions for Hz
        self.pml_sigma_Hz[:, :self.pml_layers, :] += sigma_profile_m.flip(dims=[0]).reshape(1, -1, 1)
        self.pml_sigma_Hz[:, self.Ny-self.pml_layers:, :] += sigma_profile_m.reshape(1, -1, 1)

        print("PML conductivity arrays initialized (pml_sigma_Ex, Ey, Ez, Hx, Hy, Hz).")
        # For a split-field PML (Berenger's original), auxiliary Psi fields would be initialized here:
        # self.psi_Ex_y = torch.zeros_like(self.Ex)
        # self.psi_Ex_z = torch.zeros_like(self.Ex)
        # ... and so on for all 12 split components.
        # For CPML, auxiliary fields might be needed for memory efficiency with kappa and alpha.
        # For now, this simplified sigma-based approach is a starting point.

    def apply_pml(self):
        """
        Applies PML boundary conditions.
        This is a placeholder for the actual PML update logic.
        A full PML implementation would involve modifying the E and H field
        updates in the PML regions, potentially using auxiliary Psi fields
        (for Berenger PML) or modified coefficients (for CPML).
        """
        if self.pml_layers == 0:
            return

        # print("PML updates would be applied here.")
        # In a full Berenger PML:
        # 1. Update Psi_H fields based on E-field curls.
        # 2. Update H-fields using Psi_H fields and E-field curls.
        # 3. Update Psi_E fields based on H-field curls.
        # 4. Update E-fields using Psi_E fields and H-field curls.
        # This often means separate update equations or modified coefficients
        # for field components within the PML regions.

        # For a CPML-like approach using the pml_sigma arrays initialized:
        # The update_E_fields and update_H_fields would need to be modified
        # to use these pml_sigma values instead of or in addition to self.sigma.
        # For example, in update_E_fields:
        # Cb_x = (self.dt / (EPSILON0 * self.epsilon_r * self.cell_size)) / (1.0 + (self.pml_sigma_Ex + self.sigma) * self.dt / (2 * EPSILON0 * self.epsilon_r))
        # Ca_x = (1.0 - (self.pml_sigma_Ex + self.sigma) * self.dt / (2 * EPSILON0 * self.epsilon_r)) / (1.0 + (self.pml_sigma_Ex + self.sigma) * self.dt / (2 * EPSILON0 * self.epsilon_r))
        # self.Ex = Ca_x * self.Ex + Cb_x * (dHz_dy - dHy_dz)
        # And similarly for H updates with pml_sigma_Hx, etc.
        # This is a complex change to the main update equations, so for now, this method is a stub.
        pass # Placeholder, actual PML logic is complex.

    def extract_frequency_domain_fields(self, 
                                        time_domain_snapshots: list[torch.Tensor], 
                                        time_points: list[float], 
                                        target_frequency: float, 
                                        recording_interval_for_dt: int) -> torch.Tensor:
        """
        Extracts frequency-domain fields from time-domain snapshots using DFT.

        Args:
            time_domain_snapshots (list[torch.Tensor]): List of 3D field snapshots (Nx, Ny, Nz)
                                                        recorded at uniform time intervals.
                                                        Assumed to be on CPU.
            time_points (list[float]): List of time values for each snapshot.
            target_frequency (float): Frequency (Hz) for which to calculate the DFT.
            recording_interval_for_dt (int): The recording interval used during simulation,
                                             needed to calculate dt between snapshots.

        Returns:
            torch.Tensor: Complex-valued tensor (Nx, Ny, Nz) of the field component
                          at target_frequency. Dtype is torch.complex64.
        """
        if not time_domain_snapshots:
            raise ValueError("time_domain_snapshots list cannot be empty.")
        if len(time_domain_snapshots) != len(time_points):
            raise ValueError("Number of snapshots must match number of time points.")

        # Assume all snapshots have the same shape and are on CPU as per run_fdtd_simulation
        first_snapshot_shape = time_domain_snapshots[0].shape
        for snapshot in time_domain_snapshots:
            if snapshot.shape != first_snapshot_shape:
                raise ValueError("All snapshots must have the same spatial dimensions.")
            if snapshot.device.type != 'cpu':
                 # Or move to CPU: snapshot = snapshot.cpu()
                raise ValueError("Snapshots are expected to be on CPU for stacking.")

        # Stack snapshots: (num_recorded_steps, Nx, Ny, Nz)
        # Ensure they are on the correct device for DFT computation (simulator's device)
        stacked_snapshots = torch.stack(time_domain_snapshots, dim=0).to(device=self.device, dtype=self.dtype)

        # Calculate time step between snapshots
        dt_snapshots = self.dt.item() * recording_interval_for_dt
        
        # Prepare time points tensor for broadcasting with snapshots
        # Time points should also be on the computation device
        time_points_tensor = torch.tensor(time_points, device=self.device, dtype=self.dtype)
        
        # Ensure complex dtype for DFT calculation right from the start
        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        stacked_snapshots_complex = stacked_snapshots.to(complex_dtype)
        time_points_complex = time_points_tensor.to(complex_dtype)

        # Reshape time_points_complex for broadcasting: (num_recorded_steps, 1, 1, 1)
        time_points_reshaped = time_points_complex.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Calculate exponential term for DFT: exp(-j * 2 * pi * f * t)
        # The negative sign is conventional for forward DFT (analysis)
        exp_term = torch.exp(-1j * 2 * torch.pi * target_frequency * time_points_reshaped)
        
        # Perform DFT sum: sum(X(t) * exp_term) * dt
        # Ensure exp_term is broadcastable with stacked_snapshots_complex
        field_freq_domain = torch.sum(stacked_snapshots_complex * exp_term, dim=0) * dt_snapshots
        
        print(f"Extracted frequency domain field for {target_frequency:.2e} Hz. Output shape: {field_freq_domain.shape}, dtype: {field_freq_domain.dtype}")
        return field_freq_domain

    def calculate_b1_plus_minus(self, Hx_freq: torch.Tensor, Hy_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates B1+ and B1- maps from frequency-domain Hx and Hy components.

        Args:
            Hx_freq (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz) of frequency-domain Hx.
            Hy_freq (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz) of frequency-domain Hy.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - B1_plus_map (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz).
                - B1_minus_map (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz).
        
        Raises:
            ValueError: If Hx_freq and Hy_freq have different shapes or are not complex.
        """
        if Hx_freq.shape != Hy_freq.shape:
            raise ValueError("Hx_freq and Hy_freq must have the same shape.")
        if not (Hx_freq.is_complex() and Hy_freq.is_complex()):
            raise ValueError("Hx_freq and Hy_freq must be complex-valued tensors.")
        
        # Ensure mu_r is on the same device and compatible dtype
        # Hx_freq and Hy_freq are already on self.device from extract_frequency_domain_fields
        mu_r_compatible = self.mu_r.to(device=Hx_freq.device, dtype=Hx_freq.dtype.real_dtype)

        # Calculate Bx_freq and By_freq
        # MU0 is a float, mu_r_compatible is float, Hx_freq/Hy_freq are complex.
        # Resulting Bx_freq/By_freq will be complex.
        Bx_freq = MU0 * mu_r_compatible * Hx_freq
        By_freq = MU0 * mu_r_compatible * Hy_freq
        
        # Calculate B1+ and B1-
        # (Bx_freq + 1j * By_freq) will correctly result in a complex tensor
        B1_plus_map = 0.5 * (Bx_freq + 1j * By_freq)
        B1_minus_map = 0.5 * (Bx_freq - 1j * By_freq)
        
        print(f"B1+ map calculated. Shape: {B1_plus_map.shape}, Dtype: {B1_plus_map.dtype}")
        print(f"B1- map calculated. Shape: {B1_minus_map.shape}, Dtype: {B1_minus_map.dtype}")
        
        return B1_plus_map, B1_minus_map

    def calculate_sar(self, Ex_freq: torch.Tensor, Ey_freq: torch.Tensor, Ez_freq: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Specific Absorption Rate (SAR) map from frequency-domain E-fields.

        Args:
            Ex_freq (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz) of frequency-domain Ex.
            Ey_freq (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz) of frequency-domain Ey.
            Ez_freq (torch.Tensor): Complex-valued tensor (Nx, Ny, Nz) of frequency-domain Ez.

        Returns:
            torch.Tensor: Real-valued tensor (Nx, Ny, Nz) of SAR in W/kg.
        
        Raises:
            ValueError: If E-field components have different shapes or are not complex.
        """
        if not (Ex_freq.shape == Ey_freq.shape == Ez_freq.shape):
            raise ValueError("Ex_freq, Ey_freq, and Ez_freq must have the same shape.")
        if not (Ex_freq.is_complex() and Ey_freq.is_complex() and Ez_freq.is_complex()):
            raise ValueError("E-field components must be complex-valued tensors.")

        # Ensure sigma and rho are on the correct device and compatible dtype
        sigma_compatible = self.sigma.to(device=Ex_freq.device, dtype=Ex_freq.dtype.real_dtype)
        rho_compatible = self.rho.to(device=Ex_freq.device, dtype=Ex_freq.dtype.real_dtype)
        
        # Clamp rho to prevent division by zero (e.g., in air regions if rho was 0)
        safe_rho = torch.clamp(rho_compatible, min=1e-3) # Min density of 1g/L

        # Calculate E_magnitude_squared: |Ex|^2 + |Ey|^2 + |Ez|^2
        E_mag_sq = Ex_freq.abs()**2 + Ey_freq.abs()**2 + Ez_freq.abs()**2
        
        # SAR = sigma * |E|^2 / (2 * rho)
        sar_map = (sigma_compatible * E_mag_sq) / (2 * safe_rho)
        
        print(f"SAR map calculated. Shape: {sar_map.shape}, Dtype: {sar_map.dtype}")
        return sar_map

    def calculate_s11(self, 
                        incident_voltage_time_series: torch.Tensor, 
                        reflected_voltage_time_series: torch.Tensor, 
                        time_points: torch.Tensor, 
                        target_frequency: float, 
                        reference_impedance: float, # Currently unused, but good for future
                        recording_interval_for_dt: int) -> torch.Tensor:
        """
        Calculates the S11 parameter from time series of incident and reflected voltages.

        Args:
            incident_voltage_time_series (torch.Tensor): 1D tensor of incident voltage.
            reflected_voltage_time_series (torch.Tensor): 1D tensor of reflected voltage.
            time_points (torch.Tensor): 1D tensor of time values for the voltage series.
            target_frequency (float): Frequency (Hz) for S11 calculation.
            reference_impedance (float): Reference impedance (e.g., 50 Ohms). Unused in current
                                         S11 = V_ref/V_inc definition but kept for completeness.
            recording_interval_for_dt (int): Recording interval for dt_snapshots.

        Returns:
            torch.Tensor: Complex scalar tensor representing S11 at target_frequency.
        
        Raises:
            ValueError: If input tensor shapes are incompatible or lengths do not match.
        """
        if not (incident_voltage_time_series.ndim == 1 and 
                reflected_voltage_time_series.ndim == 1 and 
                time_points.ndim == 1):
            raise ValueError("All input tensors must be 1D.")
        if not (len(incident_voltage_time_series) == len(reflected_voltage_time_series) == len(time_points)):
            raise ValueError("All input tensors must have the same length.")
        if len(time_points) == 0:
            raise ValueError("Input time_points tensor cannot be empty.")

        # Determine complex dtype based on simulator's float dtype
        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128

        # Move inputs to the correct device and ensure correct (complex) dtype for DFT
        time_points_dev = time_points.to(device=self.device, dtype=self.dtype).to(complex_dtype)
        incident_voltage_ts_dev = incident_voltage_time_series.to(device=self.device, dtype=self.dtype).to(complex_dtype)
        reflected_voltage_ts_dev = reflected_voltage_time_series.to(device=self.device, dtype=self.dtype).to(complex_dtype)
        
        dt_snapshots = self.dt.item() * recording_interval_for_dt
        
        # Calculate exponential term for DFT: exp(-j * 2 * pi * f * t)
        exp_term = torch.exp(-1j * 2 * torch.pi * target_frequency * time_points_dev)
        
        # Perform DFT for incident and reflected voltages
        V_inc_freq = torch.sum(incident_voltage_ts_dev * exp_term) * dt_snapshots
        V_ref_freq = torch.sum(reflected_voltage_ts_dev * exp_term) * dt_snapshots
        
        # S11 = V_ref_freq / V_inc_freq
        if torch.abs(V_inc_freq) < 1e-12: # Avoid division by zero or very small numbers
            print("Warning: Magnitude of incident voltage in frequency domain is very small.")
            # Depending on desired behavior, could return NaN, raise error, or return a specific value
            return torch.tensor(float('nan'), dtype=complex_dtype, device=self.device) 
            
        s11 = V_ref_freq / V_inc_freq
        
        print(f"S11 at {target_frequency:.2e} Hz calculated: {s11.item()}")
        return s11

    def run_fdtd_simulation(self, num_timesteps: int, source_config: dict, recording_interval: int = 0):
        """
        Runs the FDTD simulation for a specified number of time steps.

        Args:
            num_timesteps (int): Total number of time steps to simulate.
            source_config (dict): Configuration for the source term, passed to apply_source.
                                  Example: {'location_indices': (cx,cy,cz), 
                                            'field_component': 'Ez', 
                                            'waveform_type': 'gaussian_pulse', 
                                            'frequency': 300e6, 
                                            'amplitude': 1.0, 
                                            'waveform_params': {'t0': 30e-9, 'tau': 10e-9}}
            recording_interval (int, optional): Interval for recording field snapshots.
                                                If 0, no snapshots are recorded beyond final state.
                                                Defaults to 0.

        Returns:
            dict: A dictionary containing recorded data:
                  {'time_steps': list_of_times, 
                   'field_snapshots': {'Ex': [], 'Ey': [], 'Ez': [], 
                                       'Hx': [], 'Hy': []}}
        """
        recorded_data = {'time_steps': [], 
                         'field_snapshots': {'Ex': [], 'Ey': [], 'Ez': [], 
                                             'Hx': [], 'Hy': []}}
        
        print(f"Starting FDTD simulation for {num_timesteps} time steps...")
        print(f"Source Config: {source_config}")
        if recording_interval > 0:
            print(f"Recording Ex, Ey, Ez, Hx, Hy field snapshots every {recording_interval} time steps.")

        # Progress printing setup
        progress_interval = max(1, num_timesteps // 10) # Print progress roughly 10 times

        for n in range(num_timesteps):
            current_time = n * self.dt.item() # .item() to get float from tensor

            # Update H fields (half step)
            self.update_H_fields()
            
            # Update E fields (full step)
            self.update_E_fields()
            
            # Apply source
            # Unpack source_config, but override 'time' with the current simulation time
            # apply_source_args = {**source_config, 'time': current_time} # This is also valid
            # self.apply_source(**apply_source_args)
            self.apply_source(
                time=current_time,
                location_indices=source_config['location_indices'],
                field_component=source_config['field_component'],
                waveform_type=source_config['waveform_type'],
                frequency=source_config['frequency'],
                amplitude=source_config['amplitude'],
                waveform_params=source_config.get('waveform_params') # Use .get for optional key
            )
            
            # Apply PML (currently a stub)
            self.apply_pml()
            
            # Record snapshots
            if recording_interval > 0 and (n + 1) % recording_interval == 0:
                recorded_data['time_steps'].append(current_time)
                # Store a copy on CPU for easier access later
                recorded_data['field_snapshots']['Ex'].append(self.Ex.clone().cpu())
                recorded_data['field_snapshots']['Ey'].append(self.Ey.clone().cpu())
                recorded_data['field_snapshots']['Ez'].append(self.Ez.clone().cpu())
                recorded_data['field_snapshots']['Hx'].append(self.Hx.clone().cpu())
                recorded_data['field_snapshots']['Hy'].append(self.Hy.clone().cpu())
            
            # Print progress
            if (n + 1) % progress_interval == 0:
                print(f"Simulation progress: {(n + 1) / num_timesteps * 100:.0f}% completed ({n + 1}/{num_timesteps} steps).")

        print(f"FDTD simulation completed after {num_timesteps} time steps (Total time: {num_timesteps * self.dt.item():.2e} s).")
        return recorded_data

    def update_H_fields(self):
        """
        Updates H-fields for one time step using current E-fields.
        Assumes periodic boundary conditions due to torch.roll.
        """
        dt_mu = self.dt / (MU0 * self.mu_r) # self.mu_r is a tensor (Nx, Ny, Nz)

        # Derivatives for Hx update
        dEz_dy = (torch.roll(self.Ez, shifts=-1, dims=1) - self.Ez) / self.cell_size
        dEy_dz = (torch.roll(self.Ey, shifts=-1, dims=2) - self.Ey) / self.cell_size
        self.Hx = self.Hx - dt_mu * (dEz_dy - dEy_dz)

        # Derivatives for Hy update
        dEx_dz = (torch.roll(self.Ex, shifts=-1, dims=2) - self.Ex) / self.cell_size
        dEz_dx = (torch.roll(self.Ez, shifts=-1, dims=0) - self.Ez) / self.cell_size
        self.Hy = self.Hy - dt_mu * (dEx_dz - dEz_dx)

        # Derivatives for Hz update
        dEy_dx = (torch.roll(self.Ey, shifts=-1, dims=0) - self.Ey) / self.cell_size
        dEx_dy = (torch.roll(self.Ex, shifts=-1, dims=1) - self.Ex) / self.cell_size
        self.Hz = self.Hz - dt_mu * (dEy_dx - dEx_dy)

    def update_E_fields(self):
        """
        Updates E-fields for one time step using current H-fields and
        incorporating conductive losses.
        Assumes periodic boundary conditions due to torch.roll.
        """
        # Calculate coefficients for E-field update incorporating conductivity
        # Clamp epsilon_r to prevent division by zero if it's somehow zero anywhere.
        safe_epsilon_r = torch.clamp(self.epsilon_r, min=1e-9) 
        
        sigma_dt_eps = self.sigma * self.dt / (2 * EPSILON0 * safe_epsilon_r)
        Ca = (1.0 - sigma_dt_eps) / (1.0 + sigma_dt_eps)
        Cb = (self.dt / (EPSILON0 * safe_epsilon_r * self.cell_size)) / (1.0 + sigma_dt_eps)
        # Note: The self.cell_size was missing in Cb in the prompt, added it here as it's (dt / (epsilon * ds))

        # Derivatives of H for E-field update
        # Using H_current - H_previous_cell (forward difference in space for H)
        # This corresponds to ( H(i) - H(i-1) ) / ds
        dHz_dy = (self.Hz - torch.roll(self.Hz, shifts=1, dims=1)) / self.cell_size
        dHy_dz = (self.Hy - torch.roll(self.Hy, shifts=1, dims=2)) / self.cell_size
        curl_H_x = dHz_dy - dHy_dz
        self.Ex = Ca * self.Ex + Cb * curl_H_x

        dHx_dz = (self.Hx - torch.roll(self.Hx, shifts=1, dims=2)) / self.cell_size
        dHz_dx = (self.Hz - torch.roll(self.Hz, shifts=1, dims=0)) / self.cell_size
        curl_H_y = dHx_dz - dHz_dx
        self.Ey = Ca * self.Ey + Cb * curl_H_y

        dHy_dx = (self.Hy - torch.roll(self.Hy, shifts=1, dims=0)) / self.cell_size
        dHx_dy = (self.Hx - torch.roll(self.Hx, shifts=1, dims=1)) / self.cell_size
        curl_H_z = dHy_dx - dHx_dy
        self.Ez = Ca * self.Ez + Cb * curl_H_z


if __name__ == '__main__':
    # Example Usage:
    sim_dims = (20, 20, 20) # 20x20x20 cells
    sim_cell_size = 0.005   # 5 mm
    
    # Check if CUDA is available and use it, otherwise use CPU
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        d_type = torch.float32 # CUDA typically prefers float32
        print("CUDA device selected.")
    else:
        dev = torch.device('cpu')
        d_type = torch.float64 # CPU can handle float64 for potentially better precision
        print("CPU device selected.")
    
    pml_thickness = 10 # Number of PML layers
    simulator = FDTDSimulator(
        dimensions=sim_dims, 
        cell_size=sim_cell_size, 
        pml_layers=pml_thickness, # Test PML initialization
        dtype=d_type, 
        device=dev
    )
    
    print(f"Ex field component tensor device: {simulator.Ex.device}")
    print(f"Ex field component tensor dtype: {simulator.Ex.dtype}")
    print(f"Calculated time step dt: {simulator.dt.item():.4e} s")

    # Example of using load_material_properties
    # Create a phantom: central cube of muscle (index 2) in air (index 0)
    phantom = torch.zeros(sim_dims, dtype=torch.long, device=dev) 
    phantom[5:15, 5:15, 5:15] = 2 # Muscle
    
    # For testing, add an unknown material index if you want to test error handling
    # phantom[0,0,0] = 99 

    try:
        simulator.load_material_properties(phantom, MATERIAL_DATABASE)
        print("Successfully loaded material properties from phantom.")
        # You can verify by printing a slice of epsilon_r, for example:
        # print("Epsilon_r slice (center Z):")
        # print(simulator.epsilon_r[:, :, sim_dims[2]//2])
    except ValueError as e:
        print(f"Error loading material properties: {e}")

    # Example of applying a source
    source_loc = (sim_dims[0]//2, sim_dims[1]//2, sim_dims[2]//2) # Center of the grid
    source_comp = 'Ez'
    source_freq = 100e6 # 100 MHz
    source_amp = 1.0 # V/m
    
    # Apply Gaussian pulse at t=0 (will be small value) and t=t0 (peak)
    t0_gauss = 3.0 * (1.0 / (source_freq * 0.4 * torch.pi)) # Approximate t0 based on defaults
    
    print(f"\nApplying Gaussian pulse source to {source_comp} at {source_loc}:")
    # Time step 0 (time = 0.0)
    simulator.apply_source(
        time=0.0, 
        location_indices=source_loc, 
        field_component=source_comp,
        waveform_type='gaussian_pulse',
        frequency=source_freq,
        amplitude=source_amp,
        waveform_params={'t0': t0_gauss, 'tau': t0_gauss / 3.0} # Example params
    )
    print(f"Ez at source after t=0.0: {simulator.Ez[source_loc].item():.3e} V/m")

    # Time approx t0 (time = t0_gauss)
    simulator.apply_source(
        time=t0_gauss.item(), # .item() if t0_gauss is a tensor
        location_indices=source_loc,
        field_component=source_comp,
        waveform_type='gaussian_pulse',
        frequency=source_freq,
        amplitude=source_amp,
        waveform_params={'t0': t0_gauss, 'tau': t0_gauss / 3.0}
    )
    print(f"Ez at source after t={t0_gauss.item():.2e}s (peak): {simulator.Ez[source_loc].item():.3e} V/m")

    # Apply Sine burst at t = 1/(4f) (peak of sine)
    time_sine_peak = 1.0 / (4.0 * source_freq)
    print(f"\nApplying Sine burst source to {source_comp} at {source_loc}:")
    simulator.apply_source(
        time=time_sine_peak,
        location_indices=source_loc,
        field_component=source_comp,
        waveform_type='sine_burst',
        frequency=source_freq,
        amplitude=source_amp
    )
    # Note: field is cumulative, so this adds to the Gaussian pulse value if same component/location
    print(f"Ez at source after t={time_sine_peak:.2e}s (sine peak): {simulator.Ez[source_loc].item():.3e} V/m")

    # Demonstrate field updates (a few steps)
    print("\nDemonstrating field updates (few steps):")
    # Store initial Ez at source to see changes
    Ez_initial_at_source = simulator.Ez[source_loc].item()
    print(f"Initial Ez at source before updates: {Ez_initial_at_source:.3e} V/m")

    num_update_steps = 5
    for step in range(num_update_steps):
        simulator.update_H_fields()
        # Here one might apply E-field sources if they are time-varying within these steps
        simulator.update_E_fields()
        # Here one might apply H-field sources or other operations
        
        # For simplicity, let's re-apply the sine source continuously to Ez to see evolution
        # This is just for quick demo; proper time loop would handle time advancement for source
        current_sim_time_in_loop = (step + 1) * simulator.dt.item() # Approximate time
        # Apply PML (currently a stub)
        simulator.apply_pml()
        
        simulator.apply_source( 
            time=current_sim_time_in_loop, 
            location_indices=source_loc, 
            field_component=source_comp, # Ez
            waveform_type='sine_burst', # Keep applying sine
            frequency=source_freq, 
            amplitude=source_amp
        )
        print(f"Step {step+1}: Ez at source: {simulator.Ez[source_loc].item():.3e} V/m, Hx at source: {simulator.Hx[source_loc].item():.3e} A/m")

    # Example of running the full simulation loop
    print("\nRunning full FDTD simulation loop example:")
    total_timesteps = 200 # Short simulation for example
    snapshot_interval = 50 # Record every 50 steps
    
    # Define a source configuration (can be different from the previous one)
    sim_source_config = {
        'location_indices': (sim_dims[0]//2, sim_dims[1]//2, sim_dims[2]//2),
        'field_component': 'Ez',
        'waveform_type': 'gaussian_pulse', # Using Gaussian for this run
        'frequency': 150e6, # 150 MHz
        'amplitude': 1.0,
        'waveform_params': {'t0': 20 * simulator.dt.item(), 'tau': 10 * simulator.dt.item()} # Example t0 and tau based on dt
    }
    
    # Reset fields before new run if needed (or use a new simulator instance)
    # simulator.initialize_fdtd_grid() # This would reset fields and material properties to default
    # simulator.load_material_properties(phantom, MATERIAL_DATABASE) # Reload materials if reset

    recorded_simulation_data = simulator.run_fdtd_simulation(
        num_timesteps=total_timesteps,
        source_config=sim_source_config,
        recording_interval=snapshot_interval
    )

    if recorded_simulation_data['time_steps']:
        for component_key in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy']:
            num_snapshots = len(recorded_simulation_data['field_snapshots'].get(component_key, []))
            if num_snapshots > 0:
                print(f"Simulation recorded {num_snapshots} {component_key} snapshots.")
        print(f"Time points of snapshots: {recorded_simulation_data['time_steps']}")

        # Demonstrate frequency domain extraction
        if snapshot_interval > 0:
            target_freq_example = sim_source_config['frequency'] # Use source frequency
            center_idx_x = sim_dims[0] // 2
            center_idx_y = sim_dims[1] // 2
            center_idx_z = sim_dims[2] // 2
            
            freq_domain_fields_dict = {}

            for component_key in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy']:
                snapshots = recorded_simulation_data['field_snapshots'].get(component_key, [])
                if not snapshots:
                    print(f"No snapshots found for component {component_key}, skipping DFT.")
                    continue

                print(f"\nExtracting frequency domain fields from recorded {component_key} snapshots:")
                try:
                    freq_domain_field = simulator.extract_frequency_domain_fields(
                        time_domain_snapshots=snapshots,
                        time_points=recorded_simulation_data['time_steps'],
                        target_frequency=target_freq_example,
                        recording_interval_for_dt=snapshot_interval
                    )
                    freq_domain_fields_dict[component_key] = freq_domain_field
                    print(f"Frequency domain {component_key} field extracted. Shape: {freq_domain_field.shape}, Dtype: {freq_domain_field.dtype}")
                    mag_center = torch.abs(freq_domain_field[center_idx_x, center_idx_y, center_idx_z])
                    print(f"Magnitude of {component_key}(f={target_freq_example:.2e} Hz) at grid center: {mag_center.item():.3e}")
                except ValueError as e:
                    print(f"Error extracting frequency domain {component_key} fields: {e}")
            
            # Demonstrate B1+ and B1- calculation
            if 'Hx' in freq_domain_fields_dict and 'Hy' in freq_domain_fields_dict:
                print("\nCalculating B1+ and B1- maps:")
                try:
                    B1_plus, B1_minus = simulator.calculate_b1_plus_minus(
                        freq_domain_fields_dict['Hx'],
                        freq_domain_fields_dict['Hy']
                    )
                    # print(f"B1_plus map calculated. Shape: {B1_plus.shape}, Dtype: {B1_plus.dtype}")
                    # print(f"B1_minus map calculated. Shape: {B1_minus.shape}, Dtype: {B1_minus.dtype}")
                    mag_B1_plus_center = torch.abs(B1_plus[center_idx_x, center_idx_y, center_idx_z])
                    print(f"Magnitude of B1+ at grid center: {mag_B1_plus_center.item():.3e}")
                except ValueError as e:
                    print(f"Error calculating B1+ and B1- maps: {e}")
            else:
                print("\nSkipping B1+ / B1- calculation as Hx_freq or Hy_freq is missing.")

            # Demonstrate SAR calculation
            if ('Ex' in freq_domain_fields_dict and 
                'Ey' in freq_domain_fields_dict and 
                'Ez' in freq_domain_fields_dict):
                print("\nCalculating SAR map:")
                try:
                    sar_map = simulator.calculate_sar(
                        freq_domain_fields_dict['Ex'],
                        freq_domain_fields_dict['Ey'],
                        freq_domain_fields_dict['Ez']
                    )
                    # print(f"SAR map calculated. Shape: {sar_map.shape}, Dtype: {sar_map.dtype}")
                    sar_center = sar_map[center_idx_x, center_idx_y, center_idx_z]
                    print(f"SAR at grid center: {sar_center.item():.3e} W/kg")
                except ValueError as e:
                    print(f"Error calculating SAR map: {e}")
            else:
                print("\nSkipping SAR calculation as Ex_freq, Ey_freq, or Ez_freq is missing.")
            
            # Demonstrate S11 calculation (with dummy data)
            print("\nCalculating S11 parameter (with dummy voltage data):")
            if recorded_simulation_data['time_steps']:
                # Create dummy incident and reflected voltage time series
                # These should ideally come from the simulation's source port monitoring
                dummy_time_points_tensor = torch.tensor(recorded_simulation_data['time_steps'], device=dev, dtype=d_type)
                
                # Incident: mimics the source waveform if it were a voltage source
                # For simplicity, let's use the target_frequency for a sine wave
                dummy_incident_voltage = torch.cos(2 * torch.pi * target_freq_example * dummy_time_points_tensor)
                
                # Reflected: smaller amplitude and phase shifted
                dummy_reflected_voltage = 0.2 * torch.cos(2 * torch.pi * target_freq_example * dummy_time_points_tensor - torch.pi / 3) # e.g. 20% reflection, 60deg phase shift
                
                print(f"Using dummy incident voltage (len: {len(dummy_incident_voltage)}) and reflected voltage (len: {len(dummy_reflected_voltage)}) series.")
                print("Note: These are placeholder data for S11 calculation demonstration.")

                try:
                    s11_value = simulator.calculate_s11(
                        incident_voltage_time_series=dummy_incident_voltage,
                        reflected_voltage_time_series=dummy_reflected_voltage,
                        time_points=dummy_time_points_tensor, # Use the tensor version of time_points
                        target_frequency=target_freq_example,
                        reference_impedance=50.0, # Standard reference impedance
                        recording_interval_for_dt=snapshot_interval
                    )
                    print(f"Calculated S11 at {target_freq_example:.2e} Hz: {s11_value.item()}")
                    print(f"S11 Magnitude: {torch.abs(s11_value).item():.4f}, S11 Phase (degrees): {torch.angle(s11_value).item() * 180 / np.pi:.2f}")
                except ValueError as e:
                    print(f"Error calculating S11: {e}")
            else:
                print("Skipping S11 calculation as no time points were recorded.")
                
    else:
        print("No snapshots were recorded during the simulation.")


def plot_fdtd_results_slice(data_map: torch.Tensor, 
                            x_coords: torch.Tensor, 
                            y_coords: torch.Tensor, 
                            z_coords: torch.Tensor, 
                            slice_axis: str, 
                            slice_index: int, 
                            quantity_name: str, 
                            ax=None, 
                            show_plot: bool = True, 
                            cmap: str = 'viridis'):
    """
    Visualizes a 2D slice of a 3D FDTD-derived data map.

    Args:
        data_map (torch.Tensor): 3D PyTorch tensor (Nx, Ny, Nz) containing the data to plot.
                                 Should be real-valued (e.g., magnitude or a real component).
        x_coords (torch.Tensor): 1D PyTorch tensor for the x-coordinates of the grid.
        y_coords (torch.Tensor): 1D PyTorch tensor for the y-coordinates of the grid.
        z_coords (torch.Tensor): 1D PyTorch tensor for the z-coordinates of the grid.
        slice_axis (str): The axis perpendicular to the slice. One of 'x', 'y', 'z'.
        slice_index (int): The index along `slice_axis` where the slice is taken.
        quantity_name (str): Descriptive name for the data (e.g., "B1+ Magnitude (uT)", "SAR (W/kg)").
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object. If None, new figure/axes created.
        show_plot (bool, optional): If True and ax is None, plt.show() is called. Defaults to True.
        cmap (str, optional): Colormap for imshow. Defaults to 'viridis'.

    Raises:
        ValueError: If slice_axis is invalid or slice_index is out of bounds.
    """
    if slice_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")

    # Convert to NumPy for plotting
    data_map_np = data_map.detach().cpu().numpy()
    x_coords_np = x_coords.detach().cpu().numpy()
    y_coords_np = y_coords.detach().cpu().numpy()
    z_coords_np = z_coords.detach().cpu().numpy()

    slice_data = None
    plot_x_axis_coords, plot_y_axis_coords = None, None
    x_axis_label, y_axis_label = "", ""
    slice_coord_val_str = ""

    if slice_axis == 'x':
        if not (0 <= slice_index < data_map_np.shape[0]):
            raise ValueError(f"slice_index {slice_index} out of bounds for x-axis (shape {data_map_np.shape[0]}).")
        slice_data = data_map_np[slice_index, :, :] # Slice is (Ny, Nz)
        plot_x_axis_coords, plot_y_axis_coords = y_coords_np, z_coords_np
        x_axis_label, y_axis_label = "Y-coordinate (m)", "Z-coordinate (m)"
        slice_coord_val_str = f"X = {x_coords_np[slice_index]:.3e} m"
    elif slice_axis == 'y':
        if not (0 <= slice_index < data_map_np.shape[1]):
            raise ValueError(f"slice_index {slice_index} out of bounds for y-axis (shape {data_map_np.shape[1]}).")
        slice_data = data_map_np[:, slice_index, :] # Slice is (Nx, Nz)
        plot_x_axis_coords, plot_y_axis_coords = x_coords_np, z_coords_np
        x_axis_label, y_axis_label = "X-coordinate (m)", "Z-coordinate (m)"
        slice_coord_val_str = f"Y = {y_coords_np[slice_index]:.3e} m"
    elif slice_axis == 'z':
        if not (0 <= slice_index < data_map_np.shape[2]):
            raise ValueError(f"slice_index {slice_index} out of bounds for z-axis (shape {data_map_np.shape[2]}).")
        slice_data = data_map_np[:, :, slice_index] # Slice is (Nx, Ny)
        plot_x_axis_coords, plot_y_axis_coords = x_coords_np, y_coords_np
        x_axis_label, y_axis_label = "X-coordinate (m)", "Y-coordinate (m)"
        slice_coord_val_str = f"Z = {z_coords_np[slice_index]:.3e} m"

    # Create plot
    manage_plot_display = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        manage_plot_display = True
    
    # imshow expects data as (row, col).
    # For slice_axis='z', slice_data is (Nx, Ny). plot_x_axis_coords=x_coords, plot_y_axis_coords=y_coords.
    # We want x_coords on horizontal (columns) and y_coords on vertical (rows).
    # So, imshow(data.T) with origin='lower' and extent matching [x_min, x_max, y_min, y_max].
    img_data = slice_data.T 
    extent = [plot_x_axis_coords[0], plot_x_axis_coords[-1], 
              plot_y_axis_coords[0], plot_y_axis_coords[-1]]

    im = ax.imshow(img_data, aspect='auto', origin='lower', extent=extent, cmap=cmap)
    
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    title_str = f"{quantity_name} | Slice at {slice_coord_val_str}"
    ax.set_title(title_str)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=quantity_name)
    
    if manage_plot_display and show_plot:
        plt.show()
    
    return ax
