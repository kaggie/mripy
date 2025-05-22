import torch
import numpy as np # Still needed for np.pi

class VoltageSource:
    """
    Represents a time-varying voltage source (applied to an E-field component)
    for FDTD simulations.
    """

    def __init__(self, location_cells, axis, amplitude, frequency,
                 waveform_type='sine_burst', 
                 pulse_center_time_steps=50, 
                 pulse_width_time_steps=20,
                 sine_start_time_steps=0):
        """
        Initializes the VoltageSource.

        Args:
            location_cells (tuple): Tuple of 3 integers (ix, iy, iz) representing the
                                    cell index where the source is applied. The specific
                                    E-field component location within or on the boundary
                                    of this cell depends on the 'axis'.
            axis (str): Specifies the E-field component to drive: 'x', 'y', or 'z'.
                        Corresponds to Ex, Ey, or Ez.
            amplitude (float): Peak amplitude of the source.
            frequency (float): Frequency of the sine wave component in Hz.
            waveform_type (str, optional): Type of waveform. Can be 'sine_burst'
                                           or 'gaussian_pulse'. Defaults to 'sine_burst'.
            pulse_center_time_steps (int, optional): For 'gaussian_pulse', the time step
                                                     index at which the pulse envelope peaks.
                                                     Defaults to 50.
            pulse_width_time_steps (int, optional): For 'gaussian_pulse', related to the
                                                    standard deviation of the Gaussian
                                                    envelope (e.g., width = 2*sigma_time_steps).
                                                    Defaults to 20.
            sine_start_time_steps (int, optional): For 'sine_burst', the time step index
                                                   at which the sine wave begins. Defaults to 0.
        
        Raises:
            ValueError: If axis is not 'x', 'y', or 'z', or if waveform_type is unknown.
        """
        if not isinstance(location_cells, tuple) or len(location_cells) != 3 or \
           not all(isinstance(idx, int) for idx in location_cells):
            raise ValueError("location_cells must be a tuple of 3 integers (ix, iy, iz).")
        
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis must be one of 'x', 'y', or 'z'.")
        
        if waveform_type not in ['sine_burst', 'gaussian_pulse']:
            raise ValueError(f"Unknown waveform_type: {waveform_type}. Must be 'sine_burst' or 'gaussian_pulse'.")

        self.location_cells = location_cells
        self.axis = axis
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.waveform_type = waveform_type
        self.pulse_center_time_steps = int(pulse_center_time_steps)
        self.pulse_width_time_steps = int(pulse_width_time_steps)
        self.sine_start_time_steps = int(sine_start_time_steps)

        self.omega = 2 * np.pi * self.frequency # np.pi is fine here as omega is a float attribute
        
        print(f"VoltageSource initialized: Axis {self.axis} at cell {self.location_cells}, "
              f"Freq {self.frequency:.2e} Hz, Type '{self.waveform_type}'")

    def get_value(self, time_step_index, dt):
        """
        Calculates the source value at a given time step.

        Args:
            time_step_index (int): Current integer time step in the simulation.
            dt (float): Duration of a single time step (from FDTDGrid.dt).

        Returns:
            torch.Tensor: The source value as a PyTorch scalar tensor (float32).
        """
        t_float = float(time_step_index * dt) # Current time in seconds as float
        # Convert to tensor for torch operations if needed, or keep as float for np math then convert
        t = torch.tensor(t_float, dtype=torch.float32)
        
        value = torch.tensor(0.0, dtype=torch.float32)

        if self.waveform_type == 'gaussian_pulse':
            t0_float = float(self.pulse_center_time_steps * dt)
            sigma_t_float = (float(self.pulse_width_time_steps) / 2.0) * dt 
            if sigma_t_float == 0: sigma_t_float = 1e-9 * dt # Avoid division by zero for very narrow pulse
            
            t0 = torch.tensor(t0_float, dtype=torch.float32)
            sigma_t = torch.tensor(sigma_t_float, dtype=torch.float32)

            gaussian_envelope = torch.exp(-0.5 * ((t - t0) / sigma_t)**2)
            # Carrier sine wave, shifted to be centered with the Gaussian peak
            sine_carrier = torch.sin(self.omega * (t - t0)) 
            value = self.amplitude * gaussian_envelope * sine_carrier
        
        elif self.waveform_type == 'sine_burst':
            if time_step_index >= self.sine_start_time_steps:
                # Effective time for the sine wave, starting from zero after delay
                t_eff_float = (time_step_index - self.sine_start_time_steps) * dt
                t_eff = torch.tensor(t_eff_float, dtype=torch.float32)
                value = self.amplitude * torch.sin(self.omega * t_eff)
            # value remains 0.0 if before start time (initialized as such)
        else:
            # This case should ideally not be reached due to __init__ check, but as a fallback:
            print(f"Warning: Unknown waveform type '{self.waveform_type}' in get_value.")
            # value remains 0.0
            
        return value

    def apply_at_time(self, grid, time_step_index):
        """
        Applies the source value to the E-field in the FDTD grid at the current time step.
        This typically means adding to the E-field component (hard source).

        Args:
            grid (FDTDGrid): The FDTDGrid instance.
            time_step_index (int): Current integer time step in the simulation.
        """
        source_value = self.get_value(time_step_index, grid.dt).to(grid.device)
        
        ix, iy, iz = self.location_cells

        # Important: Check bounds for location_cells against grid dimensions
        # Ex(Nx, Ny+1, Nz+1), Ey(Nx+1, Ny, Nz+1), Ez(Nx+1, Ny+1, Nz)
        try:
            if self.axis == 'x':
                if 0 <= ix < grid.Ex.shape[0] and \
                   0 <= iy < grid.Ex.shape[1] and \
                   0 <= iz < grid.Ex.shape[2]:
                    grid.Ex[ix, iy, iz] += source_value
                else:
                    print(f"Warning: Source location {self.location_cells} for Ex out of bounds {grid.Ex.shape}.")
            elif self.axis == 'y':
                if 0 <= ix < grid.Ey.shape[0] and \
                   0 <= iy < grid.Ey.shape[1] and \
                   0 <= iz < grid.Ey.shape[2]:
                    grid.Ey[ix, iy, iz] += source_value
                else:
                    print(f"Warning: Source location {self.location_cells} for Ey out of bounds {grid.Ey.shape}.")
            elif self.axis == 'z':
                if 0 <= ix < grid.Ez.shape[0] and \
                   0 <= iy < grid.Ez.shape[1] and \
                   0 <= iz < grid.Ez.shape[2]:
                    grid.Ez[ix, iy, iz] += source_value
                else:
                    print(f"Warning: Source location {self.location_cells} for Ez out of bounds {grid.Ez.shape}.")
        except IndexError: 
             print(f"Error (IndexError): Source location {self.location_cells} for axis '{self.axis}' is out of bounds "
                   f"for grid dimensions (Ex:{grid.Ex.shape}, Ey:{grid.Ey.shape}, Ez:{grid.Ez.shape}).")


if __name__ == '__main__':
    # --- Example Usage and Testing for VoltageSource ---
    print("\\n--- VoltageSource Test (Corrected get_value with torch) ---")

    class DummyFDTDGrid:
        def __init__(self, dt_val=1e-12, device_val=torch.device("cpu")):
            self.dt = dt_val
            self.device = device_val
            self.Ex = torch.zeros((5, 6, 6), device=device_val, dtype=torch.float32) 
            self.Ey = torch.zeros((6, 5, 6), device=device_val, dtype=torch.float32) 
            self.Ez = torch.zeros((6, 6, 5), device=device_val, dtype=torch.float32) 

    dummy_grid = DummyFDTDGrid()

    print("\\nTest 1: Sine Burst Source")
    source_sine = VoltageSource(location_cells=(2, 2, 2), axis='z', 
                                amplitude=1.0, frequency=1e9, waveform_type='sine_burst',
                                sine_start_time_steps=10)
    
    val_before_start = source_sine.get_value(time_step_index=5, dt=dummy_grid.dt)
    val_at_start = source_sine.get_value(time_step_index=10, dt=dummy_grid.dt)
    val_after_start = source_sine.get_value(time_step_index=11, dt=dummy_grid.dt)
    
    print(f"  Sine burst value at step 5 (before start): {val_before_start.item():.4f}")
    print(f"  Sine burst value at step 10 (at start): {val_at_start.item():.4f}")
    print(f"  Sine burst value at step 11 (after start): {val_after_start.item():.4f}")
    assert val_before_start.item() == 0.0, "Sine burst before start should be 0."
    assert abs(val_at_start.item()) < 1e-6, "Sine burst at start (t_eff=0) should be ~0."
    assert abs(val_after_start.item()) > 1e-6, "Sine burst after start should be non-zero."
    
    source_sine.apply_at_time(dummy_grid, time_step_index=11)
    print(f"  Ez at source location after apply_at_time (step 11): {dummy_grid.Ez[2,2,2].item():.4f}")
    assert abs(dummy_grid.Ez[2,2,2].item() - val_after_start.item()) < 1e-6, "Apply_at_time for sine did not match get_value."


    print("\\nTest 2: Gaussian Pulse Source")
    source_gauss = VoltageSource(location_cells=(1, 1, 1), axis='x',
                                 amplitude=2.0, frequency=5e9, waveform_type='gaussian_pulse',
                                 pulse_center_time_steps=30, pulse_width_time_steps=10)

    val_at_gauss_center = source_gauss.get_value(time_step_index=30, dt=dummy_grid.dt)
    print(f"  Gaussian pulse value at step 30 (center of envelope): {val_at_gauss_center.item():.4f}")
    assert abs(val_at_gauss_center.item()) < 1e-6, "Gaussian pulse at t=t0 (center of envelope) should be 0 due to sin(0)."

    val_off_gauss_center = source_gauss.get_value(time_step_index=32, dt=dummy_grid.dt)
    print(f"  Gaussian pulse value at step 32 (off-center): {val_off_gauss_center.item():.4f}")
    # Expected value based on manual calculation from previous test: ~0.1157
    # This requires torch.exp and torch.sin to produce values consistent with np.exp/np.sin
    assert abs(val_off_gauss_center.item() - 0.1157) < 0.01 , f"Gaussian pulse off-center value changed. Got {val_off_gauss_center.item()}"

    source_gauss.apply_at_time(dummy_grid, time_step_index=32)
    print(f"  Ex at source location after apply_at_time (step 32): {dummy_grid.Ex[1,1,1].item():.4f}")
    assert abs(dummy_grid.Ex[1,1,1].item() - val_off_gauss_center.item()) < 1e-6, "Apply_at_time for Gaussian did not match get_value."

    print("\\nTest 3 & 4 (Invalid params) already tested in previous run, covered by constructor checks.")
    print("\\nTest 5: Source location out of bounds (expect warning)")
    source_boundary_fail = VoltageSource(location_cells=(10,10,10), axis='x', amplitude=1, frequency=1e9)
    source_boundary_fail.apply_at_time(dummy_grid, 0)
    assert dummy_grid.Ex[0,0,0].item() == 0.0, "Field should not be modified if source is OOB."


    print("\\n--- VoltageSource Test (Corrected get_value with torch) Finished ---")

```
