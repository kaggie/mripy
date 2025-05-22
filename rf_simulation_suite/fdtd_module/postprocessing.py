import torch
import numpy as np # For np.pi
import glob
import os
import re # For robust filename parsing

def _extract_step_from_filename(filename, prefix="snapshot_step_", suffix=".pt"):
    """
    Extracts the time step number from a snapshot filename.
    Assumes filename format like 'prefixNUMBERsuffix'.
    """
    base_name = os.path.basename(filename)
    if not base_name.startswith(prefix) or not base_name.endswith(suffix):
        # print(f"Warning: Filename '{base_name}' does not match expected pattern '{prefix}*{suffix}'.")
        return None # Return None to filter out non-matching files
    
    try:
        number_str = base_name[len(prefix):-len(suffix)]
        return int(number_str)
    except ValueError:
        # print(f"Warning: Could not extract step number from '{base_name}'.")
        return None

def dft_snapshots(snapshot_dir, target_frequency, dt, 
                  snapshot_file_prefix="snapshot_step_", device=None):
    """
    Performs a Discrete Fourier Transform (DFT) on FDTD field snapshots
    stored in files to calculate frequency-domain fields at a target frequency.

    The DFT is calculated as X(f) = sum_k { x(t_k) * exp(-j * 2*pi*f*t_k) * dt_k }.
    Here, dt_k is the time step dt, and t_k is the time of the k-th snapshot.

    Args:
        snapshot_dir (str): Directory where snapshot .pt files are stored.
        target_frequency (float): The frequency (in Hz) at which to calculate
                                  the frequency-domain fields.
        dt (float): Time step duration from the FDTD simulation (in seconds).
                    This is the interval between snapshots if snapshots are saved every step,
                    or the simulation dt if t_value is derived from step number.
        snapshot_file_prefix (str, optional): The prefix of snapshot filenames.
                                              Defaults to "snapshot_step_".
                                              Example: "snapshot_step_100.pt"
        device (torch.device or str, optional): PyTorch device to perform calculations on
                                                (e.g., "cuda", "cpu"). If None, defaults to CPU
                                                for accumulation, then user can move result.
                                                Defaults to None.

    Returns:
        tuple: A tuple containing two dictionaries:
            - E_fields_freq_dft (dict): Complex-valued frequency-domain E-field components
                                        (e.g., {"Ex": tensor, "Ey": tensor, ...}).
            - H_fields_freq_dft (dict): Complex-valued frequency-domain H-field components
                                        (e.g., {"Hx": tensor, "Hy": tensor, ...}).
            Returns (None, None) if no suitable snapshot files are found or an error occurs.
            
    Note:
        Assumes snapshot filenames include the time step number K (e.g., "snapshot_step_K.pt"),
        and that this K represents that K simulation steps have completed. The time value `t_value`
        for the DFT is then calculated as `K * dt`.
    """
    if device is None:
        processing_device = torch.device("cpu") 
    elif isinstance(device, str):
        processing_device = torch.device(device)
    else:
        processing_device = device
    
    print(f"DFT Snapshots: Using device '{processing_device}' for DFT accumulation.")

    file_pattern = os.path.join(snapshot_dir, f"{snapshot_file_prefix}*.pt")
    snapshot_filepaths = glob.glob(file_pattern)

    if not snapshot_filepaths:
        print(f"Error: No snapshot files found in '{snapshot_dir}' with prefix '{snapshot_file_prefix}'.")
        return None, None

    # Create a list of (step_number, filepath) tuples for sorting
    parsed_files = []
    for fp in snapshot_filepaths:
        step = _extract_step_from_filename(fp, prefix=snapshot_file_prefix, suffix=".pt")
        if step is not None:
            parsed_files.append((step, fp))
        else:
            print(f"Warning: Skipping file with unparsable step number: {fp}")
            
    if not parsed_files:
        print(f"Error: No valid snapshot files (with parsable step numbers) found in '{snapshot_dir}'.")
        return None, None

    # Sort files by the extracted step number
    parsed_files.sort(key=lambda x: x[0])
    
    # Use only the sorted filepaths from here
    sorted_snapshot_filepaths = [fp for step, fp in parsed_files]

    print(f"Found {len(sorted_snapshot_filepaths)} valid snapshot files to process, sorted by step number.")

    E_fields_freq_dft = {}
    H_fields_freq_dft = {}
    first_snapshot_processed_successfully = False

    omega = 2 * np.pi * target_frequency

    for step_number, filepath in parsed_files: # Iterate through sorted (step, filepath)
        # t_value: Time at which the fields in this snapshot are defined.
        # If snapshot_step_K.pt contains fields after K steps, t_value = K * dt.
        t_value = float(step_number * dt)

        try:
            snapshot_data = torch.load(filepath, map_location='cpu') 
        except Exception as e:
            print(f"Error loading snapshot file '{filepath}': {e}. Skipping.")
            continue

        # DFT complex exponential term: exp(-j * omega * t) for forward DFT
        exp_term = torch.exp(-1j * torch.tensor(omega * t_value, dtype=torch.cfloat, device=processing_device))
        
        for field_name, field_time_tensor_cpu in snapshot_data.items():
            # Ensure field_time_tensor is float for multiplication with complex exp_term
            field_time_tensor = field_time_tensor_cpu.to(device=processing_device, dtype=torch.float32) 
            
            target_sum_dict = None
            if field_name.startswith('E'):
                target_sum_dict = E_fields_freq_dft
            elif field_name.startswith('H'):
                target_sum_dict = H_fields_freq_dft
            else:
                continue # Skip unknown field types

            if not first_snapshot_processed_successfully or field_name not in target_sum_dict:
                target_sum_dict[field_name] = torch.zeros_like(field_time_tensor, dtype=torch.cfloat, device=processing_device)
            
            try:
                target_sum_dict[field_name] += field_time_tensor * exp_term 
            except RuntimeError as e:
                print(f"Error during DFT accumulation for field {field_name} from {filepath}: {e}")
                continue # Skip this field if accumulation fails
        
        if snapshot_data: # If this snapshot had any processable data
            first_snapshot_processed_successfully = True

    if not first_snapshot_processed_successfully:
        print("Error: No valid snapshot data was successfully processed from any file.")
        return None, None

    # Complete the DFT integral approximation by multiplying by dt
    for field_name in E_fields_freq_dft:
        E_fields_freq_dft[field_name] *= dt
    for field_name in H_fields_freq_dft:
        H_fields_freq_dft[field_name] *= dt

    print("DFT calculation complete.")
    return E_fields_freq_dft, H_fields_freq_dft


if __name__ == '__main__':
    print("\\n--- DFT Snapshots Test (Corrected Order & Logic) ---")
    
    test_snapshot_dir = "dft_test_snapshots_postproc"
    os.makedirs(test_snapshot_dir, exist_ok=True)
    
    prefix = "snap_s_"
    dt_test = 1e-12 # s
    target_freq_test = 1e9 # Hz (1 GHz)
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    field_shape = (3, 3, 3) 
    num_test_snapshots = 5
    
    print(f"Creating {num_test_snapshots} dummy snapshot files in '{test_snapshot_dir}'...")
    # Create files in a non-sequential order to test sorting
    test_steps = [3, 1, 5, 2, 4] 
    for i_orig_order, step_val in enumerate(test_steps):
        t_val_snap = step_val * dt_test
        
        ez_data = torch.sin(torch.tensor(2 * np.pi * target_freq_test * t_val_snap, dtype=torch.float32)) * \
                  torch.ones(field_shape, dtype=torch.float32) 
        hx_data = torch.cos(torch.tensor(2 * np.pi * target_freq_test * t_val_snap, dtype=torch.float32)) * \
                  torch.ones(field_shape, dtype=torch.float32) * 0.5

        snapshot_content = {"Ez": ez_data, "Hx": hx_data, "InvalidField": torch.rand(1)}
        # Use varying padding for step numbers
        filename = os.path.join(test_snapshot_dir, f"{prefix}{step_val:03d}.pt") 
        torch.save(snapshot_content, filename)
    print(f"Dummy files created.")

    # Run DFT (parameter order matches prompt)
    E_freq, H_freq = dft_snapshots(snapshot_dir=test_snapshot_dir, 
                                   target_frequency=target_freq_test, 
                                   dt=dt_test, 
                                   snapshot_file_prefix=prefix, 
                                   device=test_device)

    if E_freq is not None and H_freq is not None:
        print("\nDFT Results:")
        if "Ez" in E_freq:
            Ez_dft = E_freq["Ez"]
            print(f"  Ez DFT component shape: {Ez_dft.shape}, dtype: {Ez_dft.dtype}, device: {Ez_dft.device}")
            print(f"  Ez DFT mean amplitude: {Ez_dft.abs().mean().item():.4e}")
            print(f"  Ez DFT mean phase (rad): {Ez_dft.angle().mean().item():.4f}")
            expected_mag_ez = (num_test_snapshots * dt_test) / 2.0
            print(f"  Expected Ez DFT mean amplitude for pure sine over {num_test_snapshots} points: {expected_mag_ez:.4e}")
            # Note: Actual DFT values depend on windowing, exact number of periods, etc.
            # This is a basic check that values are generated.

        if "Hx" in H_freq:
            Hx_dft = H_freq["Hx"]
            print(f"  Hx DFT component shape: {Hx_dft.shape}, dtype: {Hx_dft.dtype}, device: {Hx_dft.device}")
            print(f"  Hx DFT mean amplitude: {Hx_dft.abs().mean().item():.4e}")
            print(f"  Hx DFT mean phase (rad): {Hx_dft.angle().mean().item():.4f}")
            expected_mag_hx = (num_test_snapshots * dt_test * 0.5) / 2.0 
            print(f"  Expected Hx DFT mean amplitude for pure cosine over {num_test_snapshots} points: {expected_mag_hx:.4e}")
    else:
        print("DFT calculation failed or returned no results.")

    import shutil
    if os.path.exists(test_snapshot_dir):
        print(f"\nCleaning up test directory: {test_snapshot_dir}")
        shutil.rmtree(test_snapshot_dir)
    
    print("\n--- DFT Snapshots Test (Corrected Order & Logic) Finished ---")


def calculate_b1_plus_minus(Hx_freq, Hy_freq):
    """
    Calculates the B1+ (co-rotating) and B1- (counter-rotating) components
    of the transverse magnetic field.

    These are defined as:
        B1+ = (Bx_freq + j*By_freq) / 2
        B1- = (Bx_freq - j*By_freq) / 2
    (Using Hx, Hy as direct proxies for Bx, By, ignoring mu for simplicity here,
     as B1 is often discussed in terms of H fields directly in MR literature, or
     it's B = mu*H and the mu is factored out or applied later).

    Args:
        Hx_freq (torch.Tensor): Complex-valued PyTorch tensor for the
                                frequency-domain Hx field component.
        Hy_freq (torch.Tensor): Complex-valued PyTorch tensor for the
                                frequency-domain Hy field component.
                                Must be the same shape as Hx_freq.

    Returns:
        tuple: A tuple containing two complex-valued PyTorch tensors:
            - B1_plus (torch.Tensor)
            - B1_minus (torch.Tensor)
    """
    if not (isinstance(Hx_freq, torch.Tensor) and Hx_freq.is_complex()):
        raise TypeError("Hx_freq must be a complex PyTorch tensor.")
    if not (isinstance(Hy_freq, torch.Tensor) and Hy_freq.is_complex()):
        raise TypeError("Hy_freq must be a complex PyTorch tensor.")
    if Hx_freq.shape != Hy_freq.shape:
        raise ValueError("Hx_freq and Hy_freq must have the same shape.")

    # j * Hy_freq
    j_Hy_freq = 1j * Hy_freq

    B1_plus = (Hx_freq + j_Hy_freq) / 2.0
    B1_minus = (Hx_freq - j_Hy_freq) / 2.0
    
    return B1_plus, B1_minus


def calculate_sar(Ex_freq, Ey_freq, Ez_freq, sigma_grid, rho_grid, epsilon_rho=1e-9):
    """
    Calculates the Specific Absorption Rate (SAR) map.

    SAR is calculated using the formula:
        SAR = sigma * (|Ex|^2 + |Ey|^2 + |Ez|^2) / (2 * rho)
    where:
        sigma is the electrical conductivity (S/m).
        rho is the mass density (kg/m^3).
        |E|^2 is the squared magnitude of the peak electric field.
              (Note: If E_freq are RMS values, factor of 2 in denominator might change
               or |E|^2 would be 2*|E_rms|^2. Assuming E_freq from DFT are peak phasors,
               so |E_freq|^2 is peak magnitude squared.)

    Args:
        Ex_freq (torch.Tensor): Complex-valued PyTorch tensor for frequency-domain Ex.
        Ey_freq (torch.Tensor): Complex-valued PyTorch tensor for frequency-domain Ey.
        Ez_freq (torch.Tensor): Complex-valued PyTorch tensor for frequency-domain Ez.
        sigma_grid (torch.Tensor): Real-valued PyTorch tensor for electrical conductivity (S/m).
                                   Shape should be compatible for broadcasting with E fields
                                   (typically cell-centered, matching E component locations
                                    after interpolation or if E fields are also cell-centered).
        rho_grid (torch.Tensor): Real-valued PyTorch tensor for mass density (kg/m^3).
                                 Shape compatible with sigma_grid.
        epsilon_rho (float, optional): A small epsilon value to add to rho_grid in the
                                       denominator to prevent division by zero.
                                       Defaults to 1e-9.

    Returns:
        torch.Tensor: Real-valued PyTorch tensor representing the SAR map (W/kg).
                      Values will be 0 where rho_grid was effectively zero.
    """
    if not (isinstance(Ex_freq, torch.Tensor) and Ex_freq.is_complex()):
        raise TypeError("Ex_freq must be a complex PyTorch tensor.")
    if not (isinstance(Ey_freq, torch.Tensor) and Ey_freq.is_complex()):
        raise TypeError("Ey_freq must be a complex PyTorch tensor.")
    if not (isinstance(Ez_freq, torch.Tensor) and Ez_freq.is_complex()):
        raise TypeError("Ez_freq must be a complex PyTorch tensor.")
    if not isinstance(sigma_grid, torch.Tensor) or sigma_grid.is_complex():
        raise TypeError("sigma_grid must be a real PyTorch tensor.")
    if not isinstance(rho_grid, torch.Tensor) or rho_grid.is_complex():
        raise TypeError("rho_grid must be a real PyTorch tensor.")

    # Ensure all tensors are on the same device
    target_device = Ex_freq.device
    Ey_freq = Ey_freq.to(target_device)
    Ez_freq = Ez_freq.to(target_device)
    sigma_grid = sigma_grid.to(target_device)
    rho_grid = rho_grid.to(target_device)

    # Check for shape compatibility. This is a simplified check.
    # E-fields might be on Yee grid locations, while sigma/rho are cell-centered.
    # For this function, we assume they are already compatible (e.g., E-fields interpolated
    # to cell centers, or sigma/rho expanded/interpolated to E-field locations).
    # A common approach is to calculate |E|^2 at cell centers.
    # If Ex, Ey, Ez are passed with their original Yee grid shapes, this needs careful handling.
    # For now, assume shapes are directly compatible for element-wise operations.
    if not (Ex_freq.shape == Ey_freq.shape == Ez_freq.shape == sigma_grid.shape == rho_grid.shape):
        print("Warning: Shapes of input E-fields, sigma_grid, and rho_grid are not all identical. "
              "Ensure they are broadcastable or represent cell-centered quantities.")
        # Example shapes: Ex(Nx,Ny+1,Nz+1), sigma(Nx,Ny,Nz). This will fail.
        # This function expects that if Ex_freq is passed, sigma_grid and rho_grid match its shape.
        # This means prior interpolation of sigma/rho or E-fields is needed.

    E_magnitude_sq = Ex_freq.abs()**2 + Ey_freq.abs()**2 + Ez_freq.abs()**2
    
    # Handle potential division by zero for rho_grid
    # Create a safe version of rho_grid for division
    rho_safe = rho_grid.clone()
    # Where rho_grid is very small (close to zero), set rho_safe to a non-zero value
    # to avoid division by zero, but SAR will be set to 0 for these regions later.
    rho_safe[rho_grid <= epsilon_rho] = 1.0 # Temporary placeholder for division
    
    sar_map = (sigma_grid * E_magnitude_sq) / (2 * rho_safe)
    
    # Ensure SAR is zero where density is effectively zero
    sar_map[rho_grid <= epsilon_rho] = 0.0 
    
    return sar_map


if __name__ == '__main__': # Append to existing if __name__ block
    # (Previous DFT test code is above this)

    print("\\n--- B1+/B1- Calculation Test ---")
    Hx_test = torch.tensor([1+1j, 2-0.5j], dtype=torch.cfloat)
    Hy_test = torch.tensor([0.5-2j, 1+1j], dtype=torch.cfloat)
    
    B1p, B1m = calculate_b1_plus_minus(Hx_test, Hy_test)
    print(f"Hx_freq: {Hx_test}")
    print(f"Hy_freq: {Hy_test}")
    print(f"B1+: {B1p}") # Expected: ( (1+1j) + j*(0.5-2j) )/2 = ( (1+1j) + (0.5j+2) )/2 = (3+1.5j)/2 = 1.5+0.75j
                         #           ( (2-0.5j) + j*(1+1j) )/2 = ( (2-0.5j) + (1j-1) )/2 = (1+0.5j)/2 = 0.5+0.25j
    print(f"B1-: {B1m}") # Expected: ( (1+1j) - j*(0.5-2j) )/2 = ( (1+1j) - (0.5j+2) )/2 = (-1+0.5j)/2 = -0.5+0.25j
                         #           ( (2-0.5j) - j*(1+1j) )/2 = ( (2-0.5j) - (1j-1) )/2 = (3-1.5j)/2 = 1.5-0.75j
    
    expected_B1p_val0 = torch.tensor(1.5 + 0.75j, dtype=torch.cfloat)
    expected_B1m_val0 = torch.tensor(-0.5 + 0.25j, dtype=torch.cfloat)
    assert torch.allclose(B1p[0], expected_B1p_val0), f"B1+ calculation error. Got {B1p[0]}, expected {expected_B1p_val0}"
    assert torch.allclose(B1m[0], expected_B1m_val0), f"B1- calculation error. Got {B1m[0]}, expected {expected_B1m_val0}"
    print("B1+/B1- test passed for first element.")


    print("\\n--- SAR Calculation Test ---")
    sar_E_shape = (2,2,2)
    Ex_f = torch.ones(sar_E_shape, dtype=torch.cfloat) * (1+0j)    # |Ex|=1
    Ey_f = torch.ones(sar_E_shape, dtype=torch.cfloat) * (0+1j)    # |Ey|=1
    Ez_f = torch.zeros(sar_E_shape, dtype=torch.cfloat)             # |Ez|=0
    # Total |E|^2 = 1^2 + 1^2 + 0^2 = 2

    sigma_test = torch.ones(sar_E_shape, dtype=torch.float32) * 0.5 # S/m
    rho_test = torch.ones(sar_E_shape, dtype=torch.float32) * 1000  # kg/m^3
    rho_test[0,0,0] = 0 # Test zero density case

    sar = calculate_sar(Ex_f, Ey_f, Ez_f, sigma_test, rho_test)
    print(f"Calculated SAR map:\n{sar}")
    
    # Expected SAR for non-zero rho: (0.5 * 2) / (2 * 1000) = 1 / 2000 = 0.0005 W/kg
    expected_sar_val = 0.0005
    assert abs(sar[1,1,1].item() - expected_sar_val) < 1e-6, "SAR calculation error."
    assert sar[0,0,0].item() == 0.0, "SAR should be 0 for zero density."
    print("SAR calculation test passed.")
    
    print("\n--- Postprocessing Functions Test Finished ---")
```
