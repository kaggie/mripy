import torch
import matplotlib.pyplot as plt

# --- Biot-Savart Module Imports ---
from rf_coil_simulation_suite.biot_savart_module import (
    CircularLoopCoil,
    SolenoidCoil,               # Added SolenoidCoil
    SingleRungBirdcageCoil,     # Added SingleRungBirdcageCoil
    generate_b_field_map,
    plot_b_field_slice as plot_biot_savart_slice, # Alias to avoid name clash
    calculate_magnetic_field_at_point 
)

# --- FDTD Module Imports ---
from rf_coil_simulation_suite.fdtd_module import (
    FDTDSimulator,
    MATERIAL_DATABASE,
    plot_fdtd_results_slice # This is the new plotting function from fdtd_module
)
import numpy as np # For FDTD example, e.g. pi


def run_biot_savart_example():
    """
    Demonstrates Biot-Savart simulation for a circular loop coil.
    """
    print("Starting Biot-Savart simulation demonstration...")

    # 1. Setup Simulation Parameters
    print("Setting up simulation parameters...")
    radius = 0.1  # meters (10 cm)
    num_segments_coil = 100
    current = 1.0  # Amperes
    coil_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    coil_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    map_x_range = [-0.05, 0.05]
    map_y_range = [-0.05, 0.05]
    resolution = 31
    
    x_coords = torch.linspace(map_x_range[0], map_x_range[1], resolution, dtype=torch.float32)
    y_coords = torch.linspace(map_y_range[0], map_y_range[1], resolution, dtype=torch.float32)
    z_coords_3d_map = torch.linspace(-0.05, 0.05, resolution, dtype=torch.float32)

    # 2. Create Coil Object
    print("Creating coil object...")
    loop_coil = CircularLoopCoil(
        radius=radius,
        num_segments=num_segments_coil,
        center=coil_center,
        normal=coil_normal
    )

    # 3. Simulate B-Field Map (for Central XY Plane visualization)
    print("Simulating B-field map for XY slice...")
    b_field_map_3d = generate_b_field_map(
        loop_coil, x_coords, y_coords, z_coords_3d_map, current
    )
    z_slice_idx = resolution // 2
    print(f"Central Z-slice for Biot-Savart XY plane plot is at Z = {z_coords_3d_map[z_slice_idx]:.3e} m (index {z_slice_idx})")

    print("Plotting B_xy_magnitude in central XY plane (Biot-Savart)...")
    fig_bs_xy, ax_bs_xy = plt.subplots(figsize=(8, 7))
    plot_biot_savart_slice( # Use aliased name
        b_field_map=b_field_map_3d,
        x_coords=x_coords, y_coords=y_coords, z_coords=z_coords_3d_map,
        slice_axis='z', slice_index=z_slice_idx, component='xy_magnitude',
        ax=ax_bs_xy, show_plot=False
    )
    ax_bs_xy.set_title(f"$B_{{xy}}$ magnitude in XY plane (Z={z_coords_3d_map[z_slice_idx]:.2e}m) - Biot-Savart")

    # 4. Simulate and Plot Bz along the Z-axis (Biot-Savart)
    print("Simulating and plotting Bz along the Z-axis (Biot-Savart)...")
    z_line_plot_range = [-0.15, 0.15]
    num_points_z_line = 100
    z_axis_points_single_dim = torch.linspace(z_line_plot_range[0], z_line_plot_range[1], num_points_z_line, dtype=torch.float32)
    z_axis_points_3d = torch.stack([
        torch.zeros_like(z_axis_points_single_dim),
        torch.zeros_like(z_axis_points_single_dim),
        z_axis_points_single_dim
    ], dim=1)
    coil_segments = loop_coil.get_segments()
    bz_values_on_axis = torch.zeros(num_points_z_line, dtype=torch.float32)
    for i, point_on_axis in enumerate(z_axis_points_3d):
        b_field_at_point = calculate_magnetic_field_at_point(
            segments=coil_segments, point=point_on_axis, current_magnitude=current
        )
        bz_values_on_axis[i] = b_field_at_point[2]

    fig_bs_z, ax_bs_z = plt.subplots(figsize=(8, 6))
    ax_bs_z.plot(z_axis_points_single_dim.numpy(), bz_values_on_axis.numpy())
    ax_bs_z.set_xlabel("Z-coordinate (m) along coil axis")
    ax_bs_z.set_ylabel("$B_z$ component (Tesla)")
    ax_bs_z.set_title("$B_z$ along coil axis (R=0.1m, I=1A) - Biot-Savart")
    ax_bs_z.grid(True)
    print("Biot-Savart simulation and plotting complete for single circular loop.")
    # --- End of Single Circular Loop Example ---

    # --- SolenoidCoil Example ---
    print("\n--- Starting SolenoidCoil Example ---")
    solenoid_radius = 0.05  # meters
    solenoid_length = 0.2   # meters
    solenoid_num_turns = 5
    solenoid_segments_per_turn = 50
    solenoid_current = 1.0  # Amperes
    solenoid_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    solenoid_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32) # Along Z-axis

    solenoid = SolenoidCoil(
        radius=solenoid_radius,
        length=solenoid_length,
        num_turns=solenoid_num_turns,
        segments_per_turn=solenoid_segments_per_turn,
        center=solenoid_center,
        axis=solenoid_axis
    )

    # Plot Bz along the Solenoid's axis
    print("Simulating and plotting Bz along the Z-axis for SolenoidCoil...")
    solenoid_z_line_plot_range = [-0.15, 0.15] # meters, covering length and beyond
    solenoid_num_points_z_line = 100
    solenoid_z_axis_points_single_dim = torch.linspace(
        solenoid_z_line_plot_range[0], solenoid_z_line_plot_range[1], 
        solenoid_num_points_z_line, dtype=torch.float32
    )
    solenoid_z_axis_points_3d = torch.stack([
        torch.zeros_like(solenoid_z_axis_points_single_dim),
        torch.zeros_like(solenoid_z_axis_points_single_dim),
        solenoid_z_axis_points_single_dim
    ], dim=1)
    
    solenoid_segments = solenoid.get_segments()
    solenoid_bz_on_axis = torch.zeros(solenoid_num_points_z_line, dtype=torch.float32)
    for i, point_on_axis in enumerate(solenoid_z_axis_points_3d):
        b_field_solenoid = calculate_magnetic_field_at_point(
            segments=solenoid_segments, point=point_on_axis, current_magnitude=solenoid_current
        )
        solenoid_bz_on_axis[i] = b_field_solenoid[2]

    fig_sol_z, ax_sol_z = plt.subplots(figsize=(8, 6))
    ax_sol_z.plot(solenoid_z_axis_points_single_dim.numpy(), solenoid_bz_on_axis.numpy())
    ax_sol_z.set_xlabel("Z-coordinate (m) along solenoid axis")
    ax_sol_z.set_ylabel("$B_z$ component (Tesla)")
    ax_sol_z.set_title(f"$B_z$ along Solenoid Axis (R={solenoid_radius}m, L={solenoid_length}m, N={solenoid_num_turns})")
    ax_sol_z.grid(True)
    print("SolenoidCoil Bz plot complete.")
    # --- End of SolenoidCoil Example ---

    # --- SingleRungBirdcageCoil Example ---
    print("\n--- Starting SingleRungBirdcageCoil Example ---")
    birdcage_radius = 0.1
    birdcage_length = 0.15
    birdcage_rung_angle_deg = 0.0
    birdcage_num_segments_rung = 20
    birdcage_num_segments_arc = 10
    birdcage_current = 1.0

    single_rung = SingleRungBirdcageCoil(
        radius=birdcage_radius,
        length=birdcage_length,
        rung_angle_deg=birdcage_rung_angle_deg,
        num_segments_per_rung=birdcage_num_segments_rung,
        num_segments_per_arc=birdcage_num_segments_arc,
        center=torch.tensor([0.0, 0.0, 0.0]),
        axis=torch.tensor([0.0, 0.0, 1.0]) # Rung parallel to Z-axis, in XZ plane at Y=0
    )
    
    # Define a plane for B-field visualization (e.g., YZ plane at X slightly offset from rung)
    # Rung is at X=radius*cos(0)=0.1, Y=radius*sin(0)=0. Let's plot in YZ plane at x = 0.08 (near the rung)
    birdcage_x_coords = torch.tensor([birdcage_radius - 0.02], dtype=torch.float32) # Single X value
    birdcage_y_coords = torch.linspace(-0.05, 0.05, resolution, dtype=torch.float32)
    birdcage_z_coords = torch.linspace(-birdcage_length/2 - 0.02, birdcage_length/2 + 0.02, resolution, dtype=torch.float32)

    print("Simulating B-field map for SingleRungBirdcageCoil...")
    b_map_rung = generate_b_field_map(
        single_rung, birdcage_x_coords, birdcage_y_coords, birdcage_z_coords, birdcage_current
    )
    # Since birdcage_x_coords has only one value, b_map_rung will be (1, Ny, Nz, 3). We squeeze it.
    b_map_rung_slice = b_map_rung.squeeze(dim=0) # Shape (Ny, Nz, 3)
    
    # For plot_b_field_slice, we need to pass the full 3D map and specify slice_axis='x', slice_index=0
    print("Plotting B-field magnitude for SingleRungBirdcageCoil (YZ slice)...")
    fig_rung, ax_rung = plt.subplots(figsize=(8, 7))
    plot_biot_savart_slice(
        b_field_map=b_map_rung, # Pass the original 3D map
        x_coords=birdcage_x_coords, # Pass the coordinates used for the map
        y_coords=birdcage_y_coords,
        z_coords=birdcage_z_coords,
        slice_axis='x', 
        slice_index=0, # Index for the single x-coordinate
        component='magnitude',
        ax=ax_rung, show_plot=False
    )
    ax_rung.set_title(f"B-field Magnitude for Single Rung (YZ slice at X={birdcage_x_coords[0]:.2f}m)")
    print("SingleRungBirdcageCoil visualization complete.")
    # --- End of SingleRungBirdcageCoil Example ---

    # --- 2x1 Array of Circular Loops Example ---
    print("\n--- Starting 2x1 Circular Loop Array Example ---")
    array_loop_radius = 0.05
    array_num_segments = 50
    array_current = 1.0
    loop1_center = torch.tensor([-0.06, 0.0, 0.0], dtype=torch.float32)
    loop2_center = torch.tensor([0.06, 0.0, 0.0], dtype=torch.float32)
    loop_array_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32) # Both loops in XY plane

    loop1 = CircularLoopCoil(array_loop_radius, array_num_segments, loop1_center, loop_array_normal)
    loop2 = CircularLoopCoil(array_loop_radius, array_num_segments, loop2_center, loop_array_normal)

    # Define grid for combined field (e.g., central XY plane)
    array_x_coords = torch.linspace(-0.15, 0.15, resolution, dtype=torch.float32)
    array_y_coords = torch.linspace(-0.1, 0.1, resolution, dtype=torch.float32)
    array_z_coords = torch.linspace(-0.01, 0.01, 5, dtype=torch.float32) # Few points in Z for XY slice
    array_z_slice_idx = array_z_coords.shape[0] // 2

    print("Simulating B-field for loop 1 of array...")
    b_map_loop1 = generate_b_field_map(loop1, array_x_coords, array_y_coords, array_z_coords, array_current)
    print("Simulating B-field for loop 2 of array...")
    b_map_loop2 = generate_b_field_map(loop2, array_x_coords, array_y_coords, array_z_coords, array_current)
    
    b_map_combined = b_map_loop1 + b_map_loop2
    
    print("Plotting combined B_xy_magnitude in central XY plane for loop array...")
    fig_array, ax_array = plt.subplots(figsize=(8, 7))
    plot_biot_savart_slice(
        b_field_map=b_map_combined,
        x_coords=array_x_coords, y_coords=array_y_coords, z_coords=array_z_coords,
        slice_axis='z', slice_index=array_z_slice_idx, component='xy_magnitude',
        ax=ax_array, show_plot=False
    )
    ax_array.set_title(f"Combined $B_{{xy}}$ magnitude for 2x1 Loop Array (XY plane at Z={array_z_coords[array_z_slice_idx]:.2e}m)")
    print("2x1 Loop Array example complete.")
    # --- End of 2x1 Array Example ---

    print("\nAll Biot-Savart examples finished.")
    # plt.show() # Moved to main execution block


def run_fdtd_example():
    """
    Main function to demonstrate FDTD simulation capabilities.
    """
    print("Starting FDTD simulation demonstration...")

    # 1. Setup Simulator
    print("Setting up FDTD simulator...")
    dims = (50, 50, 50)  # cells
    cell_size = 0.002  # meters (2 mm)
    pml_layers = 10    # Number of PML layers

    # Check for CUDA device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.float32 # CUDA prefers float32
        print("CUDA device selected for FDTD simulation.")
    else:
        device = torch.device('cpu')
        dtype = torch.float64 # CPU can use float64 for better precision
        print("CPU device selected for FDTD simulation.")

    simulator = FDTDSimulator(
        dimensions=dims, 
        cell_size=cell_size, 
        pml_layers=pml_layers,
        dtype=dtype,
        device=device
    )

    # 2. Load Materials
    print("Loading material properties...")
    phantom_data = torch.zeros(dims, dtype=torch.long, device=device) # Default to Air (index 0)
    # Create a central cube of muscle (index 2)
    center_x, center_y, center_z = dims[0]//2, dims[1]//2, dims[2]//2
    cube_half_size = 5 # cells
    phantom_data[
        center_x - cube_half_size : center_x + cube_half_size,
        center_y - cube_half_size : center_y + cube_half_size,
        center_z - cube_half_size : center_z + cube_half_size
    ] = 2 # Muscle index from MATERIAL_DATABASE
    simulator.load_material_properties(phantom_data, MATERIAL_DATABASE)

    # 3. Define Source
    print("Defining source...")
    source_frequency = 150e6  # Hz (150 MHz)
    # Gaussian pulse parameters
    tau_pulse = 1.0 / (source_frequency * 0.4 * np.pi) # Using np.pi here
    t0_pulse = 3.0 * tau_pulse
    
    source_config = {
        'location_indices': (center_x, center_y, center_z),
        'field_component': 'Ez',
        'waveform_type': 'gaussian_pulse',
        'frequency': source_frequency,
        'amplitude': 1.0, # V/m
        'waveform_params': {'t0': t0_pulse, 'tau': tau_pulse}
    }

    # 4. Run Simulation
    print("Running FDTD simulation...")
    # Simulate for a bit longer than the pulse peak to capture its main energy
    num_timesteps = int(t0_pulse * 2.5 / simulator.dt.item()) 
    recording_interval = max(1, num_timesteps // 50) # Record about 50 snapshots
    
    recorded_data = simulator.run_fdtd_simulation(
        num_timesteps=num_timesteps,
        source_config=source_config,
        recording_interval=recording_interval
    )

    # 5. Post-Processing
    print("Starting post-processing...")
    time_steps = recorded_data['time_steps']
    snapshot_interval_from_sim = recording_interval 

    if not time_steps:
        print("No snapshots recorded, skipping post-processing and visualization.")
        return

    freq_domain_fields = {}
    field_components_to_extract = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy']
    
    for field_name in field_components_to_extract:
        if recorded_data['field_snapshots'].get(field_name) and len(recorded_data['field_snapshots'][field_name]) > 0:
            print(f"Extracting {field_name} in frequency domain...")
            freq_domain_fields[field_name] = simulator.extract_frequency_domain_fields(
                recorded_data['field_snapshots'][field_name], 
                time_steps, 
                source_frequency, 
                snapshot_interval_from_sim
            )
        else:
            print(f"No snapshots for {field_name} found, skipping its DFT.")

    # Calculate B1+
    B1_plus = None
    if 'Hx' in freq_domain_fields and 'Hy' in freq_domain_fields:
        print("Calculating B1+ map...")
        B1_plus, _ = simulator.calculate_b1_plus_minus(
            freq_domain_fields['Hx'], 
            freq_domain_fields['Hy']
        )
    else:
        print("Hx or Hy frequency domain data not available, skipping B1+ calculation.")

    # Calculate SAR
    sar_map = None
    if 'Ex' in freq_domain_fields and 'Ey' in freq_domain_fields and 'Ez' in freq_domain_fields:
        print("Calculating SAR map...")
        sar_map = simulator.calculate_sar(
            freq_domain_fields['Ex'], 
            freq_domain_fields['Ey'], 
            freq_domain_fields['Ez']
        )
    else:
        print("Ex, Ey, or Ez frequency domain data not available, skipping SAR calculation.")

    # 6. Visualization
    print("Visualizing results...")
    # Generate coordinate axes for plotting
    # Ensure coordinates are on CPU for matplotlib if they were on CUDA
    x_ax = (torch.arange(dims[0], device='cpu') * cell_size).numpy()
    y_ax = (torch.arange(dims[1], device='cpu') * cell_size).numpy()
    z_ax = (torch.arange(dims[2], device='cpu') * cell_size).numpy()

    slice_idx_z = dims[2] // 2 # Central slice
    
    if B1_plus is not None:
        plot_fdtd_results_slice(
            torch.abs(B1_plus), # Plot magnitude
            torch.from_numpy(x_ax), torch.from_numpy(y_ax), torch.from_numpy(z_ax), # Pass as tensors
            'z', slice_idx_z,
            'B1+ Magnitude (arb. units)',
            show_plot=False 
        )
    
    if sar_map is not None:
        plot_fdtd_results_slice(
            sar_map,
            torch.from_numpy(x_ax), torch.from_numpy(y_ax), torch.from_numpy(z_ax), # Pass as tensors
            'z', slice_idx_z,
            'SAR (W/kg)',
            show_plot=False 
        )
    
    # plt.show() # Moved to main execution block
    print("FDTD example finished.")


if __name__ == "__main__":
    run_biot_savart = True # Set to True to run Biot-Savart example
    run_fdtd = False       # Set to True to run FDTD example

    if run_biot_savart:
        print("--- Running Biot-Savart Example ---")
        run_biot_savart_example()
        plt.show() # Show Biot-Savart plots
        if run_fdtd: # If both are run, close BS plots before FDTD plots
            plt.close('all') 
    
    if run_fdtd:
        print("\n--- Running FDTD Example ---")
        run_fdtd_example()
        plt.show() # Show FDTD plots
    
    print("\nAll selected examples complete.")
