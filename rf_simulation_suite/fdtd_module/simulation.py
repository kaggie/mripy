import torch
import os
import time
import numpy as np # For the test block

# Attempt relative imports for package structure
try:
    from .grid import FDTDGrid
    from .core_updater import FDTDUpdater
    from .sources import VoltageSource
    from .boundaries import MurABC3D
except ImportError:
    # Fallback for direct script execution (e.g., if file is run from this directory)
    print("FDTDSimulation: Using fallback imports for FDTDGrid, FDTDUpdater, VoltageSource, MurABC3D.")
    from grid import FDTDGrid
    from core_updater import FDTDUpdater
    from sources import VoltageSource
    from boundaries import MurABC3D


class FDTDSimulation:
    """
    Manages and runs a full FDTD simulation sequence.
    """

    def __init__(self, grid, updater, sources_list, boundary_condition, results_dir="fdtd_results"):
        """
        Initializes the FDTDSimulation environment.

        Args:
            grid (FDTDGrid): The FDTD grid object.
            updater (FDTDUpdater): The FDTD core updater object.
            sources_list (list): A list of source objects (e.g., VoltageSource instances).
                                 If a single source object is provided, it will be wrapped in a list.
            boundary_condition (object): The boundary condition object (e.g., MurABC3D instance).
            results_dir (str, optional): Directory to save simulation snapshots.
                                         Defaults to "fdtd_results".
        """
        self.grid = grid
        self.updater = updater
        
        if not isinstance(sources_list, list):
            self.sources_list = [sources_list] if sources_list is not None else []
        else:
            self.sources_list = sources_list
            
        self.boundary_condition = boundary_condition
        self.results_dir = results_dir

        # Create results directory if it doesn't exist
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"FDTDSimulation: Results will be saved in '{self.results_dir}'")
        except OSError as e:
            # Handle cases where directory creation might fail for other reasons (e.g. permissions)
            print(f"Error creating results directory '{self.results_dir}': {e}")
            # Optionally, re-raise or set a flag indicating saving might fail
            raise

    def run(self, num_timesteps, recording_interval, fields_to_record=['Ez', 'Hz']):
        """
        Runs the FDTD simulation for a specified number of time steps.

        Args:
            num_timesteps (int): Total number of time steps to simulate.
            recording_interval (int): Interval at which to save field snapshots.
                                      Snapshots are saved every 'recording_interval' steps.
            fields_to_record (list, optional): A list of strings specifying which field
                                               components to save (e.g., ['Ex', 'Hz']).
                                               Defaults to ['Ez', 'Hz'].
        """
        if not isinstance(num_timesteps, int) or num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer.")
        if not isinstance(recording_interval, int) or recording_interval <= 0:
            raise ValueError("recording_interval must be a positive integer.")
        if not isinstance(fields_to_record, list) or not all(isinstance(f, str) for f in fields_to_record):
            raise ValueError("fields_to_record must be a list of strings.")

        print(f"\n--- Starting FDTD Simulation ---")
        print(f"Total time steps: {num_timesteps}")
        print(f"Recording snapshots every {recording_interval} steps.")
        print(f"Fields to record: {fields_to_record}")
        print(f"Device: {self.grid.device}")

        start_time = time.time()

        for t_idx in range(num_timesteps):
            # 1. Update H-Fields (H at t + dt/2 using E at t)
            self.updater.update_H_fields()

            # 2. Apply H-Boundary Conditions (Corrects H at t + dt/2)
            if self.boundary_condition:
                self.boundary_condition.apply_H_boundary() # Currently 'pass' for MurABC3D

            # 3. Store Previous E-Fields (Stores E at t for ABC: E_p=E(t), E_pp=E(t-dt))
            # This is done before E-field update, so self.grid.Ex etc. are E(t)
            if self.boundary_condition:
                self.boundary_condition.store_previous_fields()

            # 4. Update E-Fields (E at t + dt using H at t + dt/2)
            self.updater.update_E_fields()

            # 5. Apply Sources (Modifies E at t + dt)
            # t_idx here corresponds to the current time step index, so source is applied
            # to contribute to E(t_idx * dt + dt) or E((t_idx+1)*dt)
            for source in self.sources_list:
                source.apply_at_time(self.grid, t_idx) # t_idx is time step 'n'

            # 6. Apply E-Boundary Conditions (Corrects E at t + dt using E_p, E_pp)
            if self.boundary_condition:
                self.boundary_condition.apply_E_boundary()

            # 7. Record Snapshots
            if (t_idx + 1) % recording_interval == 0:
                snapshot_data = {}
                for field_name in fields_to_record:
                    if hasattr(self.grid, field_name):
                        field_tensor = getattr(self.grid, field_name)
                        snapshot_data[field_name] = field_tensor.cpu().clone() # Save on CPU
                    else:
                        print(f"Warning: Field '{field_name}' not found in grid for recording.")
                
                if snapshot_data: # Only save if there's something to save
                    snapshot_filename = os.path.join(self.results_dir, f"snapshot_step_{t_idx+1}.pt")
                    try:
                        torch.save(snapshot_data, snapshot_filename)
                        # print(f"Saved snapshot at step {t_idx+1} to {snapshot_filename}")
                    except Exception as e:
                        print(f"Error saving snapshot at step {t_idx+1}: {e}")

            # 8. Progress Reporting
            if (t_idx + 1) % (num_timesteps // 20 or 1) == 0 or (t_idx + 1) == num_timesteps : # ~20 updates
                current_loop_time = time.time()
                elapsed = current_loop_time - start_time
                # Estimate ETA, handle t_idx=0 case for elapsed/(t_idx+1)
                eta = (elapsed / (t_idx + 1)) * (num_timesteps - (t_idx + 1)) if (t_idx + 1) > 0 else 0
                print(f"  Progress: {((t_idx + 1) / num_timesteps * 100):.1f}% "
                      f"({t_idx+1}/{num_timesteps} steps) | "
                      f"Elapsed: {elapsed:.2f}s | ETA: {eta:.2f}s")

        end_time = time.time()
        total_simulation_time = end_time - start_time
        print(f"--- FDTD Simulation Finished ---")
        print(f"Total simulation time: {total_simulation_time:.2f} seconds.")


if __name__ == '__main__':
    print("\\n--- FDTDSimulation Test ---")

    # Setup a minimal FDTD environment for testing the simulation loop
    sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test using device: {sim_device}")

    grid_dims = (10, 10, 10) # Nx, Ny, Nz cells
    cell_s = 1e-3 # 1 mm
    
    # Ensure dt_courant_factor is reasonable, e.g. 0.5 for more stability in tests if needed
    test_grid = FDTDGrid(dimensions_cells=grid_dims, cell_size_m=cell_s, device=sim_device, dt_courant_factor=0.7)
    test_updater = FDTDUpdater(test_grid)
    
    # Define a source
    source_loc = (grid_dims[0]//2, grid_dims[1]//2, grid_dims[2]//2)
    test_source = VoltageSource(location_cells=source_loc, axis='Ez', 
                                amplitude=1.0, frequency=1e9, # 1 GHz
                                waveform_type='gaussian_pulse',
                                pulse_center_time_steps=40,
                                pulse_width_time_steps=15) # Adjust for dt
    
    test_boundary = MurABC3D(test_grid)
    
    # Create results directory for this test
    test_results_dir = "fdtd_test_results"

    simulation = FDTDSimulation(grid=test_grid, 
                                updater=test_updater, 
                                sources_list=[test_source], 
                                boundary_condition=test_boundary,
                                results_dir=test_results_dir)

    print("FDTDSimulation instance created.")

    # Run the simulation
    sim_steps = 100
    rec_interval = 20
    try:
        simulation.run(num_timesteps=sim_steps, 
                       recording_interval=rec_interval, 
                       fields_to_record=['Ez', 'Hx', 'Hy'])
        
        print(f"\nSimulation run completed. Check '{test_results_dir}' for snapshots.")
        # Verify snapshot creation
        expected_snapshots = sim_steps // rec_interval
        # Add a small tolerance for floating point comparison related to num_timesteps % rec_interval
        # No, exact number of snapshots should be created.
        
        # List files and count .pt files
        num_snapshots_created = 0
        if os.path.exists(test_results_dir):
            for f_name in os.listdir(test_results_dir):
                if f_name.startswith("snapshot_step_") and f_name.endswith(".pt"):
                    num_snapshots_created +=1
            print(f"Found {num_snapshots_created} snapshot files.")
        else:
            print(f"Warning: Results directory '{test_results_dir}' not found after simulation.")

        assert num_snapshots_created == expected_snapshots, \
            f"Expected {expected_snapshots} snapshots, but found {num_snapshots_created}."
        print("Snapshot creation count matches expected.")

    except Exception as e:
        print(f"Error during simulation run test: {e}")
        raise
    finally:
        # Clean up test results directory
        # import shutil
        # if os.path.exists(test_results_dir):
        #     print(f"Cleaning up test results directory: {test_results_dir}")
        #     shutil.rmtree(test_results_dir)
        pass # Keep results for inspection for now

    print("\\n--- FDTDSimulation Test Finished ---")

```
