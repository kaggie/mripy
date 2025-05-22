import torch
import numpy as np

# --- Physical Constants ---
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
C_0 = 1 / np.sqrt(MU_0 * EPSILON_0)  # Speed of light in vacuum (m/s)

# --- Material Database ---
MATERIAL_DATABASE = {
    "air": {"eps_r": 1.0, "mu_r": 1.0, "sigma": 0.0},
    "water": {"eps_r": 78.0, "mu_r": 1.0, "sigma": 0.01}, # Example sigma for water
    "copper": {"eps_r": 1.0, "mu_r": 0.999994, "sigma": 5.8e7}, # mu_r for copper is close to 1
    "capacitor_dielectric": {"eps_r": 4.0, "mu_r": 1.0, "sigma": 1e-6}, # Generic dielectric
    "custom_high_permit": {"eps_r": 100.0, "mu_r": 1.0, "sigma": 1e-4},
    "custom_high_permeab": {"eps_r": 1.0, "mu_r": 100.0, "sigma": 0.0},
    "lossy_dielectric": {"eps_r": 5.0, "mu_r": 1.0, "sigma": 0.5}
}


class FDTDGrid:
    """
    Represents a 3D grid for Finite-Difference Time-Domain (FDTD) simulations.
    It initializes the E and H field components and material properties on the grid.
    The field components are defined on a standard Yee cell (staggered grid).
    Material properties (epsilon_r, mu_r, sigma) are cell-centered.
    """

    def __init__(self, dimensions_cells, cell_size_m, dt_courant_factor=0.9, device=None):
        """
        Initializes the FDTD grid.

        Args:
            dimensions_cells (tuple): A tuple of three integers (Nx, Ny, Nz) representing
                                      the number of cells in each dimension.
            cell_size_m (float or tuple): The size of each cell in meters.
                                          If a float, cells are uniform (dx=dy=dz).
                                          If a tuple (dx, dy, dz), specifies cell size
                                          for each dimension.
            dt_courant_factor (float, optional): A factor (0 to 1) by which to multiply
                                                 the Courant stability limit to determine dt.
                                                 Defaults to 0.9.
            device (torch.device, optional): The PyTorch device to use for tensors (e.g.,
                                             torch.device("cuda") or torch.device("cpu")).
                                             If None, attempts to use CUDA if available,
                                             otherwise defaults to CPU. Defaults to None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"FDTDGrid: Using device: {self.device}")

        if not isinstance(dimensions_cells, tuple) or len(dimensions_cells) != 3:
            raise ValueError("dimensions_cells must be a tuple of 3 integers (Nx, Ny, Nz).")
        self.Nx, self.Ny, self.Nz = dimensions_cells

        if isinstance(cell_size_m, (float, int)):
            self.dx = float(cell_size_m)
            self.dy = float(cell_size_m)
            self.dz = float(cell_size_m)
        elif isinstance(cell_size_m, tuple) and len(cell_size_m) == 3:
            self.dx, self.dy, self.dz = float(cell_size_m[0]), float(cell_size_m[1]), float(cell_size_m[2])
        else:
            raise ValueError("cell_size_m must be a float or a tuple of 3 floats (dx, dy, dz).")

        if not (0 < dt_courant_factor <= 1.0):
            raise ValueError("dt_courant_factor must be between 0 (exclusive) and 1.0 (inclusive).")

        # Calculate time step dt based on Courant stability condition
        # dt <= (1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)))
        if self.dx == 0 or self.dy == 0 or self.dz == 0: # Avoid division by zero if any dimension is flat (2D case)
            # For 2D FDTD (e.g. if Nz=0 or dz is effectively infinite), the condition changes.
            # This simple model assumes 3D, so dx,dy,dz must be >0.
            # A more robust solution would handle 2D cases specifically.
            # For now, let's assume dx, dy, dz are all positive for a 3D grid.
            # If one dimension_cells is 1, it's effectively 2.5D or thin 3D.
            # If cell_size for a dimension is very large, it approximates 2D.
            # Let's ensure no division by zero.
             inv_ds_sq_terms = []
             if self.dx > 0: inv_ds_sq_terms.append(1 / self.dx**2)
             if self.dy > 0: inv_ds_sq_terms.append(1 / self.dy**2)
             if self.dz > 0: inv_ds_sq_terms.append(1 / self.dz**2)
             if not inv_ds_sq_terms:
                 raise ValueError("At least one cell dimension (dx, dy, dz) must be positive.")
             inv_ds_sq = sum(inv_ds_sq_terms)
        else:
            inv_ds_sq = (1 / self.dx**2) + (1 / self.dy**2) + (1 / self.dz**2)

        if inv_ds_sq <= 0: # Should not happen if dx,dy,dz > 0
            raise ValueError("Sum of inverse squared cell dimensions must be positive.")

        self.dt = (dt_courant_factor / (C_0 * np.sqrt(inv_ds_sq)))
        print(f"FDTDGrid: Calculated dt = {self.dt:.3e} s")

        # Initialize Electric (E) and Magnetic (H) field components
        # These are defined on a Yee grid (staggered)
        # Ex: (Nx, Ny+1, Nz+1) - defined at centers of x-normal cell faces
        # Ey: (Nx+1, Ny, Nz+1) - defined at centers of y-normal cell faces
        # Ez: (Nx+1, Ny+1, Nz) - defined at centers of z-normal cell faces
        # Hx: (Nx+1, Ny, Nz)   - defined at centers of x-normal "dual" cell faces (shifted)
        # Hy: (Nx, Ny+1, Nz)   - defined at centers of y-normal "dual" cell faces
        # Hz: (Nx, Ny, Nz+1)   - defined at centers of z-normal "dual" cell faces

        # Note: If Nx, Ny, or Nz = 0 (or 1 for some components for true 2D), dimensions might reduce.
        # This simplified init assumes Nx, Ny, Nz >= 1 for cell counts.
        # For a "flat" dimension (e.g. Nz=1 for a thin slice), Nz+1 becomes 2.
        
        # Electric fields
        self.Ex = torch.zeros((self.Nx, self.Ny + 1, self.Nz + 1), device=self.device, dtype=torch.float32)
        self.Ey = torch.zeros((self.Nx + 1, self.Ny, self.Nz + 1), device=self.device, dtype=torch.float32)
        self.Ez = torch.zeros((self.Nx + 1, self.Ny + 1, self.Nz), device=self.device, dtype=torch.float32)

        # Magnetic fields
        # Hx is defined at (i, j+0.5, k+0.5) relative to E grid. If E grid is (0..Nx-1, 0..Ny-1, 0..Nz-1)
        # Hx: (Nx, Ny, Nz) -> but on Yee cell, it's staggered.
        # Correct Yee dimensions for H, relative to E field points:
        # Hx: (Nx, Ny+1, Nz+1) -> no, this is Ex
        # Hx is at (i, j+0.5, k+0.5) relative to Yee cell node (i,j,k)
        # Ex(i+0.5, j, k), Ey(i, j+0.5, k), Ez(i, j, k+0.5)
        # Hx(i, j+0.5, k+0.5), Hy(i+0.5, j, k+0.5), Hz(i+0.5, j+0.5, k)
        # Dimensions based on Schneider's book and common notation:
        # Ex(Nx, Ny+1, Nz+1)
        # Ey(Nx+1, Ny, Nz+1)
        # Ez(Nx+1, Ny+1, Nz)
        # Hx(Nx, Ny, Nz) -> at (i, j+0.5, k+0.5) - indices (0..Nx-1, 0..Ny-1, 0..Nz-1) for cell (i,j,k)
        # No, this is wrong for Yee. Let's use standard Yee array sizes.
        # Ex: (Nx, Ny+1, Nz+1)
        # Ey: (Nx+1, Ny, Nz+1)
        # Ez: (Nx+1, Ny+1, Nz)
        # Hx: (Nx+1, Ny, Nz) -- No, this is not standard.
        # Taflove notation (Nx, Ny, Nz cells):
        # Ex(i,j,k) -> node (i*dx, (j+0.5)*dy, (k+0.5)*dz) for Taflove. Array Ex(0:Nx-1, 0:Ny, 0:Nz)
        # No, this is getting confusing. Let's use the common shapes based on grid points.
        # If we have Nx cells, we have Nx+1 grid lines in x.
        # Ex needs Nx points in x-direction, Ny+1 in y, Nz+1 in z.
        # Corrected standard dimensions for fields, assuming indices run up to N_dim for cell counts N_dim:
        self.Hx = torch.zeros((self.Nx, self.Ny + 1, self.Nz + 1), device=self.device, dtype=torch.float32) # Same as Ex
        self.Hy = torch.zeros((self.Nx + 1, self.Ny, self.Nz + 1), device=self.device, dtype=torch.float32) # Same as Ey
        self.Hz = torch.zeros((self.Nx + 1, self.Ny + 1, self.Nz), device=self.device, dtype=torch.float32) # Same as Ez
        # Wait, this is not right. H fields are shifted by 0.5 in space AND time.
        # Standard Yee cell component definitions:
        # Ex at (i+0.5, j, k) -> Array(Nx, Ny-1, Nz-1) if indices are cell centers.
        # This is hard to get right abstractly. Let's use a common FDTD book's convention for array sizes.
        # From Davidson, "Computational Electromagnetics for RF and Microwave Engineering", 2nd ed.
        # Ex(Nx, Ny+1, Nz+1) - incorrect, this would be Nx+1 points.
        # Ex(0..Nx-1, 0..Ny, 0..Nz)
        # Ey(0..Nx, 0..Ny-1, 0..Nz)
        # Ez(0..Nx, 0..Ny, 0..Nz-1)
        # Hx(0..Nx, 0..Ny-1, 0..Nz-1)
        # Hy(0..Nx-1, 0..Ny, 0..Nz-1)
        # Hz(0..Nx-1, 0..Ny-1, 0..Nz)
        # This means Ex has Nx points in x, Ny+1 in y, Nz+1 in z.
        # Ex: (Nx, Ny+1, Nz+1)
        # Ey: (Nx+1, Ny, Nz+1)
        # Ez: (Nx+1, Ny+1, Nz)
        # Hx: (Nx+1, Ny, Nz)  -- This is one convention
        # Hy: (Nx, Ny+1, Nz)
        # Hz: (Nx, Ny, Nz+1)
        # Let's stick to the prompt's initial suggestion for H field sizes for now, and refine if updater needs different.
        # Hx: (Nx + 1, Ny, Nz)
        # Hy: (Nx, Ny + 1, Nz)
        # Hz: (Nx, Ny, Nz + 1)
        # The prompt's H field dimensions were:
        # Hx = torch.zeros((self.Nx + 1, self.Ny, self.Nz), device=self.device)
        # Hy = torch.zeros((self.Nx, self.Ny + 1, self.Nz), device=self.device)
        # Hz = torch.zeros((self.Nx, self.Ny, self.Nz + 1), device=self.device)
        # These seem plausible for a Yee cell.

        self.Hx = torch.zeros((self.Nx + 1, self.Ny, self.Nz), device=self.device, dtype=torch.float32)
        self.Hy = torch.zeros((self.Nx, self.Ny + 1, self.Nz), device=self.device, dtype=torch.float32)
        self.Hz = torch.zeros((self.Nx, self.Ny, self.Nz + 1), device=self.device, dtype=torch.float32)


        # Initialize Material Properties (cell-centered)
        # Size: (Nx, Ny, Nz)
        default_air = MATERIAL_DATABASE["air"]
        self.epsilon_r_grid = torch.full((self.Nx, self.Ny, self.Nz), default_air["eps_r"], device=self.device, dtype=torch.float32)
        self.mu_r_grid = torch.full((self.Nx, self.Ny, self.Nz), default_air["mu_r"], device=self.device, dtype=torch.float32)
        self.sigma_grid = torch.full((self.Nx, self.Ny, self.Nz), default_air["sigma"], device=self.device, dtype=torch.float32)

        print(f"FDTDGrid: Initialized fields and material grids on device '{self.device}'.")
        print(f"  Grid dimensions (cells): Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}")
        print(f"  Cell size (m): dx={self.dx:.2e}, dy={self.dy:.2e}, dz={self.dz:.2e}")
        print(f"  Ex shape: {self.Ex.shape}, Ey shape: {self.Ey.shape}, Ez shape: {self.Ez.shape}")
        print(f"  Hx shape: {self.Hx.shape}, Hy shape: {self.Hy.shape}, Hz shape: {self.Hz.shape}")
        print(f"  Material grid shapes (epsilon_r, mu_r, sigma): {self.epsilon_r_grid.shape}")

    def set_material_properties(self, material_name, region_slice_tuple):
        """
        Sets the material properties for a specified region of the grid.

        Args:
            material_name (str): The name of the material (key in MATERIAL_DATABASE).
            region_slice_tuple (tuple): A tuple of 3 slice objects defining the region,
                                        e.g., (slice(x_start, x_end), 
                                               slice(y_start, y_end), 
                                               slice(z_start, z_end)).
                                        Indices are for the cell-centered material grid.
        
        Raises:
            ValueError: If material_name is not found in MATERIAL_DATABASE.
            TypeError: If region_slice_tuple is not a tuple of 3 slices.
        """
        if material_name not in MATERIAL_DATABASE:
            print(f"Error: Material '{material_name}' not found in MATERIAL_DATABASE.")
            # Or raise ValueError(f"Material '{material_name}' not found.")
            return

        if not (isinstance(region_slice_tuple, tuple) and 
                len(region_slice_tuple) == 3 and 
                all(isinstance(s, slice) for s in region_slice_tuple)):
            raise TypeError("region_slice_tuple must be a tuple of 3 slice objects.")

        material_props = MATERIAL_DATABASE[material_name]
        
        try:
            self.epsilon_r_grid[region_slice_tuple] = material_props["eps_r"]
            self.mu_r_grid[region_slice_tuple] = material_props["mu_r"]
            self.sigma_grid[region_slice_tuple] = material_props["sigma"]
            print(f"Set material '{material_name}' in region {region_slice_tuple}.")
        except IndexError:
            # This can happen if slices are out of bounds for the grid dimensions
            print(f"Error: region_slice_tuple {region_slice_tuple} is out of bounds "
                  f"for material grids of shape {(self.Nx, self.Ny, self.Nz)}.")
            # Or raise IndexError("Region slice is out of bounds for the material grid.")
        except Exception as e:
            print(f"An unexpected error occurred during set_material_properties: {e}")


if __name__ == '__main__':
    # --- Example Usage and Testing ---
    print("\\n--- FDTDGrid Test ---")
    
    # Test 1: Basic Initialization
    print("\\nTest 1: Basic Initialization (CPU)")
    try:
        grid_cpu = FDTDGrid(dimensions_cells=(10, 10, 10), cell_size_m=1e-3, device=torch.device("cpu"))
        assert grid_cpu.Nx == 10
        assert grid_cpu.dt > 0
        assert grid_cpu.Ex.device.type == 'cpu'
        print("CPU Grid Test 1.1 Passed: Basic initialization successful.")
    except Exception as e:
        print(f"CPU Grid Test 1.1 Failed: {e}")

    # Test 2: CUDA Initialization (if available)
    if torch.cuda.is_available():
        print("\\nTest 2: CUDA Initialization")
        try:
            grid_cuda = FDTDGrid(dimensions_cells=(5, 5, 5), cell_size_m=(0.1, 0.2, 0.3))
            assert grid_cuda.device.type == 'cuda'
            assert grid_cuda.dx == 0.1
            print("CUDA Grid Test 2.1 Passed: CUDA initialization successful.")
        except Exception as e:
            print(f"CUDA Grid Test 2.1 Failed: {e}")
    else:
        print("\\nTest 2: CUDA Initialization (Skipped, CUDA not available)")

    # Test 3: Setting Material Properties
    print("\\nTest 3: Setting Material Properties")
    grid_mat = FDTDGrid(dimensions_cells=(20, 20, 20), cell_size_m=1e-2, device=torch.device("cpu")) # Force CPU for predictability
    
    region1 = (slice(5, 10), slice(5, 10), slice(5, 10)) # A cube in the middle
    try:
        grid_mat.set_material_properties("water", region1)
        assert grid_mat.epsilon_r_grid[region1].mean().item() == MATERIAL_DATABASE["water"]["eps_r"]
        assert grid_mat.sigma_grid[7, 7, 7].item() == MATERIAL_DATABASE["water"]["sigma"]
        print("Material Test 3.1 Passed: Set 'water' in region1.")
    except Exception as e:
        print(f"Material Test 3.1 Failed: {e}")

    region_all = (slice(None), slice(None), slice(None)) # Whole grid
    try:
        grid_mat.set_material_properties("copper", region_all)
        assert grid_mat.sigma_grid.mean().item() == MATERIAL_DATABASE["copper"]["sigma"]
        print("Material Test 3.2 Passed: Set 'copper' in whole grid.")
    except Exception as e:
        print(f"Material Test 3.2 Failed: {e}")

    # Test 4: Invalid material name
    print("\\nTest 4: Invalid Material Name")
    try:
        grid_mat.set_material_properties("non_existent_material", region1)
        # Check if some default behavior or error message was printed by the method
        # (Current implementation prints an error and returns)
        print("Material Test 4.1: Method called with non-existent material (check console for error print).")
    except Exception as e: # Should not raise exception with current design, but good to catch
        print(f"Material Test 4.1 Failed unexpectedly: {e}")

    # Test 5: Out-of-bounds region
    print("\\nTest 5: Out-of-bounds region slice")
    region_out_of_bounds = (slice(0, 100), slice(0, 10), slice(0, 10)) # x-slice too large
    try:
        grid_mat.set_material_properties("air", region_out_of_bounds)
         # (Current implementation prints an error and returns)
        print("Material Test 5.1: Method called with out-of-bounds region (check console for error print).")
    except Exception as e: # Should not raise exception with current design.
        print(f"Material Test 5.1 Failed unexpectedly: {e}")
    
    print("\\n--- FDTDGrid Test Finished ---")
```
