import torch

# Assuming grid.py is in the same directory or package
try:
    from .grid import C_0 # Relative import for package structure
except ImportError:
    # Fallback for direct script execution
    from grid import C_0


class MurABC3D:
    """
    Implements Mur's first-order Absorbing Boundary Conditions (ABC) for 3D FDTD simulations.
    This class stores previous E-field values and applies the ABC to the tangential
    E-field components on the six faces of the simulation domain.
    """

    def __init__(self, grid):
        """
        Initializes the MurABC3D boundary condition object.

        Args:
            grid (FDTDGrid): The FDTDGrid instance containing grid parameters and fields.
        """
        self.grid = grid
        self.device = grid.device # Assuming grid has a device attribute
        self.dt = grid.dt
        self.dx = grid.dx
        self.dy = grid.dy
        self.dz = grid.dz

        # Precompute coefficients for Mur's ABC
        # coef = (c0*dt/ds - 1) / (c0*dt/ds + 1) where ds is dx, dy, or dz
        # Avoid division by zero if dx, dy, or dz is zero (though unlikely for 3D grid)
        self.coef_x = (C_0 * self.dt / self.dx - 1) / (C_0 * self.dt / self.dx + 1) if self.dx > 0 else 0.0
        self.coef_y = (C_0 * self.dt / self.dy - 1) / (C_0 * self.dt / self.dy + 1) if self.dy > 0 else 0.0
        self.coef_z = (C_0 * self.dt / self.dz - 1) / (C_0 * self.dt / self.dz + 1) if self.dz > 0 else 0.0
        
        # History of E-fields (E at time n-1 and n-2)
        # These will be populated by store_previous_fields()
        self.Ex_p, self.Ey_p, self.Ez_p = None, None, None  # Fields at time step n
        self.Ex_pp, self.Ey_pp, self.Ez_pp = None, None, None # Fields at time step n-1

        print(f"MurABC3D initialized. Coefs: x={self.coef_x:.3f}, y={self.coef_y:.3f}, z={self.coef_z:.3f}")

    def store_previous_fields(self):
        """
        Stores the E-field values from the current time step (n) to be used as
        previous values (n-1, n-2) in the next ABC application.
        This method should be called *before* the main E-field update loop
        (i.e., when grid.Ex, Ey, Ez hold values at time step 'n').
        """
        # Store E_fields(n-1) -> E_fields(n-2)
        if self.Ex_p is not None:
            self.Ex_pp = self.Ex_p.clone()
        else: # First or second time step, initialize _pp with zeros
            self.Ex_pp = torch.zeros_like(self.grid.Ex, device=self.device)
        
        if self.Ey_p is not None:
            self.Ey_pp = self.Ey_p.clone()
        else:
            self.Ey_pp = torch.zeros_like(self.grid.Ey, device=self.device)

        if self.Ez_p is not None:
            self.Ez_pp = self.Ez_p.clone()
        else:
            self.Ez_pp = torch.zeros_like(self.grid.Ez, device=self.device)

        # Store E_fields(n) -> E_fields(n-1)
        self.Ex_p = self.grid.Ex.clone()
        self.Ey_p = self.grid.Ey.clone()
        self.Ez_p = self.grid.Ez.clone()

    def apply_E_boundary(self):
        """
        Applies Mur's first-order ABC to the tangential E-field components on all
        six faces of the simulation domain.
        This method should be called *after* the main E-field update loop
        (i.e., when grid.Ex, Ey, Ez hold values at time step 'n+1').

        The Mur ABC formula is: E_0^{n+1} = E_1^{n} + coef * (E_1^{n+1} - E_0^{n})
        where E_0 is the boundary field point, E_1 is the adjacent interior point.
        _p denotes values at time n. _pp values at n-1 (not used in 1st order Mur).
        The prompt's 1D example seems to be a slightly different form or interpretation.
        Let's use the standard form: E_0^{n+1} = E_1^{n} + \frac{c\Delta t - \Delta x}{c\Delta t + \Delta x} (E_1^{n+1} - E_0^{n})
        This is what self.coef_x etc. represent.
        The E-fields in self.grid.Ex etc. are at n+1. self.Ex_p etc. are at n.
        """
        g = self.grid
        Nx, Ny, Nz = g.Nx, g.Ny, g.Nz # Number of cells

        # Check if previous field values are available (i.e., after at least one call to store_previous_fields)
        if self.Ex_p is None or self.Ey_p is None or self.Ez_p is None:
            # This implies it's the very first E-update, ABCs cannot be applied yet.
            # Or, if store_previous_fields was not called after the first E-update.
            # For Mur's 1st order, we only need E(n+1) and E(n). E(n-1) is for 2nd order.
            # The prompt's store_previous_fields stores E(n) into _p and E(n-1) into _pp.
            # So, E(n+1) is grid.Ex, E(n) is self.Ex_p. This is sufficient.
            print("MurABC3D: Skipping boundary application, previous field values not yet stored sufficiently.")
            return

        # --- X-boundaries ---
        # x_min boundary (i=0 for tangential fields Ey[0,:,:], Ez[0,:,:])
        # Ey is (Nx+1, Ny, Nz+1). Ey[0,j,k] for j=0..Ny-1, k=0..Nz
        # Ez is (Nx+1, Ny+1, Nz). Ez[0,j,k] for j=0..Ny, k=0..Nz-1
        if self.dx > 0: # Only apply if there's a dimension in x
            # Ey components tangential to x_min face
            g.Ey[0, :, :] = self.Ey_p[1, :, :] + self.coef_x * (g.Ey[1, :, :] - self.Ey_p[0, :, :])
            # Ez components tangential to x_min face
            g.Ez[0, :, :] = self.Ez_p[1, :, :] + self.coef_x * (g.Ez[1, :, :] - self.Ez_p[0, :, :])

            # x_max boundary (i=Nx for Ey[Nx,:,:], Ez[Nx,:,:])
            g.Ey[Nx, :, :] = self.Ey_p[Nx-1, :, :] + self.coef_x * (g.Ey[Nx-1, :, :] - self.Ey_p[Nx, :, :])
            g.Ez[Nx, :, :] = self.Ez_p[Nx-1, :, :] + self.coef_x * (g.Ez[Nx-1, :, :] - self.Ez_p[Nx, :, :])

        # --- Y-boundaries ---
        # y_min boundary (j=0 for tangential fields Ex[:,0,:], Ez[:,0,:])
        # Ex is (Nx, Ny+1, Nz+1). Ex[i,0,k] for i=0..Nx-1, k=0..Nz
        # Ez is (Nx+1, Ny+1, Nz). Ez[i,0,k] for i=0..Nx, k=0..Nz-1
        if self.dy > 0:
            # Ex components tangential to y_min face
            g.Ex[:, 0, :] = self.Ex_p[:, 1, :] + self.coef_y * (g.Ex[:, 1, :] - self.Ex_p[:, 0, :])
            # Ez components tangential to y_min face
            # Note: Ez field array is (Nx+1, Ny+1, Nz). Ez[:,0,:] is (Nx+1, Nz)
            g.Ez[:, 0, :] = self.Ez_p[:, 1, :] + self.coef_y * (g.Ez[:, 1, :] - self.Ez_p[:, 0, :])

            # y_max boundary (j=Ny for Ex[:,Ny,:], Ez[:,Ny,:])
            g.Ex[:, Ny, :] = self.Ex_p[:, Ny-1, :] + self.coef_y * (g.Ex[:, Ny-1, :] - self.Ex_p[:, Ny, :])
            g.Ez[:, Ny, :] = self.Ez_p[:, Ny-1, :] + self.coef_y * (g.Ez[:, Ny-1, :] - self.Ez_p[:, Ny, :])

        # --- Z-boundaries ---
        # z_min boundary (k=0 for tangential fields Ex[:,:,0], Ey[:,:,0])
        # Ex is (Nx, Ny+1, Nz+1). Ex[i,j,0] for i=0..Nx-1, j=0..Ny
        # Ey is (Nx+1, Ny, Nz+1). Ey[i,j,0] for i=0..Nx, j=0..Ny-1
        if self.dz > 0:
            # Ex components tangential to z_min face
            g.Ex[:, :, 0] = self.Ex_p[:, :, 1] + self.coef_z * (g.Ex[:, :, 1] - self.Ex_p[:, :, 0])
            # Ey components tangential to z_min face
            g.Ey[:, :, 0] = self.Ey_p[:, :, 1] + self.coef_z * (g.Ey[:, :, 1] - self.Ey_p[:, :, 0])

            # z_max boundary (k=Nz for Ex[:,:,Nz], Ey[:,:,Nz])
            g.Ex[:, :, Nz] = self.Ex_p[:, :, Nz-1] + self.coef_z * (g.Ex[:, :, Nz-1] - self.Ex_p[:, :, Nz])
            g.Ey[:, :, Nz] = self.Ey_p[:, :, Nz-1] + self.coef_z * (g.Ey[:, :, Nz-1] - self.Ey_p[:, :, Nz])
        
        # Note on Edges and Corners:
        # The above formulation applies the 1D Mur ABC along each face.
        # For example, Ey[0,j,k] is updated by x_min ABC. If j=0, Ey[0,0,k] is on an edge.
        # It will be updated by x_min ABC. Then, if Ex[i,0,k] is updated by y_min ABC, Ex[0,0,k] (another edge component)
        # is also updated. This sequential application on faces means edge and corner components
        # might be updated multiple times if not careful or if slices overlap, or some components might be missed
        # if slicing is strictly interior for one update then boundary for another.
        # The current slicing applies to the entire face, so edge components are updated by each relevant face condition.
        # E.g., Ey[0,0,k] is updated by x_min condition. Ex[0,0,k] by y_min. Ez[0,0,0] by x_min, y_min, z_min.
        # This is a common way to handle it, though more sophisticated treatments exist.

    def apply_H_boundary(self):
        """
        Applies boundary conditions to H-field components.
        For Mur's first-order ABC, this is often implicitly handled by the E-field updates
        and subsequent H-field updates from the main FDTD loop using the corrected E-fields.
        This method can be a pass-through or implement specific H-field ABCs if needed.
        """
        pass # H-fields are typically not directly set by simple Mur ABCs.


if __name__ == '__main__':
    from grid import FDTDGrid # For testing
    print("\\n--- MurABC3D Test ---")

    dims = (5, 6, 7) # Nx, Ny, Nz cells
    cell_size = 1e-3 
    grid = FDTDGrid(dimensions_cells=dims, cell_size_m=cell_size, device=torch.device("cpu"))
    abc = MurABC3D(grid)

    # Simulate a few steps to populate history
    print("Simulating initial steps to populate history...")
    grid.Ex[:,:,:] = torch.rand_like(grid.Ex) # Fill with some initial random values
    grid.Ey[:,:,:] = torch.rand_like(grid.Ey)
    grid.Ez[:,:,:] = torch.rand_like(grid.Ez)
    
    abc.store_previous_fields() # Stores current (n=0) as _p, initializes _pp to zeros
    
    # Simulate E-field update (e.g. by core_updater or manually for test)
    # E-fields in grid are now at n+1 (or n=1 if starting from 0)
    grid.Ex *= 0.5 # Dummy update
    grid.Ey *= 0.5
    grid.Ez *= 0.5
    
    # Store these "n=1" values into _p, and "n=0" values from _p into _pp
    abc.store_previous_fields() 

    # Simulate another E-field update. E-fields in grid are now n+2 (or n=2)
    grid.Ex *= 0.5 
    grid.Ey *= 0.5
    grid.Ez *= 0.5
    
    print("Applying E-boundary conditions...")
    # Store current fields (n=2) to _p, and previous _p (n=1) to _pp.
    # This is what would happen BEFORE E-update if we were in a loop.
    # For apply_E_boundary, it expects grid.Ex to be n+1, and self.Ex_p to be n.
    # So, the sequence should be:
    # 1. E_main_update() -> grid.Ex is at n+1
    # 2. abc.apply_E_boundary() -> uses grid.Ex (n+1) and abc.Ex_p (n)
    # 3. abc.store_previous_fields() -> saves grid.Ex (n+1) into abc.Ex_p for next iteration.

    # Let's reset to a clearer sequence for testing apply_E_boundary directly:
    # Assume grid.Ex, Ey, Ez are current values (n+1) AFTER main FDTD E-update.
    # abc.Ex_p, Ey_p, Ez_p hold values from time step n.
    # abc.Ex_pp, Ey_pp, Ez_pp hold values from time step n-1.
    
    # Re-initialize for clarity in test
    grid.Ex.fill_(1.0); grid.Ey.fill_(1.0); grid.Ez.fill_(1.0) # Current state (n+1)
    abc.Ex_p = torch.full_like(grid.Ex, 0.5); abc.Ey_p = torch.full_like(grid.Ey, 0.5); abc.Ez_p = torch.full_like(grid.Ez, 0.5) # State at n
    # abc.Ex_pp, etc., are not strictly needed for 1st order Mur.

    # Capture a value before ABC
    val_before_Ey_xmin = grid.Ey[0, dims[1]//2, dims[2]//2].clone()
    val_before_Ex_ymin = grid.Ex[dims[0]//2, 0, dims[2]//2].clone()
    val_before_Ex_zmin = grid.Ex[dims[0]//2, dims[1]//2, 0].clone()


    abc.apply_E_boundary()
    print("E-boundary conditions applied.")

    val_after_Ey_xmin = grid.Ey[0, dims[1]//2, dims[2]//2].item()
    val_after_Ex_ymin = grid.Ex[dims[0]//2, 0, dims[2]//2].item()
    val_after_Ex_zmin = grid.Ex[dims[0]//2, dims[1]//2, 0].item()

    print(f"  Ey[0,mid,mid]: Before={val_before_Ey_xmin.item():.4f}, After={val_after_Ey_xmin:.4f}")
    print(f"  Ex[mid,0,mid]: Before={val_before_Ex_ymin.item():.4f}, After={val_after_Ex_ymin:.4f}")
    print(f"  Ex[mid,mid,0]: Before={val_before_Ex_zmin.item():.4f}, After={val_after_Ex_zmin:.4f}")

    # Check if values changed (they should, unless coef is 0 or fields were already satisfying condition)
    if abc.coef_x != 0 : assert abs(val_before_Ey_xmin.item() - val_after_Ey_xmin) > 1e-5, "Ey at x_min did not change."
    if abc.coef_y != 0 : assert abs(val_before_Ex_ymin.item() - val_after_Ex_ymin) > 1e-5, "Ex at y_min did not change."
    if abc.coef_z != 0 : assert abs(val_before_Ex_zmin.item() - val_after_Ex_zmin) > 1e-5, "Ex at z_min did not change."
    
    print("MurABC3D basic application test passed.")
    print("\\n--- MurABC3D Test Finished ---")

```
