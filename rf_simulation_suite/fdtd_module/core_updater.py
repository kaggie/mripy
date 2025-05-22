import torch

# Assuming grid.py is in the same directory or package
try:
    from .grid import MU_0, EPSILON_0 # Relative import for package structure
except ImportError:
    # Fallback for direct script execution (e.g., if file is run from this directory)
    from grid import MU_0, EPSILON_0


class FDTDUpdater:
    """
    Handles the core FDTD field update equations (E and H fields).
    It uses coefficients derived from material properties defined on an FDTDGrid.
    Material properties are cell-centered. For updating a field component,
    the material coefficients from the primary cell (i,j,k) associated with
    that component's location are used (as per the simplification).
    """

    def __init__(self, grid):
        """
        Initializes the FDTDUpdater.

        Args:
            grid (FDTDGrid): An instance of the FDTDGrid class, containing
                             the grid parameters, material properties, and field arrays.
        """
        self.grid = grid
        self.device = grid.device
        self.dt = grid.dt
        self.dx = grid.dx
        self.dy = grid.dy
        self.dz = grid.dz

        # --- Calculate Update Coefficients (Cell-Centered: Nx, Ny, Nz) ---
        # Material properties from grid are (Nx, Ny, Nz)
        epsilon_eff = EPSILON_0 * self.grid.epsilon_r_grid # Effective permittivity
        sigma_eff = self.grid.sigma_grid                # Effective conductivity
        
        denominator_E = (2 * epsilon_eff + sigma_eff * self.dt)
        denominator_E[denominator_E == 0] = 1e-30 # Avoid NaN/Inf

        self.Ca_E = (2 * epsilon_eff - sigma_eff * self.dt) / denominator_E
        self.Cb_E = (2 * self.dt) / denominator_E
        
        mu_eff = MU_0 * self.grid.mu_r_grid # Effective permeability
        # Assuming magnetic conductivity (sigma_m) is zero for Ca_H = 1
        self.Ca_H = torch.ones_like(self.grid.mu_r_grid, device=self.device, dtype=torch.float32)
        
        mu_eff_safe = mu_eff.clone() 
        mu_eff_safe[mu_eff_safe == 0] = 1e-30 
        self.Cb_H = self.dt / mu_eff_safe

        print("FDTDUpdater: Coefficients Ca_E, Cb_E, Ca_H, Cb_H calculated.")
        print(f"  Device: {self.device}")
        print(f"  Ca_E shape: {self.Ca_E.shape}, Cb_E shape: {self.Cb_E.shape}")
        print(f"  Ca_H shape: {self.Ca_H.shape}, Cb_H shape: {self.Cb_H.shape}")


    def update_H_fields(self):
        """
        Updates the H-field components (Hx, Hy, Hz) for one time step.
        H_new = Ca_H * H_old - Cb_H * curl(E)
        
        Field array dimensions from FDTDGrid:
        Ex(Nx, Ny+1, Nz+1), Ey(Nx+1, Ny, Nz+1), Ez(Nx+1, Ny+1, Nz)
        Hx(Nx+1, Ny, Nz),   Hy(Nx, Ny+1, Nz),   Hz(Nx, Ny, Nz+1)
        Coefficient arrays Ca_H, Cb_H are (Nx, Ny, Nz).
        Updates are for interior points. Boundary conditions are handled separately.
        Slicing for coefficients matches the field component region they apply to.
        """
        g = self.grid

        # --- Hx Update ---
        # Hx is (Nx+1, Ny, Nz). Curl E for Hx is (dEz/dy - dEy/dz).
        # These derivatives are calculated at Hx locations.
        # Hx[i,j,k] (at x_i, y_{j+0.5}, z_{k+0.5}) uses coefficients from cells (i-1,j,k) or (i,j,k).
        # Simplified: Hx[i,j,k] (0<=i<Nx) uses coeffs from cell (i,j,k). Hx[Nx,j,k] from (Nx-1,j,k).
        if g.Ny > 0 and g.Nz > 0: # Grid must have depth for curl
            # Derivatives for Hx[i, j, k] (i=0..Nx, j=0..Ny-1, k=0..Nz-1)
            dEz_dy = (g.Ez[:g.Nx+1, 1:g.Ny+1, :g.Nz] - g.Ez[:g.Nx+1, :g.Ny, :g.Nz]) / self.dy
            dEy_dz = (g.Ey[:g.Nx+1, :g.Ny, 1:g.Nz+1] - g.Ey[:g.Nx+1, :g.Ny, :g.Nz]) / self.dz
            curl_E_x = dEz_dy - dEy_dz # Shape (Nx+1, Ny, Nz) matching Hx
            
            # Apply coefficients. Ca_H and Cb_H are (Nx,Ny,Nz).
            g.Hx[:g.Nx, :, :] = self.Ca_H * g.Hx[:g.Nx, :, :] - self.Cb_H * curl_E_x[:g.Nx,:,:]
            if g.Nx > 0 : 
                 g.Hx[g.Nx, :, :] = self.Ca_H[-1,:,:] * g.Hx[g.Nx, :, :] - self.Cb_H[-1,:,:] * curl_E_x[g.Nx,:,:]

        # --- Hy Update ---
        # Hy is (Nx, Ny+1, Nz). Curl E for Hy is (dEx/dz - dEz/dx).
        if g.Nx > 0 and g.Nz > 0:
            dEx_dz = (g.Ex[:g.Nx, :g.Ny+1, 1:g.Nz+1] - g.Ex[:g.Nx, :g.Ny+1, :g.Nz]) / self.dz
            dEz_dx = (g.Ez[1:g.Nx+1, :g.Ny+1, :g.Nz] - g.Ez[:g.Nx, :g.Ny+1, :g.Nz]) / self.dx
            curl_E_y = dEx_dz - dEz_dx # Shape (Nx, Ny+1, Nz) matching Hy
            g.Hy[:, :g.Ny, :] = self.Ca_H * g.Hy[:, :g.Ny, :] - self.Cb_H * curl_E_y[:,:g.Ny,:]
            if g.Ny > 0:
                g.Hy[:, g.Ny, :] = self.Ca_H[:,-1,:] * g.Hy[:, g.Ny, :] - self.Cb_H[:,-1,:] * curl_E_y[:,g.Ny,:]
            
        # --- Hz Update ---
        # Hz is (Nx, Ny, Nz+1). Curl E for Hz is (dEy/dx - dEx/dy).
        if g.Nx > 0 and g.Ny > 0:
            dEy_dx = (g.Ey[1:g.Nx+1, :g.Ny, :g.Nz+1] - g.Ey[:g.Nx, :g.Ny, :g.Nz+1]) / self.dx
            dEx_dy = (g.Ex[:g.Nx, 1:g.Ny+1, :g.Nz+1] - g.Ex[:g.Nx, :g.Ny, :g.Nz+1]) / self.dy
            curl_E_z = dEy_dx - dEx_dy # Shape (Nx, Ny, Nz+1) matching Hz
            g.Hz[:, :, :g.Nz] = self.Ca_H * g.Hz[:, :, :g.Nz] - self.Cb_H * curl_E_z[:,:,:g.Nz]
            if g.Nz > 0:
                g.Hz[:, :, g.Nz] = self.Ca_H[:,:,-1] * g.Hz[:, :, g.Nz] - self.Cb_H[:,:,-1] * curl_E_z[:,:,g.Nz]


    def update_E_fields(self):
        """
        Updates the E-field components (Ex, Ey, Ez) for one time step.
        E_new = Ca_E * E_old + Cb_E * curl(H)
        Coeffs Ca_E, Cb_E are (Nx,Ny,Nz)
        """
        g = self.grid

        # --- Ex Update ---
        # Ex is (Nx, Ny+1, Nz+1). Curl H for Ex = (dHz/dy - dHy/dz).
        # Ex[i,j,k] (at x_{i+0.5}, y_j, z_k) uses material coeffs from cell (i,j,k).
        if g.Ny > 0 and g.Nz > 0:
            # dHz/dy: Hz is (Nx,Ny,Nz+1). (Hz[i,j,k] - Hz[i,j-1,k])/dy for Ex[i,j,k]
            # dHy/dz: Hy is (Nx,Ny+1,Nz). (Hy[i,j,k] - Hy[i,j,k-1])/dz for Ex[i,j,k]
            # These derivatives are at Ex locations.
            # For Ex[i, j=1..Ny, k=1..Nz]
            dHz_dy = (g.Hz[:g.Nx, 1:g.Ny+1, :g.Nz+1] - g.Hz[:g.Nx, :g.Ny, :g.Nz+1]) / self.dy
            dHy_dz = (g.Hy[:g.Nx, :g.Ny+1, 1:g.Nz+1] - g.Hy[:g.Nx, :g.Ny+1, :g.Nz]) / self.dz
            curl_H_x = dHz_dy - dHy_dz # Shape (Nx, Ny, Nz)
            # Update Ex[i, 1:Ny, 1:Nz] region, which is (Nx, Ny-1, Nz-1)
            # Coefficients Ca_E, Cb_E are (Nx,Ny,Nz). Slice them to match.
            g.Ex[:, 1:-1, 1:-1] = self.Ca_E[:, :-1, :-1] * g.Ex[:, 1:-1, 1:-1] + \
                                  self.Cb_E[:, :-1, :-1] * curl_H_x[:, :-1, :-1]

        # --- Ey Update ---
        # Ey is (Nx+1, Ny, Nz+1). Curl H for Ey = (dHx/dz - dHz/dx).
        if g.Nx > 0 and g.Nz > 0:
            dHx_dz = (g.Hx[:g.Nx+1, :g.Ny, 1:g.Nz+1] - g.Hx[:g.Nx+1, :g.Ny, :g.Nz]) / self.dz
            dHz_dx = (g.Hz[1:g.Nx+1, :g.Ny, :g.Nz+1] - g.Hz[:g.Nx, :g.Ny, :g.Nz+1]) / self.dx
            curl_H_y = dHx_dz - dHz_dx # Shape (Nx, Ny, Nz)
            # Update Ey[1:Nx, j, 1:Nz] region, which is (Nx-1, Ny, Nz-1)
            g.Ey[1:-1, :, 1:-1] = self.Ca_E[:-1, :, :-1] * g.Ey[1:-1, :, 1:-1] + \
                                  self.Cb_E[:-1, :, :-1] * curl_H_y[:-1, :, :-1]
                                   
        # --- Ez Update ---
        # Ez is (Nx+1, Ny+1, Nz). Curl H for Ez = (dHy/dx - dHx/dy).
        if g.Nx > 0 and g.Ny > 0:
            dHy_dx = (g.Hy[1:g.Nx+1, :g.Ny+1, :g.Nz] - g.Hy[:g.Nx, :g.Ny+1, :g.Nz]) / self.dx
            dHx_dy = (g.Hx[:g.Nx+1, 1:g.Ny+1, :g.Nz] - g.Hx[:g.Nx+1, :g.Ny, :g.Nz]) / self.dy
            curl_H_z = dHy_dx - dHx_dy # Shape (Nx, Ny, Nz)
            # Update Ez[1:Nx, 1:Ny, k] region, which is (Nx-1, Ny-1, Nz)
            g.Ez[1:-1, 1:-1, :] = self.Ca_E[:-1, :-1, :] * g.Ez[1:-1, 1:-1, :] + \
                                  self.Cb_E[:-1, :-1, :] * curl_H_z[:-1, :-1, :]


if __name__ == '__main__':
    from grid import FDTDGrid 
    print("\\n--- FDTDUpdater Test (Corrected Vectorized & Simplified Coeffs) ---")

    dims = (10, 11, 12) # Nx, Ny, Nz cells
    cell_size = 1e-3 
    test_grid = FDTDGrid(dimensions_cells=dims, cell_size_m=cell_size, device=torch.device("cpu"))
    
    source_val = 1.0
    # Ex is (Nx, Ny+1, Nz+1)
    source_idx_x, source_idx_y, source_idx_z = dims[0]//2, dims[1]//2, dims[2]//2
    test_grid.Ex[source_idx_x, source_idx_y, source_idx_z] = source_val

    updater = FDTDUpdater(test_grid)
    print(f"Updater initialized. dt={updater.dt:.3e}")

    num_steps = 20
    print(f"Performing {num_steps} FDTD steps...")
    max_Ex_history = []
    max_Hz_history = [] 

    for step in range(num_steps):
        updater.update_H_fields()
        # Apply H-field Boundary Conditions here if any (e.g., PEC, PML)
        
        updater.update_E_fields()
        # Apply E-field Boundary Conditions here if any
        
        # Re-apply hard source for Ex
        test_grid.Ex[source_idx_x, source_idx_y, source_idx_z] = source_val 

        if (step + 1) % 2 == 0:
            max_Ex_val = test_grid.Ex.abs().max().item()
            max_Hz_val = test_grid.Hz.abs().max().item()
            max_Ex_history.append(max_Ex_val)
            max_Hz_history.append(max_Hz_val)
            print(f"Step {step+1}: Max |Ex|={max_Ex_val:.2e}, Max |Hz|={max_Hz_val:.2e}")

    print("\\n--- FDTDUpdater Test (Corrected Vectorized & Simplified Coeffs) Finished ---")
    
    final_source_Ex_val = test_grid.Ex[source_idx_x, source_idx_y, source_idx_z].item()
    assert abs(final_source_Ex_val - source_val) < 1e-6, \
        f"Ex at source location changed. Expected: {source_val}, Got: {final_source_Ex_val}"
    
    assert max_Hz_history[-1] > 1e-9, f"Hz field was not significantly excited. Max Hz: {max_Hz_history[-1]}"
    
    other_E_excited = test_grid.Ey.abs().max().item() > 1e-9 or test_grid.Ez.abs().max().item() > 1e-9
    other_H_excited = test_grid.Hx.abs().max().item() > 1e-9 or test_grid.Hy.abs().max().item() > 1e-9
    
    assert other_H_excited, "Neither Hx nor Hy fields (expected from Ex source) seem to have been significantly excited."
    print(f"Max Hx: {test_grid.Hx.abs().max().item():.2e}, Max Hy: {test_grid.Hy.abs().max().item():.2e}")
    print(f"Max Ey: {test_grid.Ey.abs().max().item():.2e}, Max Ez: {test_grid.Ez.abs().max().item():.2e}")

    print(f"Max Hz values during simulation: {max_Hz_history}")
    print("Basic field excitation checks passed with corrected vectorized implementation.")

```
