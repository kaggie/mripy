import torch
from .isochromat import Isochromat
from mri_simulator.utils.constants import GAMMA_RAD_S_T_PROTON, B0_TESLA

class BlochSolver:
    """
    Solves the Bloch equations for a collection of isochromats using matrix exponentiation.
    """
    def __init__(self, isochromats: Isochromat, dt: float):
        """
        Initializes the BlochSolver.

        Args:
            isochromats (Isochromat): The isochromats to simulate.
            dt (float): The time step for the simulation in seconds.
        """
        self.isochromats = isochromats
        self.dt = dt
        self.device = isochromats.device
        self.dtype = isochromats.dtype
        self.num_isochromats = isochromats.magnetization.shape[0]

        # Pre-calculate chemical shift in rad/s
        self.chemical_shift_rad_s = isochromats.chemical_shift_hz * 2 * torch.pi
        
        # Ensure relaxation times are positive to avoid division by zero or issues with log(0) in other contexts
        # Clamping here also ensures they are PyTorch tensors on the correct device.
        self.T1 = torch.clamp(isochromats.T1.to(device=self.device, dtype=self.dtype), min=1e-6)
        self.T2_star = torch.clamp(isochromats.T2_star.to(device=self.device, dtype=self.dtype), min=1e-6)
        self.M0 = isochromats.M0.to(device=self.device, dtype=self.dtype) # This is total M0
        # For non-MT simulations, M0_free is the same as M0.
        # For the free pool in an MT context (if BlochSolver were used), M0_free is correct.
        self.M0_div_T1 = self.isochromats.M0_free / self.T1


    def run_step(self, b1_complex: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """
        Performs one time step of the Bloch equation simulation.

        Args:
            b1_complex (torch.Tensor): Complex RF pulse B1 (Bx_lab + iBy_lab) in Tesla.
                Shape: (N,) or (1,). Applied to all isochromats if (1,).
            gradients (torch.Tensor): Gradient vector [Gx, Gy, Gz] in Tesla/meter.
                Shape: (N, 3) or (1, 3). Applied to all isochromats if (1,3).

        Returns:
            torch.Tensor: Updated magnetization [Mx, My, Mz] for each isochromat.
                Shape: (N, 3).
        """
        # Ensure inputs are on the correct device and dtype, and expand if necessary
        if b1_complex.ndim == 0 or (b1_complex.ndim == 1 and b1_complex.shape[0] == 1): # Handle scalar or single element tensor
            b1_complex = b1_complex.expand(self.num_isochromats)
        b1_complex = b1_complex.to(device=self.device, dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128) # Match precision
        
        if gradients.ndim == 1 and gradients.shape[0] == 3: # Single gradient for all
             gradients = gradients.unsqueeze(0).expand(self.num_isochromats, 3)
        elif gradients.ndim == 2 and gradients.shape[0] == 1: # Already (1,3)
            gradients = gradients.expand(self.num_isochromats, 3)
        gradients = gradients.to(device=self.device, dtype=self.dtype)


        Bx_rf = b1_complex.real
        By_rf = b1_complex.imag

        # Calculate gradient-induced field offsets (N,)
        # spatial_coords: (N, 3), gradients: (N, 3) -> sum(spatial_coords * gradients, dim=1)
        Bz_gradient_offset = torch.sum(self.isochromats.spatial_coords * gradients, dim=1)

        # Effective Bz component for precession calculation (N,)
        # This is the total Z-field in the lab frame: B0 + Gz*z (assuming Gx*x + Gy*y are for off-resonance, not changing B direction)
        # The problem defines Bz(r,t) = B0 + Gx(t)*x + Gy(t)*y + Gz(t)*z. This is the z-component of B.
        Bz_total_lab = B0_TESLA + Bz_gradient_offset # This is the effective Bz in the lab frame

        # Effective Larmor frequency offset in rad/s (includes B0, gradients, and chemical shift) (N,)
        # This is ω_local = γ * B_local_z - ω_chemical_shift
        delta_omega_eff_rad_s = GAMMA_RAD_S_T_PROTON * Bz_total_lab - self.chemical_shift_rad_s

        # Construct the augmented Bloch matrix A_aug (N, 4, 4)
        A_aug = torch.zeros((self.num_isochromats, 4, 4), device=self.device, dtype=self.dtype)

        A_aug[:, 0, 0] = -1.0 / self.T2_star
        A_aug[:, 0, 1] = delta_omega_eff_rad_s
        A_aug[:, 0, 2] = GAMMA_RAD_S_T_PROTON * By_rf

        A_aug[:, 1, 0] = -delta_omega_eff_rad_s
        A_aug[:, 1, 1] = -1.0 / self.T2_star
        A_aug[:, 1, 2] = -GAMMA_RAD_S_T_PROTON * Bx_rf

        A_aug[:, 2, 0] = -GAMMA_RAD_S_T_PROTON * By_rf
        A_aug[:, 2, 1] = GAMMA_RAD_S_T_PROTON * Bx_rf
        A_aug[:, 2, 2] = -1.0 / self.T1
        A_aug[:, 2, 3] = self.M0_div_T1 # M0/T1

        # Last row is zeros, A_aug[:, 3, :] = 0, which is default

        # Compute matrix exponential: exp(A_aug * dt)
        exp_A_dt = torch.matrix_exp(A_aug * self.dt)

        # Prepare augmented magnetization M_aug = [Mx, My, Mz, 1]^T (N, 4, 1)
        M_current = self.isochromats.magnetization
        M_aug = torch.ones((self.num_isochromats, 4, 1), device=self.device, dtype=self.dtype)
        M_aug[:, 0:3, 0] = M_current

        # Update magnetization: M_new_aug = exp_A_dt @ M_aug
        M_new_aug = torch.matmul(exp_A_dt, M_aug)

        # Return updated magnetization [Mx, My, Mz] (N, 3)
        # And update the isochromats object directly
        new_magnetization = M_new_aug[:, 0:3, 0]
        self.isochromats.magnetization = new_magnetization # Update internal state
        return new_magnetization

if __name__ == '__main__':
    # Example Usage:
    # Create a single isochromat at origin, on-resonance, no gradients
    num_spins = 1
    initial_mag = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32) # Mz = 1
    M0_val = torch.tensor([1.0], dtype=torch.float32)
    T1_val = torch.tensor([1000e-3], dtype=torch.float32) # 1s
    T2_val = torch.tensor([100e-3], dtype=torch.float32)  # 100ms
    T2_star_val = torch.tensor([50e-3], dtype=torch.float32) # 50ms
    chem_shift_hz_val = torch.tensor([0.0], dtype=torch.float32)
    coords_val = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    isochromat_example = Isochromat(
        name="test_spin",
        magnetization=initial_mag.clone(),
        M0=M0_val.clone(), T1=T1_val.clone(), T2=T2_val.clone(), T2_star=T2_star_val.clone(),
        chemical_shift_hz=chem_shift_hz_val.clone(),
        spatial_coords=coords_val.clone(),
        device='cpu'
    )

    dt_example = 1e-5 # 0.01 ms time step, smaller for better pulse simulation
    solver = BlochSolver(isochromat_example, dt_example)

    # Test 1: T2* decay
    print("--- Testing T2* decay ---")
    isochromat_example.magnetization = torch.tensor([[0.5, 0.5, 0.7]], dtype=torch.float32)
    b1_no_rf = torch.tensor(0.0j, dtype=torch.complex64) # Scalar complex
    gradients_none = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) # Single gradient for all
    
    print(f"Initial M: {isochromat_example.magnetization.squeeze()}")
    mxy_initial = torch.norm(isochromat_example.magnetization.squeeze()[0:2])
    
    num_steps_t2_star_half = int((T2_star_val.item() / 2.0) / dt_example)
    for i in range(num_steps_t2_star_half):
        new_M = solver.run_step(b1_no_rf, gradients_none)
    
    print(f"M after T2*/2 ({num_steps_t2_star_half*dt_example*1e3:.2f} ms): {new_M.squeeze()}")
    mxy_final = torch.norm(new_M.squeeze()[0:2])
    expected_mxy_final = mxy_initial * torch.exp(- (T2_star_val / 2.0) / T2_star_val)
    print(f"Initial Mxy: {mxy_initial.item():.4f}")
    print(f"Expected Mxy after T2*/2: {expected_mxy_final.item():.4f}, Simulated Mxy: {mxy_final.item():.4f}")
    assert torch.isclose(mxy_final, expected_mxy_final, rtol=1e-2), "T2* decay Mxy mismatch"
    assert torch.isclose(new_M.squeeze()[2], torch.tensor(0.7), rtol=1e-2), "T2* decay Mz mismatch (should be relatively stable if T1 >> T2*)"


    # Test 2: T1 recovery
    print("\n--- Testing T1 recovery ---")
    isochromat_example.magnetization = torch.tensor([[0.01, 0.01, 0.0]], dtype=torch.float32) # Mz = 0, small Mxy
    # solver.isochromats.M0 = M0_val.clone() # M0 is total, M0_free is used now.
    # For this non-MT isochromat, M0_free is already M0_val.
    # Re-initialize solver to correctly pick up M0_free from the isochromat object if it were changed.
    # However, M0_val is used to define M0_free within the Isochromat object, so no need to re-assign M0 directly.
    solver = BlochSolver(isochromat_example, dt_example) # Re-init to ensure M0_free is correctly used if it was dynamic

    # solver.T1 = T1_val.clone() # T1 is set from isochromat_example
    # solver.M0_div_T1 = solver.isochromats.M0_free / solver.T1 # This is now done in __init__


    print(f"Initial M: {isochromat_example.magnetization.squeeze()}")
    num_steps_t1 = int(T1_val.item() / dt_example)
    for i in range(num_steps_t1):
        new_M = solver.run_step(b1_no_rf, gradients_none)
        if (i+1) % (num_steps_t1 // 4) == 0: # Print progress
             print(f"M after {(i+1)*dt_example*1e3:.1f} ms ({(i+1.0)/num_steps_t1*100:.0f}% of T1): {new_M.squeeze()}")


    print(f"M after 1xT1 ({num_steps_t1*dt_example*1e3:.2f} ms): {new_M.squeeze()}")
    mz_expected_t1 = M0_val * (1 - torch.exp(-T1_val/T1_val))
    print(f"Expected Mz after 1xT1: {mz_expected_t1.item():.4f}, Simulated Mz: {new_M.squeeze()[2].item():.4f}")
    assert torch.isclose(new_M.squeeze()[2], mz_expected_t1, rtol=1e-2), "T1 recovery Mz mismatch"
    mxy_after_t1 = torch.norm(new_M.squeeze()[0:2])
    assert mxy_after_t1 < 0.01, "Mxy should decay significantly during T1 recovery"


    # Test 3: 90-degree pulse (on-resonance)
    print("\n--- Testing 90-degree pulse (on-resonance) ---")
    isochromat_example.magnetization = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32) # Mz = 1
    isochromat_example.T1 = torch.tensor([1e6], dtype=torch.float32) # Long T1
    isochromat_example.T2_star = torch.tensor([1e6], dtype=torch.float32) # Long T2*
    isochromat_example.M0 = torch.tensor([1.0], dtype=torch.float32)
    solver = BlochSolver(isochromat_example, dt_example) # Re-init solver with new T1/T2*

    pulse_duration = 0.5e-3 # 0.5 ms
    flip_angle_rad = torch.pi / 2.0
    
    # For B1 along X_lab: b1_complex = (B1_amp, 0j)
    # theta = gamma * B1_amp * pulse_duration
    # B1_amp = theta / (gamma * pulse_duration)
    b1_amplitude_tesla = flip_angle_rad / (GAMMA_RAD_S_T_PROTON * pulse_duration)
    b1_pulse_x = torch.tensor(complex(b1_amplitude_tesla, 0.0), dtype=torch.complex64) # Along X lab

    print(f"Initial M: {isochromat_example.magnetization.squeeze()}")
    print(f"Applying {pulse_duration*1e3} ms pulse with B1x = {b1_amplitude_tesla:.2e} T for {flip_angle_rad*180/torch.pi:.1f} deg flip.")

    num_steps_pulse = int(pulse_duration / dt_example)
    if num_steps_pulse == 0: num_steps_pulse = 1 # Ensure at least one step

    for _ in range(num_steps_pulse):
        new_M = solver.run_step(b1_pulse_x, gradients_none)
    
    print(f"M after pulse: {new_M.squeeze()}")
    # Expected: Mz=0, My=-1 (for B1x pulse)
    assert torch.isclose(new_M.squeeze()[0], torch.tensor(0.0), atol=1e-3), "90deg (X) Mx mismatch"
    assert torch.isclose(new_M.squeeze()[1], torch.tensor(-1.0), atol=1e-2), "90deg (X) My mismatch"
    assert torch.isclose(new_M.squeeze()[2], torch.tensor(0.0), atol=1e-2), "90deg (X) Mz mismatch"
    print("90-degree pulse (X) test successful.")

    # Test 4: 90-degree pulse (on-resonance, B1 along Y_lab)
    print("\n--- Testing 90-degree pulse (on-resonance, B1 along Y_lab) ---")
    isochromat_example.magnetization = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32) # Reset Mz = 1
    # (Solver still has long T1/T2*)

    b1_pulse_y = torch.tensor(complex(0.0, b1_amplitude_tesla), dtype=torch.complex64) # Along Y lab
    print(f"Initial M: {isochromat_example.magnetization.squeeze()}")
    print(f"Applying {pulse_duration*1e3} ms pulse with B1y = {b1_amplitude_tesla:.2e} T for {flip_angle_rad*180/torch.pi:.1f} deg flip.")

    for _ in range(num_steps_pulse):
        new_M = solver.run_step(b1_pulse_y, gradients_none)

    print(f"M after pulse: {new_M.squeeze()}")
    # Expected: Mz=0, Mx=1 (for B1y pulse)
    assert torch.isclose(new_M.squeeze()[0], torch.tensor(1.0), atol=1e-2), "90deg (Y) Mx mismatch"
    assert torch.isclose(new_M.squeeze()[1], torch.tensor(0.0), atol=1e-3), "90deg (Y) My mismatch"
    assert torch.isclose(new_M.squeeze()[2], torch.tensor(0.0), atol=1e-2), "90deg (Y) Mz mismatch"
    print("90-degree pulse (Y) test successful.")

    # Test 5: Off-resonance precession
    print("\n--- Testing Off-resonance precession ---")
    isochromat_example.magnetization = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32) # M starts along X
    # solver still has long T1/T2*
    # Introduce chemical shift
    delta_f_hz = 100 # 100 Hz off-resonance
    isochromat_example.chemical_shift_hz = torch.tensor([float(delta_f_hz)], dtype=torch.float32)
    solver = BlochSolver(isochromat_example, dt_example) # Re-init to pick up new chemical shift

    precession_time_ms = 5.0 # ms
    num_steps_precession = int(precession_time_ms * 1e-3 / dt_example)
    
    print(f"Initial M: {isochromat_example.magnetization.squeeze()}")
    print(f"Simulating {precession_time_ms} ms of precession with {delta_f_hz} Hz offset.")

    for _ in range(num_steps_precession):
        new_M = solver.run_step(b1_no_rf, gradients_none)
    
    print(f"M after {precession_time_ms} ms: {new_M.squeeze()}")
    # Expected phase evolution: angle = 2 * pi * delta_f * time
    angle_rad = 2 * torch.pi * delta_f_hz * (precession_time_ms * 1e-3)
    # M will rotate in transverse plane. If starts at (1,0,0), new Mxy is (cos(angle), -sin(angle))
    # Note: delta_omega_eff = G*B0 - w_chem. Positive delta_f means w_chem is lower freq.
    # Or, w_Larmor = G*B0. delta_omega_eff = w_Larmor - w_chem.
    # Positive chemical_shift_hz means it's shifted to a higher frequency (e.g. fat vs water at higher field).
    # Bloch equation has dMx/dt = ... + delta_omega_eff * My
    # dMy/dt = ... - delta_omega_eff * Mx
    # If delta_omega_eff is positive, Mx -> My, My -> -Mx. This is counter-clockwise rotation for positive delta_omega_eff.
    # angle = -delta_omega_eff_rad_s * time
    # Here, delta_omega_eff_rad_s = GAMMA_RAD_S_T_PROTON * B0_TESLA - self.chemical_shift_rad_s
    # If chemical_shift_hz = 100Hz, chemical_shift_rad_s = 2*pi*100.
    # This means the isochromat's natural frequency is 100Hz *higher* than the reference (0 Hz).
    # So it precesses faster. In the frame rotating at 0Hz, it precesses at chemical_shift_rad_s.
    # The matrix term for dMy/dt has -delta_omega_eff_rad_s * Mx.
    # For Mx=1,My=0, dMy/dt is negative if delta_omega_eff_rad_s is positive. My becomes negative.
    # This corresponds to a clockwise rotation if delta_omega_eff_rad_s is positive in the matrix.
    # However, delta_omega_eff_rad_s itself is (something - chemical_shift_rad_s).
    # If on resonance with B0 (B0_TESLA term matches reference), then delta_omega_eff_rad_s = -chemical_shift_rad_s.
    # So, if chemical_shift_rad_s is positive (e.g. +100Hz * 2pi), then delta_omega_eff_rad_s is negative.
    # Then dMy/dt = -(-chemical_shift_rad_s) * Mx = chemical_shift_rad_s * Mx. (My becomes positive).
    # This is counter-clockwise rotation for a positive chemical shift value.
    # Angle = chemical_shift_rad_s * time
    expected_Mx = 1.0 * torch.cos(angle_rad)
    expected_My = 1.0 * torch.sin(angle_rad) # Positive chemical shift = faster precession = positive angle for My from Mx

    print(f"Expected angle (rad): {angle_rad.item():.4f}, Expected Mx: {expected_Mx.item():.4f}, Expected My: {expected_My.item():.4f}")
    assert torch.isclose(new_M.squeeze()[0], expected_Mx, atol=1e-2), "Off-resonance Mx mismatch"
    assert torch.isclose(new_M.squeeze()[1], expected_My, atol=1e-2), "Off-resonance My mismatch"
    print("Off-resonance precession test successful.")
