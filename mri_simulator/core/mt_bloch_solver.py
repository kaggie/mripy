import torch
from .isochromat import Isochromat
from mri_simulator.utils.constants import GAMMA_RAD_S_T_PROTON, B0_TESLA

class MTBlochSolver:
    """
    Solves the Bloch equations for a two-pool (free water and semi-solid)
    system with Magnetization Transfer (MT) effects.
    Uses matrix exponentiation for a 5x5 augmented system.
    State vector: [Mxf, Myf, Mzf, Mzs, 1]^T
    """
    def __init__(self, isochromats: Isochromat, dt: float):
        """
        Initializes the MTBlochSolver.

        Args:
            isochromats (Isochromat): The isochromats to simulate. Must have
                                      MT parameters initialized (is_mt_active=True).
            dt (float): The time step for the simulation in seconds.
        """
        if not isochromats.is_mt_active:
            raise ValueError("MTBlochSolver requires Isochromat object with MT parameters activated.")

        self.isochromats = isochromats
        self.dt = dt
        self.device = isochromats.device
        self.dtype = isochromats.dtype
        self.num_isochromats = isochromats.magnetization.shape[0] # Free pool magnetization

        # Pre-calculate terms for free pool (f)
        self.T1f = torch.clamp(isochromats.T1.to(self.device, self.dtype), min=1e-6) # T1 of free pool
        self.T2f_star = torch.clamp(isochromats.T2_star.to(self.device, self.dtype), min=1e-6) # T2* of free pool
        self.M0f = isochromats.M0_free.to(self.device, self.dtype)
        self.M0f_div_T1f = self.M0f / self.T1f
        self.chemical_shift_rad_s_free = isochromats.chemical_shift_hz.to(self.device, self.dtype) * 2 * torch.pi

        # Pre-calculate terms for semi-solid pool (s)
        self.T1s = torch.clamp(isochromats.T1_s.to(self.device, self.dtype), min=1e-6)
        self.M0s = isochromats.M0_semi_solid.to(self.device, self.dtype)
        self.M0s_div_T1s = self.M0s / self.T1s
        self.k_fw = isochromats.k_fw.to(self.device, self.dtype) # Rate from free to semi-solid
        self.k_sf = isochromats.k_sf.to(self.device, self.dtype) # Rate from semi-solid to free
        
        # Linsehape T2 for semi-solid pool (for saturation calculation)
        self.T2_sl_s = isochromats.T2_sl_s.to(self.device, self.dtype) # Already in seconds
        self.semi_solid_resonance_offset_rad_s = isochromats.semi_solid_resonance_offset_hz.to(self.device, self.dtype) * 2 * torch.pi


    def _calculate_saturation_rate_W(self, b1_complex_tesla: torch.Tensor, rf_pulse_offset_rad_s: torch.Tensor) -> torch.Tensor:
        """
        Calculates the saturation rate (W) for the semi-solid pool using a Lorentzian lineshape.

        Args:
            b1_complex_tesla (torch.Tensor): RF pulse amplitude (complex Bx + iBy) in Tesla. (N,)
            rf_pulse_offset_rad_s (torch.Tensor): Offset of the RF pulse from the main Larmor frequency (0 Hz), in rad/s. (N,)

        Returns:
            torch.Tensor: Saturation rate W in Hz (s^-1). (N,)
        """
        b1_amp_tesla = torch.abs(b1_complex_tesla)
        
        # Offset of RF pulse from the semi-solid pool's resonance center
        delta_omega_rf_vs_semisolid_rad_s = rf_pulse_offset_rad_s - self.semi_solid_resonance_offset_rad_s
        
        # Lorentzian lineshape G(delta_omega_rf_vs_semisolid_rad_s)
        # G = (1/pi) * T2sl / (1 + (delta_omega_rf_vs_semisolid_rad_s * T2sl)^2)
        lorentzian_lineshape = (1.0 / torch.pi) * \
            (self.T2_sl_s / (1.0 + (delta_omega_rf_vs_semisolid_rad_s * self.T2_sl_s)**2))
            
        # W = pi * (gamma * B1_amp)^2 * G
        W_saturation_rate = torch.pi * (GAMMA_RAD_S_T_PROTON * b1_amp_tesla)**2 * lorentzian_lineshape
        return W_saturation_rate


    def run_step(self, b1_complex_tesla: torch.Tensor, gradients: torch.Tensor, rf_pulse_frequency_offset_hz: torch.Tensor) -> torch.Tensor:
        """
        Performs one time step of the MT Bloch equation simulation.

        Args:
            b1_complex_tesla (torch.Tensor): Complex RF pulse B1 (Bx + iBy) in Tesla.
                Shape: (N,) or (1,).
            gradients (torch.Tensor): Gradient vector [Gx, Gy, Gz] in Tesla/meter.
                Shape: (N, 3) or (1, 3).
            rf_pulse_frequency_offset_hz (torch.Tensor): Frequency offset of the RF pulse
                from the reference Larmor frequency (0 Hz), in Hz. Shape: (N,) or (1,).

        Returns:
            torch.Tensor: Updated magnetization of the free pool [Mxf, Myf, Mzf].
                Shape: (N, 3).
        """
        # Expand inputs if necessary and move to device/dtype
        if b1_complex_tesla.ndim == 0 or (b1_complex_tesla.ndim == 1 and b1_complex_tesla.shape[0] == 1):
            b1_complex_tesla = b1_complex_tesla.expand(self.num_isochromats)
        b1_complex_tesla = b1_complex_tesla.to(device=self.device, dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128)

        if gradients.ndim == 1 and gradients.shape[0] == 3: # Single gradient for all
             gradients = gradients.unsqueeze(0).expand(self.num_isochromats, 3)
        elif gradients.ndim == 2 and gradients.shape[0] == 1:
            gradients = gradients.expand(self.num_isochromats, 3)
        gradients = gradients.to(device=self.device, dtype=self.dtype)
        
        if rf_pulse_frequency_offset_hz.ndim == 0 or (rf_pulse_frequency_offset_hz.ndim == 1 and rf_pulse_frequency_offset_hz.shape[0] == 1):
            rf_pulse_frequency_offset_hz = rf_pulse_frequency_offset_hz.expand(self.num_isochromats)
        rf_pulse_frequency_offset_hz = rf_pulse_frequency_offset_hz.to(device=self.device, dtype=self.dtype)
        rf_pulse_offset_rad_s = rf_pulse_frequency_offset_hz * 2 * torch.pi

        # --- Calculate terms for the matrix ---
        Bx_rf = b1_complex_tesla.real
        By_rf = b1_complex_tesla.imag

        Bz_gradient_offset = torch.sum(self.isochromats.spatial_coords * gradients, dim=1)
        Bz_total_lab_eff_free = B0_TESLA + Bz_gradient_offset # Effective Bz for free pool in lab frame
        
        # Delta omega for free pool (relative to 0 Hz reference)
        delta_omega_eff_rad_s_free = GAMMA_RAD_S_T_PROTON * Bz_total_lab_eff_free - self.chemical_shift_rad_s_free

        # Saturation rate W for semi-solid pool
        W_sat = self._calculate_saturation_rate_W(b1_complex_tesla, rf_pulse_offset_rad_s)

        # --- Construct the 5x5 augmented MT Bloch matrix A_mt_aug (N, 5, 5) ---
        A_mt = torch.zeros((self.num_isochromats, 5, 5), device=self.device, dtype=self.dtype)

        # Row 0: dMxf/dt
        A_mt[:, 0, 0] = -1.0 / self.T2f_star
        A_mt[:, 0, 1] = delta_omega_eff_rad_s_free
        A_mt[:, 0, 2] = GAMMA_RAD_S_T_PROTON * By_rf
        # A_mt[:, 0, 3] = 0 (no direct coupling Mzs -> Mxf)
        # A_mt[:, 0, 4] = 0 (no M0f/T1f term for Mxf)

        # Row 1: dMyf/dt
        A_mt[:, 1, 0] = -delta_omega_eff_rad_s_free
        A_mt[:, 1, 1] = -1.0 / self.T2f_star
        A_mt[:, 1, 2] = -GAMMA_RAD_S_T_PROTON * Bx_rf
        # A_mt[:, 1, 3] = 0 (no direct coupling Mzs -> Myf)
        # A_mt[:, 1, 4] = 0 (no M0f/T1f term for Myf)

        # Row 2: dMzf/dt
        A_mt[:, 2, 0] = -GAMMA_RAD_S_T_PROTON * By_rf
        A_mt[:, 2, 1] = GAMMA_RAD_S_T_PROTON * Bx_rf
        A_mt[:, 2, 2] = -(1.0 / self.T1f + self.k_fw)
        A_mt[:, 2, 3] = self.k_sf
        A_mt[:, 2, 4] = self.M0f_div_T1f

        # Row 3: dMzs/dt
        # A_mt[:, 3, 0] = 0 (no direct coupling Mxf -> Mzs)
        # A_mt[:, 3, 1] = 0 (no direct coupling Myf -> Mzs)
        A_mt[:, 3, 2] = self.k_fw
        A_mt[:, 3, 3] = -(1.0 / self.T1s + self.k_sf + W_sat)
        A_mt[:, 3, 4] = self.M0s_div_T1s
        
        # Row 4: d(1)/dt = 0 (for augmentation)
        # A_mt[:, 4, :] = 0 (already zeros)

        exp_A_mt_dt = torch.matrix_exp(A_mt * self.dt)

        # Prepare augmented magnetization M_aug = [Mxf, Myf, Mzf, Mzs, 1]^T (N, 5, 1)
        M_aug = torch.ones((self.num_isochromats, 5, 1), device=self.device, dtype=self.dtype)
        M_aug[:, 0:3, 0] = self.isochromats.magnetization # Current Mf
        M_aug[:, 3, 0] = self.isochromats.magnetization_semi_solid.squeeze(-1) # Current Mzs

        # Update magnetization: M_new_aug = exp_A_mt_dt @ M_aug
        M_new_aug = torch.matmul(exp_A_mt_dt, M_aug)

        # Update isochromat object's magnetization state
        self.isochromats.magnetization = M_new_aug[:, 0:3, 0]
        self.isochromats.magnetization_semi_solid = M_new_aug[:, 3, 0].unsqueeze(-1)

        return self.isochromats.magnetization # Return updated free pool magnetization

if __name__ == '__main__':
    # --- Example Usage for MTBlochSolver ---
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    print(f"Using device: {dev}")

    num_mt_spins = 2
    # Initial state: Mf = [0,0,M0f], Ms = M0s (equilibrium)
    # M0_total = 1.0, fs = 0.1 -> M0f = 0.9, M0s = 0.1
    
    fs_example = 0.1
    m0_total_example = 1.0
    m0f_init = m0_total_example * (1-fs_example)
    
    mag_initial_free = torch.zeros((num_mt_spins, 3), device=dev)
    mag_initial_free[:, 2] = m0f_init 

    # Isochromat for MT
    isochromats_mt = Isochromat(
        name="mt_tissue_example",
        magnetization=mag_initial_free,
        M0=torch.full((num_mt_spins,), m0_total_example, device=dev),
        T1=torch.full((num_mt_spins,), 1.0, device=dev),        # T1f = 1s
        T2=torch.full((num_mt_spins,), 0.08, device=dev),       # T2f = 80ms
        T2_star=torch.full((num_mt_spins,), 0.05, device=dev),  # T2*f = 50ms
        chemical_shift_hz=torch.full((num_mt_spins,), 0.0, device=dev), # Free pool on resonance
        spatial_coords=torch.zeros((num_mt_spins, 3), device=dev),
        # MT parameters
        fraction_semi_solid=torch.full((num_mt_spins,), fs_example, device=dev),
        k_free_to_semi_solid=torch.full((num_mt_spins,), 20.0, device=dev),    # kfw = 20 Hz
        k_semi_solid_to_free=torch.full((num_mt_spins,), 20.0 * fs_example / (1-fs_example) , device=dev), # ksf based on kfw*M0f=ksf*M0s
        T1_semi_solid=torch.full((num_mt_spins,), 1.0, device=dev),          # T1s = 1s
        T2_semi_solid_lineshape_us=torch.full((num_mt_spins,), 12.0, device=dev), # T2sl = 12 us
        semi_solid_resonance_offset_hz=torch.full((num_mt_spins,), -2500.0, device=dev), # Macromolecules at -2.5 kHz
        magnetization_semi_solid=None, # Defaults to M0s
        device=dev
    )
    print(f"Initial Mf: {isochromats_mt.magnetization}")
    print(f"Initial Ms: {isochromats_mt.magnetization_semi_solid}")
    print(f"M0f: {isochromats_mt.M0_free}, M0s: {isochromats_mt.M0_semi_solid}")
    assert torch.allclose(isochromats_mt.magnetization_semi_solid.squeeze(), isochromats_mt.M0_semi_solid)


    dt_mt = 100e-6 # 100 us time step
    mt_solver = MTBlochSolver(isochromats_mt, dt_mt)

    # Scenario 1: Continuous wave (CW) saturation pulse off-resonance for water, on semi-solid resonance
    # RF pulse params
    b1_cw_amplitude_uT = 5.0  # microTesla
    b1_cw_amplitude_T = b1_cw_amplitude_uT * 1e-6 # Tesla
    
    # RF pulse is a complex value. Assume applied along X' in rotating frame of pulse.
    # If pulse is at offset freq, B1_complex is B1_amp * exp(i*phase_at_time_t)
    # For CW, phase can be 0. B1_complex = B1_amp + 0j
    b1_val_T = torch.tensor([complex(b1_cw_amplitude_T, 0.0)], device=dev, dtype=torch.complex64 if isochromats_mt.dtype==torch.float32 else torch.complex128)
    
    # RF pulse frequency offset set to the semi-solid pool's resonance
    rf_offset_hz = isochromats_mt.semi_solid_resonance_offset_hz 
    
    gradients_none = torch.tensor([0.0, 0.0, 0.0], device=dev)
    
    saturation_duration_ms = 50.0 # ms
    num_sat_steps = int(saturation_duration_ms * 1e-3 / dt_mt)

    print(f"\n--- Simulating CW MT saturation for {saturation_duration_ms} ms ---")
    print(f"RF B1: {b1_cw_amplitude_uT} uT, RF offset: {rf_offset_hz[0].item()} Hz (targeting semi-solid pool)")

    # Calculate expected W for verification (for the first isochromat)
    W_expected = mt_solver._calculate_saturation_rate_W(b1_val_T.expand(num_mt_spins), rf_offset_hz*2*torch.pi)
    print(f"Calculated W (saturation rate): {W_expected[0].item():.2f} s^-1")

    for i in range(num_sat_steps):
        Mf_new = mt_solver.run_step(b1_val_T, gradients_none, rf_offset_hz)
        if (i+1) % (num_sat_steps // 5) == 0 or i == num_sat_steps -1:
            print(f"Step {i+1}/{num_sat_steps}: Mf={Mf_new[0].tolist()}, Ms={isochromats_mt.magnetization_semi_solid[0].item():.4f}")

    print("--- Saturation complete ---")
    print(f"Final Mf[0]: {isochromats_mt.magnetization[0].tolist()}")
    print(f"Final Ms[0]: {isochromats_mt.magnetization_semi_solid[0].squeeze().item():.4f}")
    
    # Check if Mzs is significantly reduced
    assert isochromats_mt.magnetization_semi_solid[0].item() < isochromats_mt.M0_semi_solid[0].item() * 0.5, "Mzs not significantly saturated"
    # Check if Mzf is affected (reduced due to exchange from saturated Mzs)
    assert isochromats_mt.magnetization[0,2].item() < isochromats_mt.M0_free[0].item() * 0.95, "Mzf not significantly affected by MT"

    print("\nMTBlochSolver example finished.")
