import torch

class Isochromat:
    def __init__(self,
                 name: str,
                 magnetization: torch.Tensor, # This will be for the free pool
                 M0: torch.Tensor, # Total M0 for the voxel/isochromat
                 T1: torch.Tensor, # T1 for free pool
                 T2: torch.Tensor, # T2 for free pool
                 chemical_shift_hz: torch.Tensor,
                 spatial_coords: torch.Tensor,
                 T2_star: torch.Tensor = None,
                 # MT parameters
                 fraction_semi_solid: torch.Tensor = None,
                 k_free_to_semi_solid: torch.Tensor = None, # kfw
                 k_semi_solid_to_free: torch.Tensor = None, # ksf
                 T1_semi_solid: torch.Tensor = None,    # T1s
                 T2_semi_solid_lineshape_us: torch.Tensor = None, # T2sl (for lineshape) in microseconds
                 semi_solid_resonance_offset_hz: torch.Tensor = None, # Offset of semi-solid pool from 0 Hz
                 magnetization_semi_solid: torch.Tensor = None, # Initial Mzs (N,1)
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.name = name
        self.device = device
        self.dtype = dtype
        
        if magnetization is None or M0 is None or T1 is None or T2 is None or chemical_shift_hz is None or spatial_coords is None:
            raise ValueError("Core parameters (magnetization, M0, T1, T2, chemical_shift_hz, spatial_coords) cannot be None.")

        num_isochromats = magnetization.shape[0]

        # Standard parameters (mostly for free pool)
        self.magnetization = magnetization.to(device=device, dtype=dtype) # Free pool
        self.M0 = M0.to(device=device, dtype=dtype) # Total M0
        self.T1 = T1.to(device=device, dtype=dtype) # T1f
        self.T2 = T2.to(device=device, dtype=dtype) # T2f
        
        if T2_star is None:
            self.T2_star = self.T2.clone() # T2*f
        else:
            self.T2_star = T2_star.to(device=device, dtype=dtype)
            
        self.chemical_shift_hz = chemical_shift_hz.to(device=device, dtype=dtype) # Free pool chemical shift
        self.spatial_coords = spatial_coords.to(device=device, dtype=dtype)

        # MT parameters
        self.is_mt_active = fraction_semi_solid is not None
        if self.is_mt_active:
            # Check if all MT params are provided
            if not all(p is not None for p in [fraction_semi_solid, k_free_to_semi_solid, k_semi_solid_to_free, 
                                               T1_semi_solid, T2_semi_solid_lineshape_us, semi_solid_resonance_offset_hz]):
                raise ValueError("If fraction_semi_solid is provided, all other core MT parameters (k_free_to_semi_solid, k_semi_solid_to_free, T1_semi_solid, T2_semi_solid_lineshape_us, semi_solid_resonance_offset_hz) must also be provided.")

            self.fraction_semi_solid = fraction_semi_solid.to(device=device, dtype=dtype)
            self.k_fw = k_free_to_semi_solid.to(device=device, dtype=dtype) 
            self.k_sf = k_semi_solid_to_free.to(device=device, dtype=dtype) 
            self.T1_s = T1_semi_solid.to(device=device, dtype=dtype)
            self.T2_sl_us = T2_semi_solid_lineshape_us.to(device=device, dtype=dtype)
            self.T2_sl_s = self.T2_sl_us * 1e-6 # Convert to seconds
            self.semi_solid_resonance_offset_hz = semi_solid_resonance_offset_hz.to(device=device, dtype=dtype)

            self.M0_free = self.M0 * (1 - self.fraction_semi_solid)
            self.M0_semi_solid = self.M0 * self.fraction_semi_solid

            if magnetization_semi_solid is None:
                self.magnetization_semi_solid = self.M0_semi_solid.clone().unsqueeze(-1) # Ensure (N,1)
            else:
                self.magnetization_semi_solid = magnetization_semi_solid.to(device=device, dtype=dtype)
                if self.magnetization_semi_solid.ndim == 1: # if (N,)
                     self.magnetization_semi_solid = self.magnetization_semi_solid.unsqueeze(-1) # convert to (N,1)
            
            # Validation for MT parameters
            mt_param_shapes = {
                "fraction_semi_solid": self.fraction_semi_solid.shape,
                "k_fw": self.k_fw.shape,
                "k_sf": self.k_sf.shape,
                "T1_s": self.T1_s.shape,
                "T2_sl_us": self.T2_sl_us.shape,
                "semi_solid_resonance_offset_hz": self.semi_solid_resonance_offset_hz.shape,
                "magnetization_semi_solid": self.magnetization_semi_solid.shape,
                "M0_free": self.M0_free.shape,
                "M0_semi_solid": self.M0_semi_solid.shape
            }
            expected_1d_shape = (num_isochromats,)
            expected_2d_shape_N1 = (num_isochromats, 1)
            
            error_messages = []
            if self.fraction_semi_solid.shape != expected_1d_shape: error_messages.append(f"fraction_semi_solid: expected {expected_1d_shape}, got {self.fraction_semi_solid.shape}")
            if self.k_fw.shape != expected_1d_shape: error_messages.append(f"k_fw: expected {expected_1d_shape}, got {self.k_fw.shape}")
            if self.k_sf.shape != expected_1d_shape: error_messages.append(f"k_sf: expected {expected_1d_shape}, got {self.k_sf.shape}")
            if self.T1_s.shape != expected_1d_shape: error_messages.append(f"T1_s: expected {expected_1d_shape}, got {self.T1_s.shape}")
            if self.T2_sl_us.shape != expected_1d_shape: error_messages.append(f"T2_sl_us: expected {expected_1d_shape}, got {self.T2_sl_us.shape}")
            if self.semi_solid_resonance_offset_hz.shape != expected_1d_shape: error_messages.append(f"semi_solid_resonance_offset_hz: expected {expected_1d_shape}, got {self.semi_solid_resonance_offset_hz.shape}")
            if self.magnetization_semi_solid.shape != expected_2d_shape_N1: error_messages.append(f"magnetization_semi_solid: expected {expected_2d_shape_N1}, got {self.magnetization_semi_solid.shape}")
            if self.M0_free.shape != expected_1d_shape: error_messages.append(f"M0_free: expected {expected_1d_shape}, got {self.M0_free.shape}")
            if self.M0_semi_solid.shape != expected_1d_shape: error_messages.append(f"M0_semi_solid: expected {expected_1d_shape}, got {self.M0_semi_solid.shape}")

            if error_messages:
                full_error_msg = "MT parameter tensor shape mismatch. num_isochromats={}:\n".format(num_isochromats) + "\n".join(error_messages)
                raise ValueError(full_error_msg)
        else: # Not MT active
            self.M0_free = self.M0.clone()
            self.fraction_semi_solid = None
            self.k_fw = None
            self.k_sf = None
            self.T1_s = None
            self.T2_sl_us = None
            self.T2_sl_s = None
            self.semi_solid_resonance_offset_hz = None
            self.M0_semi_solid = None
            self.magnetization_semi_solid = None


        # Validate shapes for standard parameters
        # (Assuming num_isochromats is correctly derived from magnetization.shape[0])
        core_param_shapes = {
            "M0": self.M0.shape, "T1": self.T1.shape, "T2": self.T2.shape, "T2_star": self.T2_star.shape,
            "chemical_shift_hz": self.chemical_shift_hz.shape,
            "spatial_coords": self.spatial_coords.shape,
            "magnetization (free pool)": self.magnetization.shape,
            "M0_free": self.M0_free.shape
        }
        expected_1d_shape = (num_isochromats,)
        expected_3d_shape_coords = (num_isochromats, 3)
        expected_3d_shape_mag = (num_isochromats, 3)

        error_messages_core = []
        if self.M0.shape != expected_1d_shape: error_messages_core.append(f"M0: expected {expected_1d_shape}, got {self.M0.shape}")
        if self.T1.shape != expected_1d_shape: error_messages_core.append(f"T1: expected {expected_1d_shape}, got {self.T1.shape}")
        if self.T2.shape != expected_1d_shape: error_messages_core.append(f"T2: expected {expected_1d_shape}, got {self.T2.shape}")
        if self.T2_star.shape != expected_1d_shape: error_messages_core.append(f"T2_star: expected {expected_1d_shape}, got {self.T2_star.shape}")
        if self.chemical_shift_hz.shape != expected_1d_shape: error_messages_core.append(f"chemical_shift_hz: expected {expected_1d_shape}, got {self.chemical_shift_hz.shape}")
        if self.spatial_coords.shape != expected_3d_shape_coords: error_messages_core.append(f"spatial_coords: expected {expected_3d_shape_coords}, got {self.spatial_coords.shape}")
        if self.magnetization.shape != expected_3d_shape_mag: error_messages_core.append(f"magnetization: expected {expected_3d_shape_mag}, got {self.magnetization.shape}")
        if self.M0_free.shape != expected_1d_shape: error_messages_core.append(f"M0_free: expected {expected_1d_shape}, got {self.M0_free.shape}")

        if error_messages_core:
            full_error_msg_core = "Standard parameter tensor shape mismatch. num_isochromats={}:\n".format(num_isochromats) + "\n".join(error_messages_core)
            raise ValueError(full_error_msg_core)


    def __repr__(self):
        base_repr = (f"Isochromat(name='{self.name}', "
                     f"num_isochromats={self.magnetization.shape[0]}, "
                     f"device='{self.device}', dtype={self.dtype}, "
                     f"MT_active={self.is_mt_active})")
        return base_repr

if __name__ == '__main__':
    # Example Usage (from BlochSolver, slightly adapted)
    num_spins = 2 # Reduced for brevity
    mag_initial = torch.zeros((num_spins, 3))
    mag_initial[:, 2] = 1.0

    isochromats_example = Isochromat(
        name="water_phantom",
        magnetization=mag_initial,
        M0=torch.ones(num_spins),
        T1=torch.full((num_spins,), 1000e-3),
        T2=torch.full((num_spins,), 100e-3),
        chemical_shift_hz=torch.zeros(num_spins),
        spatial_coords=torch.rand((num_spins, 3)) * 0.1,
        device='cpu'
    )
    print("--- Standard Isochromat ---")
    print(isochromats_example)
    print(f"  M0_total (same as M0_free here): {isochromats_example.M0}")
    print(f"  M0_free: {isochromats_example.M0_free}")


    # Example Usage with MT parameters
    print("\n--- Isochromat with MT parameters ---")
    num_mt_spins = 2 # Reduced for brevity
    mag_initial_mt_free = torch.zeros((num_mt_spins, 3), dtype=torch.float32)
    mag_initial_mt_free[:, 2] = 0.8 # Mzf, e.g. after some saturation

    total_M0_mt = torch.ones(num_mt_spins, dtype=torch.float32) * 1.0
    initial_Mzs = torch.ones(num_mt_spins, dtype=torch.float32) * 0.05 # Specific initial Mzs (N,), will be unsqueezed

    isochromats_mt_example = Isochromat(
        name="tissue_with_mt",
        magnetization=mag_initial_mt_free, # Free pool magnetization
        M0=total_M0_mt, # Total M0
        T1=torch.full((num_mt_spins,), 1.2, dtype=torch.float32),  # T1f (s)
        T2=torch.full((num_mt_spins,), 0.07, dtype=torch.float32),    # T2f (s)
        chemical_shift_hz=torch.zeros(num_mt_spins, dtype=torch.float32), # Water resonance at 0 Hz
        spatial_coords=torch.zeros((num_mt_spins, 3), dtype=torch.float32),
        T2_star=torch.full((num_mt_spins,), 0.05, dtype=torch.float32), # T2*f (s)
        # MT parameters
        fraction_semi_solid=torch.full((num_mt_spins,), 0.15, dtype=torch.float32), # fs = 15%
        k_free_to_semi_solid=torch.full((num_mt_spins,), 30.0, dtype=torch.float32),  # kfw (Hz)
        k_semi_solid_to_free=torch.full((num_mt_spins,), 5.0, dtype=torch.float32),   # ksf (Hz)
        T1_semi_solid=torch.full((num_mt_spins,), 1.0, dtype=torch.float32), # T1s (s)
        T2_semi_solid_lineshape_us=torch.full((num_mt_spins,), 12.0, dtype=torch.float32), # T2sl (µs) for lineshape
        semi_solid_resonance_offset_hz=torch.full((num_mt_spins,), -2000.0, dtype=torch.float32), # e.g., -2 kHz offset for semi-solid pool
        magnetization_semi_solid=initial_Mzs, # Initial Mzs, should be (N,1) or (N,)
        device='cpu',
        dtype=torch.float32
    )
    print(isochromats_mt_example)
    if isochromats_mt_example.is_mt_active:
        print(f"  M0_total: {isochromats_mt_example.M0.tolist()}")
        print(f"  Fraction semi-solid: {isochromats_mt_example.fraction_semi_solid.tolist()}")
        print(f"  M0_free: {isochromats_mt_example.M0_free.tolist()}")
        print(f"  M0_semi_solid: {isochromats_mt_example.M0_semi_solid.tolist()}")
        print(f"  Initial Mz_free: {isochromats_mt_example.magnetization[:, 2].tolist()}")
        print(f"  Initial Mz_semi_solid: {isochromats_mt_example.magnetization_semi_solid.squeeze().tolist()}")
        print(f"  k_fw (free to semi): {isochromats_mt_example.k_fw.tolist()} Hz")
        print(f"  k_sf (semi to free): {isochromats_mt_example.k_sf.tolist()} Hz")
        print(f"  T1_s: {isochromats_mt_example.T1_s.tolist()} s")
        print(f"  T2_sl (lineshape): {isochromats_mt_example.T2_sl_s.tolist()} s")
        print(f"  Semi-solid pool offset: {isochromats_mt_example.semi_solid_resonance_offset_hz.tolist()} Hz")
    
    # Test MT init with magnetization_semi_solid=None
    print("\n--- Isochromat with MT parameters (Mzs defaults to M0s) ---")
    isochromats_mt_default_Mzs = Isochromat(
        name="tissue_with_mt_default_Mzs",
        magnetization=mag_initial_mt_free, M0=total_M0_mt, T1=torch.full((num_mt_spins,), 1.2), T2=torch.full((num_mt_spins,), 0.07),
        chemical_shift_hz=torch.zeros(num_mt_spins), spatial_coords=torch.zeros((num_mt_spins, 3)),
        fraction_semi_solid=torch.full((num_mt_spins,), 0.10), k_free_to_semi_solid=torch.full((num_mt_spins,), 25.0),
        k_semi_solid_to_free=torch.full((num_mt_spins,), 4.0), T1_semi_solid=torch.full((num_mt_spins,), 0.9),
        T2_semi_solid_lineshape_us=torch.full((num_mt_spins,), 15.0), semi_solid_resonance_offset_hz=torch.full((num_mt_spins,), -2500.0),
        magnetization_semi_solid=None # Test default initialization
    )
    print(isochromats_mt_default_Mzs)
    if isochromats_mt_default_Mzs.is_mt_active:
        print(f"  M0_semi_solid: {isochromats_mt_default_Mzs.M0_semi_solid.tolist()}")
        print(f"  Initial Mz_semi_solid (should be M0s): {isochromats_mt_default_Mzs.magnetization_semi_solid.squeeze().tolist()}")
        assert torch.allclose(isochromats_mt_default_Mzs.M0_semi_solid, isochromats_mt_default_Mzs.magnetization_semi_solid.squeeze()), "Default Mzs initialization failed"

    print("\nAll Isochromat examples processed.")
