import torch

class GradientPulse:
    """
    Represents a gradient pulse with defined waveform, amplitude, and duration.
    """
    def __init__(self,
                 name: str,
                 duration: float, # seconds
                 amplitude_tesla_p_m: torch.Tensor | list[float] | tuple[float, float, float], # Peak for [Gx, Gy, Gz]
                 waveform_shape: str | torch.Tensor = 'rect', # 'rect', 'trapezoid', or (N_samples,) tensor for common shape
                 num_waveform_samples: int = 100,
                 ramp_up_time: float = 0.0, # For 'trapezoid'
                 ramp_down_time: float = 0.0, # For 'trapezoid'
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            name (str): Name of the gradient pulse.
            duration (float): Total duration of the pulse in seconds.
            amplitude_tesla_p_m (torch.Tensor | list[float]): Peak amplitudes [Gx, Gy, Gz] in Tesla/meter.
                If list, converted to tensor. Tensor should be shape (3,).
            waveform_shape (str | torch.Tensor, optional): Shape of the pulse envelope.
                'rect', 'trapezoid', or a user-provided 1D PyTorch tensor for a common
                normalized amplitude envelope (0 to 1) applied to all axes. Defaults to 'rect'.
            num_waveform_samples (int, optional): Samples for discretizing generated waveforms. Defaults to 100.
            ramp_up_time (float, optional): Ramp-up time for 'trapezoid' waveform, in seconds. Defaults to 0.0.
            ramp_down_time (float, optional): Ramp-down time for 'trapezoid' waveform, in seconds. Defaults to 0.0.
            device (str, optional): PyTorch device. Defaults to 'cpu'.
            dtype (torch.dtype, optional): PyTorch dtype. Defaults to torch.float32.
        """
        self.name = name
        self.duration = torch.tensor(duration, device=device, dtype=dtype)
        
        if isinstance(amplitude_tesla_p_m, (list, tuple)):
            if len(amplitude_tesla_p_m) != 3:
                raise ValueError("amplitude_tesla_p_m as list/tuple must have 3 elements for Gx, Gy, Gz.")
            self.amplitude_tesla_p_m = torch.tensor(amplitude_tesla_p_m, device=device, dtype=dtype)
        elif isinstance(amplitude_tesla_p_m, torch.Tensor):
            if amplitude_tesla_p_m.shape != (3,):
                raise ValueError("amplitude_tesla_p_m as tensor must have shape (3,).")
            self.amplitude_tesla_p_m = amplitude_tesla_p_m.to(device=device, dtype=dtype)
        else:
            raise TypeError("amplitude_tesla_p_m must be a list, tuple, or torch.Tensor of 3 elements.")

        self.num_waveform_samples = num_waveform_samples # Store original
        _num_samples_internal = num_waveform_samples
        self.ramp_up_time = torch.tensor(ramp_up_time, device=device, dtype=dtype)
        self.ramp_down_time = torch.tensor(ramp_down_time, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        if self.duration <= 0:
            raise ValueError("Pulse duration must be positive.")
        if self.ramp_up_time < 0 or self.ramp_down_time < 0:
            raise ValueError("Ramp times cannot be negative.")
        if self.ramp_up_time + self.ramp_down_time > self.duration:
            raise ValueError("Sum of ramp_up_time and ramp_down_time cannot exceed pulse duration.")

        # Generate or set the normalized common amplitude waveform (1D)
        normalized_common_envelope: torch.Tensor
        if isinstance(waveform_shape, str):
            if waveform_shape == 'rect':
                normalized_common_envelope = torch.ones(_num_samples_internal, device=device, dtype=dtype)
            elif waveform_shape == 'trapezoid':
                if _num_samples_internal < 2: # Need at least 2 points for linspace/any shape
                     _num_samples_internal = 2 
                
                time_pts_trap = torch.linspace(0, self.duration.item(), _num_samples_internal, device=device, dtype=dtype)
                normalized_common_envelope = torch.zeros_like(time_pts_trap)
                
                flat_top_start_time = self.ramp_up_time.item()
                flat_top_end_time = self.duration.item() - self.ramp_down_time.item()

                if flat_top_start_time > 0: # Has ramp up
                    ramp_up_mask = (time_pts_trap > 0) & (time_pts_trap < flat_top_start_time)
                    # Avoid division by zero if ramp_up_time is very small but > 0 and no points fall in mask
                    if torch.any(ramp_up_mask) and flat_top_start_time > 1e-9: # check flat_top_start_time > 0 strictly
                         normalized_common_envelope[ramp_up_mask] = time_pts_trap[ramp_up_mask] / flat_top_start_time
                
                # Flat top part (includes start and end if ramp times are zero)
                flat_top_mask = (time_pts_trap >= flat_top_start_time) & (time_pts_trap <= flat_top_end_time)
                normalized_common_envelope[flat_top_mask] = 1.0
                
                if self.ramp_down_time.item() > 0: # Has ramp down
                    ramp_down_mask = (time_pts_trap > flat_top_end_time) & (time_pts_trap < self.duration.item())
                     # Avoid division by zero if ramp_down_time is very small but > 0 and no points fall in mask
                    if torch.any(ramp_down_mask) and self.ramp_down_time.item() > 1e-9:
                        normalized_common_envelope[ramp_down_mask] = (self.duration - time_pts_trap[ramp_down_mask]) / self.ramp_down_time
                
                # Ensure start and end points are correct (0 if ramp, 1 if no ramp at that end)
                if self.ramp_up_time > 0 : normalized_common_envelope[0] = 0.0
                else: normalized_common_envelope[0] = 1.0
                if self.ramp_down_time > 0 : normalized_common_envelope[-1] = 0.0
                else: normalized_common_envelope[-1] = 1.0


            else:
                raise ValueError(f"Unsupported string waveform_shape: {waveform_shape}. Use 'rect', 'trapezoid', or provide a Tensor.")
        elif isinstance(waveform_shape, torch.Tensor):
            if waveform_shape.ndim != 1:
                raise ValueError("Provided waveform_shape tensor must be 1D (common envelope).")
            
            current_num_samples_from_tensor = waveform_shape.numel()
            if _num_samples_internal != current_num_samples_from_tensor:
                if self.num_waveform_samples == 100: # Default value was unchanged
                    _num_samples_internal = current_num_samples_from_tensor
                else: # User explicitly set num_waveform_samples, and it mismatches tensor
                     raise ValueError(
                        f"Provided waveform_shape tensor has {current_num_samples_from_tensor} samples, "
                        f"but num_waveform_samples was explicitly set to {self.num_waveform_samples}. "
                        f"They must match, or leave num_waveform_samples to default if providing a tensor."
                    )
            normalized_common_envelope = waveform_shape.to(device=device, dtype=dtype)
        else:
            raise TypeError(f"Invalid waveform_shape type: {type(waveform_shape)}. Must be str or torch.Tensor.")

        self.actual_num_waveform_samples = _num_samples_internal
        self.time_points = torch.linspace(0, self.duration.item(), self.actual_num_waveform_samples, device=device, dtype=dtype)
        
        # Expand normalized_common_envelope (N_samples,) to (N_samples, 1)
        # Then multiply by amplitude_tesla_p_m (1, 3) using broadcasting -> (N_samples, 3)
        self.gradient_waveform_tesla_p_m = normalized_common_envelope.unsqueeze(1) * self.amplitude_tesla_p_m.unsqueeze(0)


    def get_gradients(self, time_within_pulse: float) -> torch.Tensor:
        """
        Get the gradient amplitudes [Gx, Gy, Gz] at a specific time within the pulse.

        Args:
            time_within_pulse (float): Time relative to the start of the pulse (0 to self.duration).

        Returns:
            torch.Tensor: Gradient amplitudes [Gx, Gy, Gz] in Tesla/meter. Shape (3,).
        """
        time_within_pulse_tensor = torch.tensor(time_within_pulse, device=self.device, dtype=self.dtype)

        if time_within_pulse_tensor < -1e-9 * self.duration or            time_within_pulse_tensor > self.duration * (1 + 1e-9):
            return torch.zeros(3, device=self.device, dtype=self.dtype)

        if self.actual_num_waveform_samples == 1:
            grad_val = self.gradient_waveform_tesla_p_m[0, :]
        else:
            norm_time = torch.clamp(time_within_pulse_tensor / self.duration, 0.0, 1.0)
            idx_float = norm_time * (self.actual_num_waveform_samples - 1)
            
            idx_floor = torch.floor(idx_float).long()
            idx_ceil = torch.ceil(idx_float).long()

            idx_floor = torch.clamp(idx_floor, 0, self.actual_num_waveform_samples - 1)
            idx_ceil = torch.clamp(idx_ceil, 0, self.actual_num_waveform_samples - 1)

            if idx_floor == idx_ceil:
                grad_val = self.gradient_waveform_tesla_p_m[idx_floor, :]
            else:
                weight_ceil = idx_float - idx_floor
                weight_floor = 1.0 - weight_ceil
                
                g_floor = self.gradient_waveform_tesla_p_m[idx_floor, :]
                g_ceil = self.gradient_waveform_tesla_p_m[idx_ceil, :]
                grad_val = weight_floor * g_floor + weight_ceil * g_ceil
        
        return grad_val.squeeze() # Ensure (3,)

    def __repr__(self):
        amp_str = ", ".join([f"{a.item():.2e}" for a in self.amplitude_tesla_p_m])
        return (f"GradientPulse(name='{self.name}', duration={self.duration.item():.2e}s, "
                f"amplitude_T_p_m=[{amp_str}], "
                f"waveform_samples={self.actual_num_waveform_samples} (requested: {self.num_waveform_samples}))")

if __name__ == '__main__':
    print("--- GradientPulse Examples ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # 1. Rectangular gradient
    rect_grad = GradientPulse(name="rect_Gz", duration=2e-3, 
                              amplitude_tesla_p_m=[0.0, 0.0, 0.01], device=device, dtype=dtype) # 10 mT/m Gz
    print(rect_grad)
    g_vals = rect_grad.get_gradients(1e-3)
    print(f"Rect Gz at 1ms: {g_vals.tolist()} T/m")
    assert torch.allclose(g_vals, torch.tensor([0.0, 0.0, 0.01], device=device, dtype=dtype))

    # 2. Trapezoidal gradient
    trap_grad = GradientPulse(name="trap_Gx", duration=3e-3,
                              amplitude_tesla_p_m=torch.tensor([0.02, 0.0, 0.0], device=device, dtype=dtype), # 20 mT/m Gx
                              waveform_shape='trapezoid', 
                              num_waveform_samples=300, # More samples for smoother trapezoid
                              ramp_up_time=0.5e-3,
                              ramp_down_time=0.5e-3,
                              device=device, dtype=dtype)
    print(trap_grad)
    g_ramp_up = trap_grad.get_gradients(0.25e-3) # Mid ramp-up
    g_flat_top = trap_grad.get_gradients(1.5e-3) # Mid flat-top
    g_ramp_down = trap_grad.get_gradients(2.75e-3) # Mid ramp-down
    g_start = trap_grad.get_gradients(0.0)
    g_end = trap_grad.get_gradients(3e-3)

    print(f"Trap Gx at 0.0ms (start): {g_start.tolist()} T/m")
    print(f"Trap Gx at 0.25ms (mid ramp-up): {g_ramp_up.tolist()} T/m") # Expected Gx = 0.02 * (0.25/0.5) = 0.01
    print(f"Trap Gx at 1.5ms (flat top): {g_flat_top.tolist()} T/m")   # Expected Gx = 0.02
    print(f"Trap Gx at 2.75ms (mid ramp-down): {g_ramp_down.tolist()} T/m") # Expected Gx = 0.02 * ( (3-2.75)/(0.5) ) = 0.01
    print(f"Trap Gx at 3.0ms (end): {g_end.tolist()} T/m")

    assert torch.allclose(g_start, torch.tensor([0.0,0,0], device=device, dtype=dtype), atol=1e-7)
    assert torch.isclose(g_ramp_up[0], torch.tensor(0.01, device=device, dtype=dtype), rtol=1e-2)
    assert torch.isclose(g_flat_top[0], torch.tensor(0.02, device=device, dtype=dtype), rtol=1e-2)
    assert torch.isclose(g_ramp_down[0], torch.tensor(0.01, device=device, dtype=dtype), rtol=1e-2)
    assert torch.allclose(g_end, torch.tensor([0.0,0,0], device=device, dtype=dtype), atol=1e-7)


    # 3. Custom waveform gradient
    custom_grad_samples = 60
    # Example: a sine envelope for Gx, zero for Gy, Gz
    t_custom = torch.linspace(0, torch.pi, custom_grad_samples, device=device, dtype=dtype) # Half a sine wave
    custom_envelope_g = torch.sin(t_custom) 
    
    custom_grad = GradientPulse(name="custom_sin_Gx", duration=1.5e-3,
                                amplitude_tesla_p_m=[0.025, 0.0, 0.0], # 25 mT/m peak for Gx
                                waveform_shape=custom_envelope_g,
                                device=device, dtype=dtype)
    print(custom_grad)
    assert custom_grad.actual_num_waveform_samples == custom_grad_samples

    g_custom_mid = custom_grad.get_gradients(0.75e-3) # Mid-point (peak of sine)
    expected_g_custom_mid = torch.tensor([0.025, 0.0, 0.0], device=device, dtype=dtype) # sin(pi/2)=1
    print(f"Custom Sin Gx at 0.75ms (peak): {g_custom_mid.tolist()} T/m")
    assert torch.allclose(g_custom_mid, expected_g_custom_mid, rtol=1e-2, atol=1e-7)
    
    g_custom_quarter = custom_grad.get_gradients(0.375e-3) # Quarter point (sin(pi/4))
    expected_g_custom_quarter = 0.025 * np.sin(np.pi/4)
    print(f"Custom Sin Gx at 0.375ms: {g_custom_quarter.tolist()} T/m, Expected Gx: {expected_g_custom_quarter:.4e}")
    assert torch.isclose(g_custom_quarter[0], torch.tensor(expected_g_custom_quarter, device=device, dtype=dtype), rtol=1e-2)

    print("GradientPulse examples finished.")
