import torch
import numpy as np # For sinc function if needed, or use torch.sinc

class RF_Pulse:
    """
    Represents an RF pulse with defined waveform, amplitude, phase, and frequency offset.
    """
    def __init__(self,
                 name: str,
                 duration: float, # seconds
                 amplitude_tesla: float, # Peak amplitude in Tesla
                 phase_rad: float = 0.0, # Overall phase in radians
                 frequency_offset_hz: float = 0.0, # Offset from main Larmor freq, in Hz
                 waveform_shape: str | torch.Tensor = 'rect', # 'rect', 'sinc', or (N_samples,) tensor
                 num_waveform_samples: int = 100, # Used if waveform_shape is 'sinc' or other generated shapes
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            name (str): Name of the RF pulse.
            duration (float): Duration of the pulse in seconds.
            amplitude_tesla (float): Peak amplitude of the B1 field in Tesla.
                This scales the normalized waveform.
            phase_rad (float, optional): Overall phase of the RF pulse in radians.
                Defines the axis of B1 in the transverse plane (e.g., 0 for X', pi/2 for Y').
                Defaults to 0.0.
            frequency_offset_hz (float, optional): Frequency offset of the RF pulse
                relative to the scanner's base Larmor frequency (0 Hz). Defaults to 0.0.
            waveform_shape (str | torch.Tensor, optional): Shape of the pulse envelope.
                Can be 'rect', 'sinc', or a user-provided 1D PyTorch tensor
                representing the normalized amplitude envelope (values typically 0 to 1).
                Defaults to 'rect'.
            num_waveform_samples (int, optional): Number of samples for discretizing
                generated waveforms like 'sinc'. Defaults to 100.
            device (str, optional): PyTorch device. Defaults to 'cpu'.
            dtype (torch.dtype, optional): PyTorch dtype. Defaults to torch.float32.
        """
        self.name = name
        self.duration = torch.tensor(duration, device=device, dtype=dtype)
        self.amplitude_tesla = torch.tensor(amplitude_tesla, device=device, dtype=dtype)
        self.phase_rad = torch.tensor(phase_rad, device=device, dtype=dtype)
        self.frequency_offset_hz = torch.tensor(frequency_offset_hz, device=device, dtype=dtype)
        self.num_waveform_samples = num_waveform_samples # Store original for reference
        _num_samples_internal = num_waveform_samples # This might change if waveform_shape is a tensor
        self.device = device
        self.dtype = dtype

        if self.duration <= 0:
            raise ValueError("Pulse duration must be positive.")
        if self.amplitude_tesla < 0: 
            raise ValueError("Pulse amplitude cannot be negative.")

        normalized_amplitude_envelope: torch.Tensor
        if isinstance(waveform_shape, str):
            if waveform_shape == 'rect':
                normalized_amplitude_envelope = torch.ones(_num_samples_internal, device=device, dtype=dtype)
            elif waveform_shape == 'sinc':
                lobes = 3 
                t_sinc = torch.linspace(-lobes, lobes, _num_samples_internal, device=device, dtype=dtype)
                normalized_amplitude_envelope = torch.sinc(t_sinc)
            else:
                raise ValueError(f"Unsupported string waveform_shape: {waveform_shape}. Use 'rect', 'sinc', or provide a Tensor.")
        elif isinstance(waveform_shape, torch.Tensor):
            if waveform_shape.ndim != 1:
                raise ValueError("Provided waveform_shape tensor must be 1D.")
            
            current_num_samples_from_tensor = waveform_shape.numel()
            if _num_samples_internal != current_num_samples_from_tensor:
                # If num_waveform_samples was default (100) and tensor has different N, use tensor's N
                if self.num_waveform_samples == 100: 
                    _num_samples_internal = current_num_samples_from_tensor
                else: # User explicitly set num_waveform_samples, and it mismatches tensor
                    raise ValueError(
                        f"Provided waveform_shape tensor has {current_num_samples_from_tensor} samples, "
                        f"but num_waveform_samples was explicitly set to {self.num_waveform_samples}. "
                        f"They must match, or leave num_waveform_samples to default if providing a tensor."
                    )
            normalized_amplitude_envelope = waveform_shape.to(device=device, dtype=dtype)
        else:
            raise TypeError(f"Invalid waveform_shape type: {type(waveform_shape)}. Must be str or torch.Tensor.")

        # Update actual number of samples used for the waveform
        self.actual_num_waveform_samples = _num_samples_internal
        self.time_points = torch.linspace(0, self.duration.item(), self.actual_num_waveform_samples, device=device, dtype=dtype)

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        # Ensure phase_rad is complex for multiplication if it's a tensor itself (future use)
        phase_exp = torch.exp(1j * self.phase_rad.to(complex_dtype if self.phase_rad.is_floating_point() else torch.complex128))

        self.b1_waveform_complex_tesla = normalized_amplitude_envelope *                                          self.amplitude_tesla.to(complex_dtype) *                                          phase_exp
        
    def get_b1_and_offset(self, time_within_pulse: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the complex B1 field and frequency offset at a specific time within the pulse.
        """
        time_within_pulse_tensor = torch.tensor(time_within_pulse, device=self.device, dtype=self.dtype)

        # Check if time is outside pulse duration (with a small tolerance)
        if time_within_pulse_tensor < -1e-9 * self.duration or            time_within_pulse_tensor > self.duration * (1 + 1e-9):
            return (torch.tensor(0.0j, device=self.device, dtype=self.b1_waveform_complex_tesla.dtype), 
                    self.frequency_offset_hz)

        if self.actual_num_waveform_samples == 1:
            # If only one sample, return that sample if within duration (already checked), else 0 (covered by above)
            b1_val = self.b1_waveform_complex_tesla[0]
        else:
            # Normalized time (0 to 1), clamped to ensure it's within [0,1] for indexing
            norm_time = torch.clamp(time_within_pulse_tensor / self.duration, 0.0, 1.0)
            
            # Fractional index into the waveform samples
            idx_float = norm_time * (self.actual_num_waveform_samples - 1)
            
            idx_floor = torch.floor(idx_float).long()
            idx_ceil = torch.ceil(idx_float).long()

            # Clamp indices to be within the valid range [0, actual_num_waveform_samples - 1]
            idx_floor = torch.clamp(idx_floor, 0, self.actual_num_waveform_samples - 1)
            idx_ceil = torch.clamp(idx_ceil, 0, self.actual_num_waveform_samples - 1)
            
            if idx_floor == idx_ceil: # Exact hit or at the very boundaries after clamping
                b1_val = self.b1_waveform_complex_tesla[idx_floor]
            else:
                # Linear interpolation
                weight_ceil = idx_float - idx_floor # Proximity to ceil_idx
                weight_floor = 1.0 - weight_ceil   # Proximity to floor_idx (should be weight for floor)
                
                b1_val = (weight_floor * self.b1_waveform_complex_tesla[idx_floor] +
                          weight_ceil * self.b1_waveform_complex_tesla[idx_ceil])
        
        current_freq_offset = self.frequency_offset_hz
        # Ensure scalar output if inputs were scalar time
        return b1_val.squeeze(), current_freq_offset.squeeze()


    def __repr__(self):
        return (f"RF_Pulse(name='{self.name}', duration={self.duration.item():.2e}s, "
                f"amplitude={self.amplitude_tesla.item():.2e}T, phase={self.phase_rad.item():.2f}rad, "
                f"offset={self.frequency_offset_hz.item():.1f}Hz, "
                f"waveform_samples={self.actual_num_waveform_samples} (requested: {self.num_waveform_samples}))")

if __name__ == '__main__':
    print("--- RF_Pulse Examples ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    # 1. Rectangular pulse
    rect_pulse = RF_Pulse(name="rect_90", duration=1e-3, amplitude_tesla=1e-6, device=device, dtype=dtype)
    print(rect_pulse)
    b1, freq_off = rect_pulse.get_b1_and_offset(0.5e-3)
    print(f"Rect pulse at 0.5ms: B1={b1}, FreqOffset={freq_off} Hz")
    expected_b1_rect = torch.tensor(1e-6 + 0j, device=device, dtype=complex_dtype)
    assert torch.isclose(b1, expected_b1_rect), f"Rect pulse B1 mismatch. Got {b1}, expected {expected_b1_rect}"
    b1_out, _ = rect_pulse.get_b1_and_offset(1.1e-3) # outside
    assert torch.isclose(b1_out, torch.tensor(0.0j, device=device, dtype=complex_dtype)), "Rect pulse outside duration should be 0"


    # 2. Sinc pulse
    # Use odd number of samples for sinc to have a sample exactly at the center peak
    sinc_pulse = RF_Pulse(name="sinc_excite", duration=2e-3, amplitude_tesla=2e-6, 
                          waveform_shape='sinc', num_waveform_samples=201, phase_rad=torch.pi/2, device=device, dtype=dtype)
    print(sinc_pulse)
    b1_center, _ = sinc_pulse.get_b1_and_offset(1e-3) # Center of pulse (duration/2)
    b1_edge_start, _ = sinc_pulse.get_b1_and_offset(0.0)   # Start edge of pulse
    b1_edge_end, _ = sinc_pulse.get_b1_and_offset(2e-3)   # End edge of pulse
    
    expected_b1_center = 2e-6 * torch.exp(1j * torch.tensor(torch.pi/2, device=device, dtype=dtype)).to(complex_dtype)
    
    print(f"Sinc pulse at center (1ms): B1={b1_center} (Expected approx: {expected_b1_center})")
    print(f"Sinc pulse at start edge (0ms): B1={b1_edge_start}") # Should be close to zero for 3 lobes
    print(f"Sinc pulse at end edge (2ms): B1={b1_edge_end}")     # Should be close to zero for 3 lobes
    
    assert torch.isclose(b1_center, expected_b1_center, rtol=1e-3, atol=1e-9), f"Sinc center B1 mismatch. Got {b1_center}, expected {expected_b1_center}"
    # For sinc(t) with t from -3 to 3, sinc(-3)=0 and sinc(3)=0.
    assert torch.isclose(b1_edge_start, torch.tensor(0.0j, device=device, dtype=complex_dtype), atol=1e-7), f"Sinc start edge should be near zero. Got {b1_edge_start}"
    assert torch.isclose(b1_edge_end, torch.tensor(0.0j, device=device, dtype=complex_dtype), atol=1e-7), f"Sinc end edge should be near zero. Got {b1_edge_end}"


    # 3. Custom waveform pulse (Hann window)
    custom_samples_count = 50
    # Create a Hann window that peaks at 1.0
    custom_envelope = torch.hann_window(custom_samples_count, periodic=False, device=device, dtype=dtype) 
    
    custom_pulse = RF_Pulse(name="custom_hann", duration=0.8e-3, amplitude_tesla=1.5e-6,
                            waveform_shape=custom_envelope, # num_waveform_samples will be derived
                            frequency_offset_hz=100.0, device=device, dtype=dtype)
    print(custom_pulse)
    # Check if num_waveform_samples was updated internally
    assert custom_pulse.actual_num_waveform_samples == custom_samples_count 
    
    # Hann window is symmetric, peak is at duration/2
    b1_custom_mid, freq_off_custom = custom_pulse.get_b1_and_offset(0.4e-3) 
    # Hann window value at center is 1.0. Phase is 0.
    expected_b1_custom_mid = (1.5e-6 * torch.exp(1j * torch.tensor(0.0, device=device, dtype=dtype))).to(complex_dtype) 
    
    print(f"Custom Hann pulse at 0.4ms (center): B1={b1_custom_mid}, FreqOffset={freq_off_custom} Hz")
    assert torch.isclose(b1_custom_mid, expected_b1_custom_mid, rtol=1e-3, atol=1e-9), f"Custom pulse B1 mismatch. Got {b1_custom_mid}, expected {expected_b1_custom_mid}"
    assert torch.isclose(freq_off_custom, torch.tensor(100.0, device=device, dtype=dtype)), "Custom pulse freq offset mismatch"
    
    # Test point slightly off-center for interpolation
    b1_custom_off_center, _ = custom_pulse.get_b1_and_offset(0.3e-3)
    print(f"Custom Hann pulse at 0.3ms (off-center): B1={b1_custom_off_center}")
    # Should be less than peak, greater than edge.
    assert torch.abs(b1_custom_off_center) < torch.abs(expected_b1_custom_mid)
    assert torch.abs(b1_custom_off_center) > 0


    b1_outside, _ = custom_pulse.get_b1_and_offset(1.0e-3) # duration is 0.8ms
    print(f"Custom Hann pulse at 1.0ms (outside duration): B1={b1_outside}")
    assert torch.isclose(b1_outside, torch.tensor(0.0j, device=device, dtype=complex_dtype)), "B1 should be zero outside pulse duration"

    # Test with num_waveform_samples explicitly set for a tensor waveform (should match)
    try:
        error_pulse = RF_Pulse(name="error_pulse", duration=1e-3, amplitude_tesla=1e-6,
                               waveform_shape=torch.ones(10, device=device), 
                               num_waveform_samples=20) # Mismatch
        assert False, "Should have raised ValueError for mismatched num_waveform_samples and tensor shape"
    except ValueError as e:
        print(f"Caught expected error for mismatched samples: {e}")

    print("RF_Pulse examples finished.")
