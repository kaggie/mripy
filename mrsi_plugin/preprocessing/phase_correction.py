import numpy as np

def correct_phase_zero_order(spectrum, phase_rad):
    """
    Applies zero-order phase correction to a single complex frequency-domain spectrum.

    Args:
        spectrum (numpy.ndarray): The input 1D NumPy array, complex-valued,
                                  representing the frequency-domain signal.
        phase_rad (float): The zero-order phase shift to apply, in radians.

    Returns:
        numpy.ndarray: The phase-corrected spectrum (1D NumPy array, complex-valued).
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")
    if not np.iscomplexobj(spectrum):
        raise ValueError("Input spectrum must be complex-valued.")
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    if not isinstance(phase_rad, (int, float)):
        raise TypeError("Phase (phase_rad) must be a number (float or int).")

    corrected_spectrum = spectrum * np.exp(1j * phase_rad)
    return corrected_spectrum

def correct_phase_first_order(spectrum, pivot_point_index, phase_rad_per_point):
    """
    Applies first-order phase correction to a single complex frequency-domain spectrum.

    The phase correction is linear with frequency, pivoting around a specified point.

    Args:
        spectrum (numpy.ndarray): The input 1D NumPy array, complex-valued,
                                  representing the frequency-domain signal.
        pivot_point_index (int): The index of the pivot point in the spectrum
                                 around which the linear phase ramp is applied.
                                 This is often the index of a prominent peak or
                                 the center of the spectral region of interest.
        phase_rad_per_point (float): The linear phase change in radians per point
                                     (index difference) from the pivot point.
                                     A positive value means phase increases with
                                     increasing index relative to the pivot.

    Returns:
        numpy.ndarray: The phase-corrected spectrum (1D NumPy array, complex-valued).
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")
    if not np.iscomplexobj(spectrum):
        raise ValueError("Input spectrum must be complex-valued.")
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    if not isinstance(pivot_point_index, int):
        raise TypeError("Pivot point index must be an integer.")
    if not (0 <= pivot_point_index < len(spectrum)):
        raise ValueError(f"Pivot point index {pivot_point_index} is out of bounds for spectrum length {len(spectrum)}.")
    if not isinstance(phase_rad_per_point, (int, float)):
        raise TypeError("Phase per point (phase_rad_per_point) must be a number (float or int).")

    num_points = len(spectrum)
    indices = np.arange(num_points)
    
    # Calculate the first-order phase ramp
    # phase_ramp(i) = (i - pivot_idx) * phase_per_point
    phase_ramp = (indices - pivot_point_index) * phase_rad_per_point
    
    corrected_spectrum = spectrum * np.exp(1j * phase_ramp)
    return corrected_spectrum

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Test Data ---
    num_points = 1024
    # Create a dummy frequency-domain spectrum (e.g., a Lorentzian peak)
    # For simplicity, let's make a real peak and then apply phase errors to it.
    peak_center_index = num_points // 3 # Somewhat off-center
    peak_width_points = 20
    
    indices = np.arange(num_points)
    original_real_spectrum = 1.0 / (1 + ((indices - peak_center_index) / peak_width_points)**2)
    
    # Introduce phase errors
    zero_order_error_deg = 30  # degrees
    first_order_error_deg_per_100_points = 10 # degrees per 100 points
    
    zero_order_error_rad = np.deg2rad(zero_order_error_deg)
    # For first order, let pivot be the peak center.
    # phase_rad_per_point = (error_deg_per_N_points / N_points) * (pi/180)
    first_order_error_rad_per_point = np.deg2rad(first_order_error_deg_per_100_points / 100.0)
    
    # Apply errors
    phase_error_0 = zero_order_error_rad
    phase_error_1 = (indices - peak_center_index) * first_order_error_rad_per_point
    
    phased_spectrum = original_real_spectrum * np.exp(1j * (phase_error_0 + phase_error_1))

    print(f"Original (real) spectrum peak index: {peak_center_index}")
    print(f"Applied zero-order phase error: {zero_order_error_deg:.2f} degrees")
    print(f"Applied first-order phase error: {first_order_error_deg_per_100_points:.2f} deg / 100 points, pivoted at {peak_center_index}\\n")

    # --- Test Zero-Order Correction ---
    # Correct only the zero-order component
    corrected_spec_zo = correct_phase_zero_order(phased_spectrum.copy(), -zero_order_error_rad)
    
    print(f"Correcting zero-order phase with: {-zero_order_error_deg:.2f} degrees")

    # --- Test First-Order Correction ---
    # Correct only the first-order component from the original phased_spectrum
    corrected_spec_fo = correct_phase_first_order(phased_spectrum.copy(), 
                                                  pivot_point_index=peak_center_index, 
                                                  phase_rad_per_point=-first_order_error_rad_per_point)
    print(f"Correcting first-order phase with pivot {peak_center_index} and {-first_order_error_rad_per_point:.5f} rad/point")

    # --- Test Combined Correction (Sequential) ---
    # 1. Correct zero-order from phased_spectrum
    temp_corrected_spec = correct_phase_zero_order(phased_spectrum.copy(), -zero_order_error_rad)
    # 2. Correct first-order from the result
    corrected_spec_combined = correct_phase_first_order(temp_corrected_spec, 
                                                        pivot_point_index=peak_center_index, 
                                                        phase_rad_per_point=-first_order_error_rad_per_point)
    print("Combined correction applied sequentially.\\n")

    # --- Plotting (requires matplotlib) ---
    fig, axs = plt.subplots(4, 2, figsize=(12, 15)) # Real and Imaginary parts
    fig.suptitle("Phase Correction Effects", fontsize=16)

    def plot_spectrum(ax_real, ax_imag, spectrum_data, title_prefix):
        ax_real.set_title(f"{title_prefix} - Real Part")
        ax_real.plot(spectrum_data.real)
        ax_real.grid(True)
        
        ax_imag.set_title(f"{title_prefix} - Imaginary Part")
        ax_imag.plot(spectrum_data.imag)
        ax_imag.grid(True)

    plot_spectrum(axs[0,0], axs[0,1], phased_spectrum, "Phased (Error Introduced)")
    plot_spectrum(axs[1,0], axs[1,1], corrected_spec_zo, "Zero-Order Corrected")
    plot_spectrum(axs[2,0], axs[2,1], corrected_spec_fo, "First-Order Corrected (from original error)")
    plot_spectrum(axs[3,0], axs[3,1], corrected_spec_combined, "Combined (Sequential) Corrected")
    
    for ax_row in axs:
        for ax_col in ax_row:
            ax_col.set_xlabel("Frequency Point Index")
            ax_col.set_ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    try:
        plt.show()
        print("Plot displayed (if environment supports it).")
    except Exception as e:
        print(f"Could not display plot: {e}. Saving instead.")
        plt.savefig("phase_correction_effects.png")
        print("Plot saved to phase_correction_effects.png")

    # Basic checks
    assert corrected_spec_combined.shape == phased_spectrum.shape, "Shape mismatch after combined correction"
    # Check if the imaginary part of the combined corrected spectrum is close to zero
    # (since the original was purely real)
    mean_abs_imag_after_correction = np.mean(np.abs(corrected_spec_combined.imag))
    print(f"Mean absolute imaginary part after combined correction: {mean_abs_imag_after_correction:.4g}")
    assert mean_abs_imag_after_correction < 1e-3, "Combined correction did not make spectrum sufficiently real."
    print("\\nAll basic checks passed.")
    
    import os
    if os.path.exists("phase_correction_effects.png"):
        # os.remove("phase_correction_effects.png") # Keep for inspection
        pass
```
