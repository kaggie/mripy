import numpy as np

def apodize_gaussian(spectrum, sigma):
    """
    Applies Gaussian apodization to a single time-domain spectrum (FID).

    The Gaussian window is centered at the first point of the spectrum (time zero).

    Args:
        spectrum (numpy.ndarray): The input 1D NumPy array representing the
                                  time-domain signal (FID).
        sigma (float): Standard deviation for the Gaussian window. This parameter
                       defines the width of the Gaussian. A larger sigma results
                       in a wider window and less aggressive apodization.
                       It can be related to desired line broadening.

    Returns:
        numpy.ndarray: The apodized spectrum (1D NumPy array).
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("Sigma must be a positive number.")

    num_points = len(spectrum)
    time_points = np.arange(num_points)
    
    # Gaussian window centered at t=0 (first point of the FID)
    # w(t) = exp(-t^2 / (2 * sigma^2))
    # Here, 'time_points' serves as 't' scaled by some arbitrary unit,
    # 'sigma' is in the same units.
    gaussian_window = np.exp(-(time_points**2) / (2 * sigma**2))
    
    apodized_spectrum = spectrum * gaussian_window
    return apodized_spectrum

def apodize_lorentzian(spectrum, line_broadening_hz, sampling_frequency_hz):
    """
    Applies Lorentzian (exponential decay) apodization to a single
    time-domain spectrum (FID).

    Args:
        spectrum (numpy.ndarray): The input 1D NumPy array representing the
                                  time-domain signal (FID).
        line_broadening_hz (float): The desired amount of line broadening, in Hz.
                                    This determines the rate of exponential decay.
        sampling_frequency_hz (float): The sampling frequency of the MRSI data, in Hz.

    Returns:
        numpy.ndarray: The apodized spectrum (1D NumPy array).
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    if not isinstance(line_broadening_hz, (int, float)) or line_broadening_hz < 0:
        raise ValueError("Line broadening must be a non-negative number.")
    if not isinstance(sampling_frequency_hz, (int, float)) or sampling_frequency_hz <= 0:
        raise ValueError("Sampling frequency must be a positive number.")

    num_points = len(spectrum)
    time_vector = np.arange(num_points) / sampling_frequency_hz
    
    # Lorentzian window (exponential decay)
    # w(t) = exp(-pi * LB * t)
    lorentzian_window = np.exp(-np.pi * line_broadening_hz * time_vector)
    
    apodized_spectrum = spectrum * lorentzian_window
    return apodized_spectrum

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Test Data ---
    sampling_freq = 2000.0  # Hz
    n_points = 1024
    time = np.arange(n_points) / sampling_freq
    
    # Create a dummy FID (e.g., a decaying sinusoid)
    frequency = 50  # Hz
    decay_rate = 10 # Hz (1/T2*)
    fid = np.exp(-np.pi * decay_rate * time) * np.cos(2 * np.pi * frequency * time)
    fid_noise = fid + (np.random.randn(n_points) * 0.1) + (np.random.randn(n_points) * 0.1j) # Add some noise
    fid_noise_freq = np.fft.fftshift(np.fft.fft(fid_noise))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_points, d=1/sampling_freq))

    # --- Test Gaussian Apodization ---
    sigma_gaussian = 20.0  # Example sigma value (adjust based on desired effect)
    # A common way to choose sigma for a desired line broadening (LB_gauss_hz) is:
    # sigma_time_domain = N / (np.pi * LB_gauss_hz * np.sqrt(2)) 
    # Or, if sigma is given in terms of points:
    # sigma_points = desired_line_broadening_points / (2 * np.sqrt(2 * np.log(2))) # FWHM to sigma
    
    apodized_fid_gaussian = apodize_gaussian(fid_noise.copy(), sigma=sigma_gaussian)
    apodized_spec_gaussian = np.fft.fftshift(np.fft.fft(apodized_fid_gaussian))

    print(f"Gaussian Apodization: sigma = {sigma_gaussian}")
    print(f"Original FID norm: {np.linalg.norm(fid_noise):.2f}")
    print(f"Gaussian Apodized FID norm: {np.linalg.norm(apodized_fid_gaussian):.2f}\\n")


    # --- Test Lorentzian Apodization ---
    lb_lorentzian = 5.0  # Hz
    apodized_fid_lorentzian = apodize_lorentzian(fid_noise.copy(), 
                                                 line_broadening_hz=lb_lorentzian, 
                                                 sampling_frequency_hz=sampling_freq)
    apodized_spec_lorentzian = np.fft.fftshift(np.fft.fft(apodized_fid_lorentzian))

    print(f"Lorentzian Apodization: line_broadening_hz = {lb_lorentzian} Hz")
    print(f"Original FID norm: {np.linalg.norm(fid_noise):.2f}")
    print(f"Lorentzian Apodized FID norm: {np.linalg.norm(apodized_fid_lorentzian):.2f}\\n")

    # --- Plotting (requires matplotlib) ---
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Apodization Effects", fontsize=16)

    # Time domain plots
    axs[0, 0].set_title("Original FID (Real Part)")
    axs[0, 0].plot(time, fid_noise.real)
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")

    axs[1, 0].set_title(f"Gaussian Apodized FID (sigma={sigma_gaussian})")
    axs[1, 0].plot(time, apodized_fid_gaussian.real)
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Amplitude")

    axs[2, 0].set_title(f"Lorentzian Apodized FID (LB={lb_lorentzian} Hz)")
    axs[2, 0].plot(time, apodized_fid_lorentzian.real)
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Amplitude")

    # Frequency domain plots (magnitude)
    axs[0, 1].set_title("Original Spectrum (Magnitude)")
    axs[0, 1].plot(freq_axis, np.abs(fid_noise_freq))
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Magnitude")

    axs[1, 1].set_title("Gaussian Apodized Spectrum")
    axs[1, 1].plot(freq_axis, np.abs(apodized_spec_gaussian))
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")

    axs[2, 1].set_title("Lorentzian Apodized Spectrum")
    axs[2, 1].plot(freq_axis, np.abs(apodized_spec_lorentzian))
    axs[2, 1].set_xlabel("Frequency (Hz)")
    axs[2, 1].set_ylabel("Magnitude")

    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # To save the plot instead of showing it (useful in non-interactive environments)
    # plt.savefig("apodization_effects.png")
    # print("Plot saved to apodization_effects.png")
    
    # If in an interactive environment, you might use plt.show()
    # For this tool, saving or just printing confirmation is better.
    try:
        plt.show() # This might not work in all sandboxed environments
        print("Plot displayed (if environment supports it).")
    except Exception as e:
        print(f"Could not display plot: {e}. Saving instead.")
        plt.savefig("apodization_effects.png")
        print("Plot saved to apodization_effects.png")

    # Basic checks for function outputs
    assert apodized_fid_gaussian.shape == fid_noise.shape, "Gaussian apodization shape mismatch"
    assert apodized_fid_lorentzian.shape == fid_noise.shape, "Lorentzian apodization shape mismatch"
    assert not np.array_equal(apodized_fid_gaussian, fid_noise), "Gaussian apodization did not change FID"
    assert not np.array_equal(apodized_fid_lorentzian, fid_noise), "Lorentzian apodization did not change FID"
    print("\\nAll basic checks passed.")

    # Clean up the saved plot if it was created
    import os
    if os.path.exists("apodization_effects.png"):
        # os.remove("apodization_effects.png") # Keep it for inspection if needed
        pass

```
