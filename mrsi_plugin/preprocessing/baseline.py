import numpy as np

def subtract_polynomial_baseline(spectrum, polynomial_order, region_indices=None):
    """
    Estimates and subtracts a polynomial baseline from a single spectrum.

    The spectrum is typically real-valued and in the frequency domain.

    Args:
        spectrum (numpy.ndarray): The input 1D NumPy array (real-valued, frequency-domain).
        polynomial_order (int): The order of the polynomial to fit.
        region_indices (list or numpy.ndarray, optional):
            A list or NumPy array of integer indices that define regions of the
            spectrum considered to be baseline (i.e., without significant signal peaks).
            If None, all points in the spectrum are used for fitting the baseline,
            which might lead to signal distortion if prominent peaks are present.
            Defaults to None.

    Returns:
        tuple: A tuple containing:
            - spectrum_corrected (numpy.ndarray): The baseline-corrected spectrum (1D array).
            - estimated_baseline (numpy.ndarray): The estimated polynomial baseline (1D array).
    
    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If inputs have invalid values (e.g., negative polynomial_order,
                    out-of-bounds region_indices).
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    # It's generally expected to be real, but polyfit can handle complex if needed,
    # though baseline usually applied to real part. Add check if strictly real needed.
    # if np.iscomplexobj(spectrum):
    #     print("Warning: Input spectrum is complex. Polynomial fitting will use real part or magnitude implicitly depending on usage if not handled prior.")

    if not isinstance(polynomial_order, int):
        raise TypeError("Polynomial order must be an integer.")
    if polynomial_order < 0:
        raise ValueError("Polynomial order must be non-negative.")

    num_points = len(spectrum)
    x_axis = np.arange(num_points)

    if region_indices is not None:
        if not isinstance(region_indices, (list, np.ndarray)):
            raise TypeError("region_indices must be a list or NumPy array of integers.")
        region_indices = np.array(region_indices, dtype=int) # Ensure it's a NumPy array of int
        if region_indices.ndim != 1:
            raise ValueError("region_indices must be a 1D array of indices.")
        if np.any(region_indices < 0) or np.any(region_indices >= num_points):
            raise ValueError("region_indices contains values out of bounds for the spectrum length.")
        if len(region_indices) == 0:
            raise ValueError("region_indices cannot be empty if provided.")
        if len(region_indices) <= polynomial_order:
            raise ValueError(f"Number of points in region_indices ({len(region_indices)}) "
                             f"must be greater than polynomial_order ({polynomial_order}) for a stable fit.")
        
        x_fit = x_axis[region_indices]
        y_fit = spectrum[region_indices]
    else:
        if num_points <= polynomial_order:
             raise ValueError(f"Number of points in spectrum ({num_points}) "
                             f"must be greater than polynomial_order ({polynomial_order}) for a stable fit when region_indices is None.")
        x_fit = x_axis
        y_fit = spectrum

    try:
        # Fit polynomial
        coeffs = np.polyfit(x_fit, y_fit, polynomial_order)
        
        # Evaluate polynomial over the entire spectrum range
        estimated_baseline = np.polyval(coeffs, x_axis)
        
        spectrum_corrected = spectrum - estimated_baseline
        
        return spectrum_corrected, estimated_baseline

    except np.linalg.LinAlgError as e:
        # This can happen if the problem is ill-conditioned, e.g. too few points or collinear points
        raise ValueError(f"Polynomial fitting failed. This might be due to ill-conditioned data "
                         f"(e.g., too few points in region_indices, or all points in a line for low order poly). Original error: {e}")
    except Exception as e: # Catch other potential errors from polyfit/polyval
        raise RuntimeError(f"An unexpected error occurred during polynomial baseline subtraction: {e}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Test Data ---
    num_points = 1024
    x = np.arange(num_points)

    # Create a spectrum with a known baseline and some peaks
    true_baseline = 0.05 * (x - num_points/2)**2 / (num_points/2)**2 + 10 # Quadratic baseline
    peak1 = 20 * np.exp(-((x - num_points * 0.25)**2) / (2 * 30**2)) # Gaussian peak 1
    peak2 = 15 * np.exp(-((x - num_points * 0.6)**2) / (2 * 50**2)) # Gaussian peak 2
    noise = np.random.normal(0, 0.5, num_points)
    original_spectrum = true_baseline + peak1 + peak2 + noise

    poly_order = 2

    # --- Test Case 1: No region_indices (fit to whole spectrum) ---
    print("--- Test Case 1: Fitting to whole spectrum ---")
    try:
        corrected_spec1, baseline1 = subtract_polynomial_baseline(original_spectrum.copy(), poly_order)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x, original_spectrum, label="Original Spectrum")
        ax1.plot(x, baseline1, label=f"Estimated Baseline (Order {poly_order}, Whole Spectrum)", linestyle='--')
        ax1.plot(x, corrected_spec1, label="Corrected Spectrum", alpha=0.8)
        ax1.plot(x, true_baseline, label="True Baseline", linestyle=':', color='gray')
        ax1.set_title("Baseline Correction (Whole Spectrum Fit)")
        ax1.legend()
        ax1.grid(True)
        plt.savefig("baseline_test_whole_spectrum.png")
        print("Plot saved to baseline_test_whole_spectrum.png")
        # plt.show()
    except Exception as e:
        print(f"Error in Test Case 1: {e}")
    print("--------------------------------------------\\n")


    # --- Test Case 2: With region_indices ---
    print("--- Test Case 2: Fitting with region_indices ---")
    # Define regions that are mostly baseline (avoiding peaks)
    baseline_regions = np.concatenate([
        np.arange(0, int(num_points * 0.15)),
        np.arange(int(num_points * 0.4), int(num_points * 0.5)),
        np.arange(int(num_points * 0.75), num_points)
    ])
    
    try:
        corrected_spec2, baseline2 = subtract_polynomial_baseline(original_spectrum.copy(), poly_order, region_indices=baseline_regions)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x, original_spectrum, label="Original Spectrum")
        ax2.scatter(x[baseline_regions], original_spectrum[baseline_regions], color='red', s=10, label="Baseline Fit Points", zorder=5)
        ax2.plot(x, baseline2, label=f"Estimated Baseline (Order {poly_order}, Regions)", linestyle='--')
        ax2.plot(x, corrected_spec2, label="Corrected Spectrum", alpha=0.8)
        ax2.plot(x, true_baseline, label="True Baseline", linestyle=':', color='gray')
        ax2.set_title("Baseline Correction (Region-Based Fit)")
        ax2.legend()
        ax2.grid(True)
        plt.savefig("baseline_test_regions.png")
        print("Plot saved to baseline_test_regions.png")
        # plt.show()
    except Exception as e:
        print(f"Error in Test Case 2: {e}")
    print("--------------------------------------------\\n")

    # --- Test Case 3: Polynomial order too high for given points in regions ---
    print("--- Test Case 3: Polynomial order too high for region_indices ---")
    small_region = np.arange(0, 5) # Only 5 points
    high_poly_order = 5 # Order 5 polynomial needs at least 6 points
    try:
        subtract_polynomial_baseline(original_spectrum.copy(), high_poly_order, region_indices=small_region)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected error in Test Case 3: {e}")
    print("--------------------------------------------\\n")

    # --- Test Case 4: Empty region_indices ---
    print("--- Test Case 4: Empty region_indices ---")
    try:
        subtract_polynomial_baseline(original_spectrum.copy(), poly_order, region_indices=[])
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected error in Test Case 4: {e}")

    # Cleanup (optional)
    import os
    # if os.path.exists("baseline_test_whole_spectrum.png"): os.remove("baseline_test_whole_spectrum.png")
    # if os.path.exists("baseline_test_regions.png"): os.remove("baseline_test_regions.png")
    print("\\nTest plots generated (if matplotlib is available and configured).")
```
