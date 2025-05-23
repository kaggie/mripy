import numpy as np

def signal_to_concentration(
    dce_series_data: np.ndarray,
    t10_map_data: np.ndarray,
    r1_relaxivity: float,
    TR: float,
    baseline_time_points: int = 5,
):
    """
    Converts DCE-MRI signal intensity time-series to contrast agent concentration.

    This function uses the simplified approach assuming a fixed TR and a known
    pre-contrast T1 map (T10_map_data). The conversion is based on the change
    in longitudinal relaxation rate (R1 = 1/T1) due to the contrast agent.

    The formula used for R1(t) at each time point t is derived from the
    steady-state signal equation for a spoiled gradient echo sequence:
        S(t) = S0 * sin(alpha) * (1 - exp(-TR * R1(t))) / (1 - cos(alpha) * exp(-TR * R1(t)))
    Assuming S0 and alpha are constant and that the signal is proportional to
    (1 - exp(-TR * R1(t))), we can write:
        S(t) / S_pre = (1 - exp(-TR * R1(t))) / (1 - exp(-TR * R1_0))
    Rearranging for R1(t):
        R1(t) = (-1/TR) * ln(1 - (S(t)/S_pre) * (1 - exp(-TR * R1_0)))
    Then, concentration C(t) is calculated as:
        C(t) = (R1(t) - R1_0) / r1_relaxivity

    Args:
        dce_series_data (np.ndarray): 4D array (x, y, z, time) of DCE signal
                                      intensities.
        t10_map_data (np.ndarray): 3D array (x, y, z) of pre-contrast T1
                                   values (in seconds).
        r1_relaxivity (float): Longitudinal relaxivity of the contrast agent
                               (e.g., in s⁻¹ mM⁻¹).
        TR (float): Repetition Time (in seconds).
        baseline_time_points (int, optional): Number of initial time points
                                             to average for S_pre (baseline
                                             signal). Defaults to 5.

    Returns:
        np.ndarray: 4D array (x, y, z, time) of concentration-time curves.
                    The units of concentration depend on the units of
                    r1_relaxivity.

    Raises:
        ValueError: If inputs are invalid (e.g., incorrect dimensions,
                    non-positive parameters, insufficient time points).
    """
    # Validate inputs
    if not isinstance(dce_series_data, np.ndarray) or dce_series_data.ndim != 4:
        raise ValueError("dce_series_data must be a 4D NumPy array.")
    if not isinstance(t10_map_data, np.ndarray) or t10_map_data.ndim != 3:
        raise ValueError("t10_map_data must be a 3D NumPy array.")
    if dce_series_data.shape[:3] != t10_map_data.shape:
        raise ValueError(
            "Spatial dimensions of dce_series_data (x,y,z) must match "
            "t10_map_data."
        )
    if TR <= 0:
        raise ValueError("TR must be positive.")
    if r1_relaxivity <= 0:
        raise ValueError("r1_relaxivity must be positive.")
    if baseline_time_points <= 0:
        raise ValueError("baseline_time_points must be positive.")
    if baseline_time_points >= dce_series_data.shape[3]:
        raise ValueError(
            "baseline_time_points must be less than the number of time points "
            "in dce_series_data."
        )

    S_pre = np.mean(dce_series_data[..., :baseline_time_points], axis=3)
    R1_0 = 1.0 / (t10_map_data + 1e-9)  # Ensure t10_map_data is not zero

    num_time_points = dce_series_data.shape[3]
    Ct_data = np.zeros_like(dce_series_data, dtype=float)

    # Vectorized calculation for efficiency
    S_pre_expanded = np.expand_dims(S_pre, axis=3) # Shape (x,y,z,1)
    R1_0_expanded = np.expand_dims(R1_0, axis=3)   # Shape (x,y,z,1)

    # Add epsilon to S_pre_expanded to avoid division by zero.
    S_pre_safe = np.where(S_pre_expanded == 0, 1e-9, S_pre_expanded)
    
    signal_ratio_term = dce_series_data / S_pre_safe
    E1_0_term = 1.0 - np.exp(-TR * R1_0_expanded)
    
    log_arg = 1.0 - signal_ratio_term * E1_0_term
    log_arg = np.maximum(log_arg, 1e-9) # Clip to avoid log(0) or log(negative)
    
    R1_t = (-1.0 / TR) * np.log(log_arg)
    delta_R1_t = R1_t - R1_0_expanded
    
    # Add epsilon to r1_relaxivity in denominator (already checked >0, but good practice)
    Ct_data = delta_R1_t / (r1_relaxivity + 1e-9)

    return Ct_data


def signal_tc_to_concentration_tc(
    signal_tc: np.ndarray, 
    t10_scalar: float, 
    r1_relaxivity: float, 
    TR: float, 
    baseline_time_points: int = 5
) -> np.ndarray:
    """
    Converts a single time-course (TC) of signal intensity to concentration.

    This is a scalar version of the main signal_to_concentration, intended for
    processing a single voxel's or ROI's mean signal time-course, typically for AIF extraction.

    Args:
        signal_tc (np.ndarray): 1D array of signal intensities over time.
        t10_scalar (float): Pre-contrast T1 value (in seconds) for this specific tissue/ROI.
        r1_relaxivity (float): Longitudinal relaxivity of the contrast agent (s⁻¹ mM⁻¹).
        TR (float): Repetition Time (in seconds).
        baseline_time_points (int, optional): Number of initial time points
                                             to average for S_pre_tc. Defaults to 5.

    Returns:
        np.ndarray: 1D array of concentration-time curve.

    Raises:
        ValueError: If inputs are invalid (e.g., signal_tc not 1D, non-positive parameters,
                    insufficient time points).
    """
    if not isinstance(signal_tc, np.ndarray) or signal_tc.ndim != 1:
        raise ValueError("signal_tc must be a 1D NumPy array.")
    if not isinstance(t10_scalar, (int, float)) or t10_scalar <= 0:
        raise ValueError("t10_scalar must be a positive number.")
    if not isinstance(r1_relaxivity, (int, float)) or r1_relaxivity <= 0:
        raise ValueError("r1_relaxivity must be a positive number.")
    if not isinstance(TR, (int, float)) or TR <= 0:
        raise ValueError("TR must be a positive number.")
    if not isinstance(baseline_time_points, int) or baseline_time_points <= 0:
        raise ValueError("baseline_time_points must be a positive integer.")
    if baseline_time_points >= len(signal_tc):
        raise ValueError(
            "baseline_time_points must be less than the number of time points "
            "in signal_tc."
        )
    if len(signal_tc) == 0:
        raise ValueError("signal_tc cannot be empty.")

    S_pre_tc = np.mean(signal_tc[:baseline_time_points])
    S_pre_tc = S_pre_tc if S_pre_tc > 1e-9 else 1e-9  # Avoid division by zero

    R1_0_scalar = 1.0 / (t10_scalar + 1e-9)
    E1_0_term = (1.0 - np.exp(-TR * R1_0_scalar))
    
    Ct_tc = np.zeros_like(signal_tc, dtype=float)

    for i in range(len(signal_tc)):
        S_t_current = signal_tc[i]
        signal_ratio_term = S_t_current / S_pre_tc
        log_arg = 1.0 - signal_ratio_term * E1_0_term
        log_arg = np.maximum(log_arg, 1e-9)  # Clipping
        
        R1_t_current = (-1.0 / TR) * np.log(log_arg)
        delta_R1_t_current = R1_t_current - R1_0_scalar
        Ct_tc[i] = delta_R1_t_current / (r1_relaxivity + 1e-9) # Add epsilon to r1_relaxivity
        
    return Ct_tc
