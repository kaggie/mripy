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

    # Calculate S_pre (baseline signal)
    # Add a small epsilon to S_pre in the denominator later to avoid division by zero.
    S_pre = np.mean(dce_series_data[..., :baseline_time_points], axis=3)

    # Calculate R1_0 (pre-contrast relaxation rate)
    # Add a small epsilon to t10_map_data to avoid division by zero.
    R1_0 = 1.0 / (t10_map_data + 1e-9)  # Ensure t10_map_data is not zero

    # Initialize Ct_data
    num_time_points = dce_series_data.shape[3]
    Ct_data = np.zeros_like(dce_series_data, dtype=float)

    # Iterate over each time point
    for t in range(num_time_points):
        S_t = dce_series_data[..., t]

        # Calculate E1_0_term = (1.0 - np.exp(-TR * R1_0))
        E1_0_term = 1.0 - np.exp(-TR * R1_0)

        # Calculate signal_ratio_term = S_t / (S_pre + epsilon)
        # Add epsilon to S_pre to avoid division by zero.
        signal_ratio_term = S_t / (S_pre + 1e-9)

        # Calculate log_arg = 1.0 - signal_ratio_term * E1_0_term
        log_arg = 1.0 - signal_ratio_term * E1_0_term

        # Clip log_arg to a small positive value to avoid log(0) or log(negative)
        log_arg = np.maximum(log_arg, 1e-9)

        # Calculate R1_t = (-1.0 / TR) * np.log(log_arg)
        R1_t = (-1.0 / TR) * np.log(log_arg)

        # Calculate delta_R1_t = R1_t - R1_0
        delta_R1_t = R1_t - R1_0

        # Calculate Ct_t = delta_R1_t / r1_relaxivity
        # r1_relaxivity is validated to be > 0, so no division by zero here
        Ct_t = delta_R1_t / r1_relaxivity

        Ct_data[..., t] = Ct_t

    return Ct_data
