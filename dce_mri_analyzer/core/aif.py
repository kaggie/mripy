import numpy as np
import csv
import os
from ..core import conversion # Added for signal_tc_to_concentration_tc

def load_aif_from_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads an AIF from a TXT or CSV file.
    The file is expected to have two columns: time and concentration.
    A header row is optionally supported and will be skipped if present.

    Args:
        filepath (str): Path to the AIF file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       time_points and concentrations.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not in the expected format (e.g., wrong number
                    of columns, non-numeric data after skipping header, empty file).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AIF file not found at: {filepath}")

    time_points = []
    concentrations = []

    try:
        with open(filepath, 'r', newline='') as f:
            content_to_sniff = f.read(1024)
            f.seek(0)  # Reset file pointer

            is_csv = False
            if filepath.lower().endswith(".csv"):
                try:
                    dialect = csv.Sniffer().sniff(content_to_sniff)
                    reader = csv.reader(f, dialect)
                    is_csv = True
                except csv.Error:
                    pass 

            if not is_csv: 
                lines = f.readlines()
                if not lines:
                    raise ValueError(f"AIF file is empty: {filepath}")

                first_line_parts = lines[0].strip().split()
                start_line_index = 0
                if first_line_parts: 
                    try:
                        float(first_line_parts[0]) 
                    except ValueError:
                        start_line_index = 1 
                
                if start_line_index >= len(lines) and len(lines) > 0: 
                     raise ValueError(f"No numeric data found after header in AIF file: {filepath}")
                if not lines[start_line_index:]: 
                     raise ValueError(f"No numeric data found in AIF file: {filepath}")


                for line_num, line_content in enumerate(lines[start_line_index:]):
                    parts = line_content.strip().split()
                    if not parts:  
                        continue
                    if len(parts) != 2:
                        raise ValueError(
                            f"Incorrect format in AIF file: {filepath} at line {line_num + start_line_index + 1}. "
                            f"Expected 2 columns, got {len(parts)}."
                        )
                    try:
                        time_points.append(float(parts[0]))
                        concentrations.append(float(parts[1]))
                    except ValueError:
                         raise ValueError(
                            f"Non-numeric data found in AIF file: {filepath} at line {line_num + start_line_index + 1}."
                        )
            else: 
                header_skipped = False
                for i, row in enumerate(reader):
                    if not row:  
                        continue
                    if not header_skipped:
                        try:
                            float(row[0])
                        except ValueError:
                            header_skipped = True
                            continue  
                    
                    if len(row) != 2:
                        raise ValueError(
                            f"Incorrect format in AIF file: {filepath} at line {i + 1}. "
                            f"Expected 2 columns, got {len(row)}."
                        )
                    try:
                        time_points.append(float(row[0]))
                        concentrations.append(float(row[1]))
                    except ValueError:
                        raise ValueError(
                            f"Non-numeric data found in AIF file: {filepath} at line {i + 1} "
                            f"after potential header."
                        )
            
            if not time_points:  
                raise ValueError(f"No numeric data found in AIF file: {filepath}")

    except FileNotFoundError: 
        raise
    except ValueError:  
        raise
    except Exception as e:  
        raise ValueError(f"Error reading AIF file {filepath}: {e}")

    return np.array(time_points), np.array(concentrations)


def parker_aif(time_points: np.ndarray, D=1.0, A1=0.809, m1=0.171, A2=0.330, m2=2.05) -> np.ndarray:
    """
    Implements a bi-exponential Parker Arterial Input Function (AIF).
    This is a common simplification.

    The formula used:
    Cp(t) = D * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))

    Args:
        time_points (np.ndarray): Array of time points (units should be consistent
                                   with m1, m2 units, e.g., minutes or seconds).
        D (float, optional): Overall scaling factor. Defaults to 1.0.
        A1 (float, optional): Amplitude of the first exponential term. Defaults to 0.809.
        m1 (float, optional): Decay rate of the first exponential term. Defaults to 0.171.
                               (Units should match 1/time_points units).
        A2 (float, optional): Amplitude of the second exponential term. Defaults to 0.330.
        m2 (float, optional): Decay rate of the second exponential term. Defaults to 2.05.
                               (Units should match 1/time_points units).

    Returns:
        np.ndarray: Array of AIF concentration values corresponding to time_points.
    
    Raises:
        TypeError: if time_points is not a NumPy array.
        ValueError: if any of the parameters D, A1, A2, m1, m2 are negative.
    """
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 :
        raise ValueError("AIF parameters (D, A1, A2, m1, m2) must be non-negative.")

    term1 = A1 * np.exp(-m1 * time_points)
    term2 = A2 * np.exp(-m2 * time_points)
    Cp_t = D * (term1 + term2)
    return Cp_t

POPULATION_AIFS = {
    "parker": parker_aif,
}

def generate_population_aif(name: str, time_points: np.ndarray, params: dict = None) -> np.ndarray | None:
    """
    Generates an AIF using a predefined population model.

    Args:
        name (str): The name of the population AIF model to use (e.g., "parker").
        time_points (np.ndarray): Array of time points for which to generate the AIF.
        params (dict, optional): Dictionary of parameters to pass to the AIF model function.
                                 If None, the model's default parameters will be used.
                                 Example: {'D': 1.0, 'A1': 0.8, ...}

    Returns:
        np.ndarray | None: An array of AIF concentration values, or None if the
                           model name is not found or if there's an error during generation.
    Raises:
        ValueError: if the model requires parameters not provided in the params dict
                    or if parameters are of incorrect type for the model.
    """
    if name in POPULATION_AIFS:
        model_function = POPULATION_AIFS[name]
        try:
            if params:
                return model_function(time_points, **params)
            else:
                return model_function(time_points) 
        except TypeError as e:
            raise ValueError(f"Error calling AIF model '{name}' with provided parameters: {e}")
        except Exception as e: 
            print(f"Unexpected error generating population AIF '{name}': {e}")
            return None 
    else:
        return None

def extract_aif_from_roi(
    dce_4d_data: np.ndarray, 
    roi_2d_coords: tuple, # (x_start, y_start, width, height) in original X, Y index space
    slice_index_z: int, 
    t10_blood: float, 
    r1_blood: float, 
    TR: float,
    baseline_time_points_aif: int = 5 
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts an Arterial Input Function (AIF) by averaging the signal within a
    Region of Interest (ROI) in the DCE data and converting it to concentration.

    Args:
        dce_4d_data (np.ndarray): The 4D DCE image data (X, Y, Z, Time).
        roi_2d_coords (tuple): (x_start, y_start, width, height) defining the ROI 
                               in the original X, Y index space of the dce_4d_data.
        slice_index_z (int): The Z index of the slice where the ROI is defined.
        t10_blood (float): Pre-contrast T1 value of blood (in seconds).
        r1_blood (float): Longitudinal relaxivity of the contrast agent in blood
                          (e.g., in s⁻¹ mM⁻¹).
        TR (float): Repetition Time (in seconds).
        baseline_time_points_aif (int, optional): Number of initial time points
                                                 to use for baseline calculation
                                                 for the AIF. Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       aif_time_tc (time points for the AIF) and
                                       aif_concentration_tc (AIF concentrations).

    Raises:
        ValueError: If ROI coordinates are out of bounds or parameters are invalid.
    """
    if dce_4d_data.ndim != 4:
        raise ValueError("dce_4d_data must be a 4D array.")
    
    x_start, y_start, width, height = roi_2d_coords

    # Validate coordinates against dce_4d_data.shape (X, Y, Z, T)
    if not (0 <= x_start < dce_4d_data.shape[0] and \
            0 <= y_start < dce_4d_data.shape[1] and \
            0 <= slice_index_z < dce_4d_data.shape[2]):
        raise ValueError(f"ROI start coordinates or Z-slice index out of bounds for DCE data shape {dce_4d_data.shape}.")
    
    if not (x_start + width <= dce_4d_data.shape[0] and \
            y_start + height <= dce_4d_data.shape[1]):
        raise ValueError(f"ROI dimensions (start+size) exceed DCE data spatial bounds. "
                         f"X: {x_start}+{width} vs {dce_4d_data.shape[0]}, "
                         f"Y: {y_start}+{height} vs {dce_4d_data.shape[1]}.")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive.")

    # Extract the 3D region for ROI: dce_4d_data[X, Y, Z, Time]
    # ROI patch is (width_x, height_y, num_timepoints)
    roi_patch_3d = dce_4d_data[x_start : x_start + width, 
                               y_start : y_start + height, 
                               slice_index_z, 
                               :]
    
    if roi_patch_3d.size == 0: # Should be caught by width/height check, but as safeguard
        raise ValueError("ROI patch is empty. Check ROI coordinates and dimensions.")

    # Calculate mean_roi_signal_tc: average over X and Y axes of the patch
    mean_roi_signal_tc = np.mean(roi_patch_3d, axis=(0, 1)) # Result is 1D array (time)

    if len(mean_roi_signal_tc) == 0:
        raise ValueError("Mean ROI signal time course is empty.")

    # Call conversion.signal_tc_to_concentration_tc
    aif_concentration_tc = conversion.signal_tc_to_concentration_tc(
        signal_tc=mean_roi_signal_tc,
        t10_scalar=t10_blood,
        r1_relaxivity=r1_blood,
        TR=TR,
        baseline_time_points=baseline_time_points_aif
    )

    # Create time vector for the AIF
    aif_time_tc = np.arange(len(mean_roi_signal_tc)) * TR
    
    return aif_time_tc, aif_concentration_tc
