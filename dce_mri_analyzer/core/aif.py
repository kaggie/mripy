import numpy as np
import csv
import os
import json 
from ..core import conversion 

# --- AIF Parameter Metadata ---
AIF_PARAMETER_METADATA = {
    "parker": [
        # Parameter name, Default value, Min value, Max value, Tooltip (optional)
        # Units for Parker: time in minutes. A1/A2 in mM*min, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor (e.g., dose adjustment)"),
        ('A1', 0.809, 0.0, 5.0, "Amplitude of first exponential (mM*min)"),
        ('m1', 0.171, 0.0, 5.0, "Decay rate of first exponential (min^-1)"),
        ('A2', 0.330, 0.0, 5.0, "Amplitude of second exponential (mM*min)"),
        ('m2', 2.05, 0.0, 10.0, "Decay rate of second exponential (min^-1)")
    ],
    "weinmann": [
        # Units for Weinmann: time in minutes. A1/A2 in mM*min, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor"),
        ('A1', 3.99, 0.0, 10.0, "Amplitude of first exponential (mM*min)"), 
        ('m1', 0.144, 0.0, 2.0, "Decay rate of first exponential (min^-1)"), 
        ('A2', 4.78, 0.0, 10.0, "Amplitude of second exponential (mM*min)"),
        ('m2', 0.0111, 0.0, 1.0, "Decay rate of second exponential (min^-1)")
    ],
    "fast_biexponential": [
        # Units: time in minutes. A1/A2 unitless proportions, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor"),
        ('A1', 0.6, 0.0, 1.0, "Proportion of first exponential"),
        ('m1', 3.0, 0.0, 10.0, "Decay rate of first exponential (min^-1)"),
        ('A2', 0.4, 0.0, 1.0, "Proportion of second exponential"),
        ('m2', 0.3, 0.0, 5.0, "Decay rate of second exponential (min^-1)")
    ]
}

def load_aif_from_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(filepath): raise FileNotFoundError(f"AIF file not found at: {filepath}")
    time_points, concentrations = [], []
    try:
        with open(filepath, 'r', newline='') as f:
            content_to_sniff = f.read(1024); f.seek(0)
            is_csv = False
            if filepath.lower().endswith(".csv"):
                try: dialect = csv.Sniffer().sniff(content_to_sniff); reader = csv.reader(f, dialect); is_csv = True
                except csv.Error: pass 
            if not is_csv: 
                lines = f.readlines()
                if not lines: raise ValueError(f"AIF file is empty: {filepath}")
                first_line_parts = lines[0].strip().split(); start_line_index = 0
                if first_line_parts: 
                    try: float(first_line_parts[0]) 
                    except ValueError: start_line_index = 1 
                if start_line_index >= len(lines) and len(lines) > 0: raise ValueError(f"No numeric data found after header in AIF file: {filepath}")
                if not lines[start_line_index:]: raise ValueError(f"No numeric data found in AIF file: {filepath}")
                for line_num, line_content in enumerate(lines[start_line_index:]):
                    parts = line_content.strip().split()
                    if not parts: continue
                    if len(parts) != 2: raise ValueError(f"Incorrect format in AIF file: {filepath} at line {line_num + start_line_index + 1}. Expected 2 columns, got {len(parts)}.")
                    try: time_points.append(float(parts[0])); concentrations.append(float(parts[1]))
                    except ValueError: raise ValueError(f"Non-numeric data found in AIF file: {filepath} at line {line_num + start_line_index + 1}.")
            else: 
                header_skipped = False
                for i, row in enumerate(reader):
                    if not row: continue
                    if not header_skipped:
                        try: float(row[0])
                        except ValueError: header_skipped = True; continue  
                    if len(row) != 2: raise ValueError(f"Incorrect format in AIF file: {filepath} at line {i + 1}. Expected 2 columns, got {len(row)}.")
                    try: time_points.append(float(row[0])); concentrations.append(float(row[1]))
                    except ValueError: raise ValueError(f"Non-numeric data found in AIF file: {filepath} at line {i + 1} after potential header.")
            if not time_points: raise ValueError(f"No numeric data found in AIF file: {filepath}")
    except Exception as e: raise ValueError(f"Error reading AIF file {filepath}: {e}")
    return np.array(time_points), np.array(concentrations)

def save_aif_curve(time_points: np.ndarray, concentrations: np.ndarray, filepath: str):
    if len(time_points) != len(concentrations): raise ValueError("Time points and concentrations arrays must have the same length.")
    if time_points.ndim != 1 or concentrations.ndim != 1: raise ValueError("Time points and concentrations must be 1D arrays.")
    data = np.vstack((time_points, concentrations)).T 
    try:
        delimiter = ',' if filepath.lower().endswith('.csv') else '\t'
        if not (filepath.lower().endswith('.csv') or filepath.lower().endswith('.txt')):
            print(f"Warning: Unknown file extension for AIF curve ('{os.path.splitext(filepath)[1]}'), saving as CSV with delimiter ','.")
            delimiter = ',' 
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter); writer.writerow(['Time', 'Concentration']); writer.writerows(data)
    except Exception as e: raise IOError(f"Failed to save AIF curve to {filepath}: {e}")

def parker_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 0.809, m1: float = 0.171, A2: float = 0.330, m2: float = 2.05) -> np.ndarray:
    """
    Implements a bi-exponential Parker Arterial Input Function (AIF).
    Based on Parker et al. (2006), Magn Reson Med, 56(5), 993-1000.
    Default parameters (A1, m1, A2, m2) assume time_points are in minutes.
    Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))

    Args:
        time_points (np.ndarray): Array of time points (typically in minutes).
        D_scaler (float, optional): Overall scaling factor. Defaults to 1.0.
        A1 (float, optional): Amplitude of the first exponential term (mM*min). Defaults to 0.809.
        m1 (float, optional): Decay rate of the first exponential term (min⁻¹). Defaults to 0.171.
        A2 (float, optional): Amplitude of the second exponential term (mM*min). Defaults to 0.330.
        m2 (float, optional): Decay rate of the second exponential term (min⁻¹). Defaults to 2.05.
    Returns: np.ndarray: AIF concentration values.
    Raises: TypeError, ValueError for invalid inputs.
    """
    if not isinstance(time_points, np.ndarray): raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 : raise ValueError("AIF parameters must be non-negative.")
    valid_time_points = np.maximum(time_points, 0); term1 = A1 * np.exp(-m1 * valid_time_points); term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

def weinmann_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 3.99, m1: float = 0.144, A2: float = 4.78, m2: float = 0.0111) -> np.ndarray:
    '''
    Weinmann population-averaged AIF.
    Cp(t) = D_scaler * (A1*exp(-m1*t) + A2*exp(-m2*t)).
    Default parameters based on Weinmann et al. (1982), Am J Roentgenol, 142(3), 619-624.
    Parameters assume time_points are in minutes.

    Args:
        time_points (np.ndarray): Time points for AIF calculation (typically in minutes).
        D_scaler (float): General scaling factor. Defaults to 1.0.
        A1 (float): Amplitude of the first exponential component. Defaults to 3.99.
        m1 (float): Rate constant of the first exponential component (min⁻¹). Defaults to 0.144.
        A2 (float): Amplitude of the second exponential component. Defaults to 4.78.
        m2 (float): Rate constant of the second exponential component (min⁻¹). Defaults to 0.0111.
    Returns: np.ndarray: Concentration values for the AIF.
    Raises: TypeError, ValueError for invalid inputs.
    '''
    if not isinstance(time_points, np.ndarray): raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 : raise ValueError("AIF parameters must be non-negative.")
    valid_time_points = np.maximum(time_points, 0); term1 = A1 * np.exp(-m1 * valid_time_points); term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

def fast_biexponential_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 0.6, m1: float = 3.0, A2: float = 0.4, m2: float = 0.3) -> np.ndarray:
    '''
    A 'Fast' Bi-exponential population-averaged AIF.
    Cp(t) = D_scaler * (A1*exp(-m1*t) + A2*exp(-m2*t)).
    Assumes time_points are in minutes for these default rate constants.

    Args:
        time_points (np.ndarray): Time points for AIF calculation (e.g., in minutes).
        D_scaler (float): General scaling factor. Defaults to 1.0.
        A1 (float): Amplitude/proportion of the first exponential component. Defaults to 0.6.
        m1 (float): Rate constant of the first exponential component (min⁻¹). Defaults to 3.0.
        A2 (float): Amplitude/proportion of the second exponential component. Defaults to 0.4.
        m2 (float): Rate constant of the second exponential component (min⁻¹). Defaults to 0.3.
    Returns:
        np.ndarray: Concentration values for the AIF.
    Raises: TypeError, ValueError for invalid inputs.
    '''
    if not isinstance(time_points, np.ndarray): raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 : raise ValueError("AIF parameters must be non-negative.")
    valid_time_points = np.maximum(time_points, 0)
    term1 = A1 * np.exp(-m1 * valid_time_points); term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

POPULATION_AIFS = {
    "parker": parker_aif,
    "weinmann": weinmann_aif,
    "fast_biexponential": fast_biexponential_aif,
}

def generate_population_aif(name: str, time_points: np.ndarray, params: dict = None) -> np.ndarray | None:
    if name in POPULATION_AIFS:
        model_function = POPULATION_AIFS[name]
        try:
            if params: return model_function(time_points, **params)
            else: return model_function(time_points) 
        except TypeError as e: raise ValueError(f"Error calling AIF model '{name}' with provided parameters: {e}")
        except Exception as e: print(f"Unexpected error generating population AIF '{name}': {e}"); return None 
    else: return None

def extract_aif_from_roi(dce_4d_data, roi_2d_coords, slice_index_z, t10_blood, r1_blood, TR, baseline_time_points_aif=5):
    if dce_4d_data.ndim != 4: raise ValueError("dce_4d_data must be a 4D array.")
    x_start, y_start, width, height = roi_2d_coords
    if not (0<=x_start<dce_4d_data.shape[0] and 0<=y_start<dce_4d_data.shape[1] and 0<=slice_index_z<dce_4d_data.shape[2]): raise ValueError(f"ROI start/Z out of bounds.")
    if not (x_start+width<=dce_4d_data.shape[0] and y_start+height<=dce_4d_data.shape[1]): raise ValueError(f"ROI dimensions exceed bounds.")
    if width<=0 or height<=0: raise ValueError("ROI width/height must be positive.")
    roi_patch_3d = dce_4d_data[x_start:x_start+width, y_start:y_start+height, slice_index_z, :]
    if roi_patch_3d.size == 0: raise ValueError("ROI patch is empty.")
    mean_roi_signal_tc = np.mean(roi_patch_3d, axis=(0,1)) 
    if len(mean_roi_signal_tc) == 0: raise ValueError("Mean ROI signal TC empty.")
    aif_conc_tc = conversion.signal_tc_to_concentration_tc(mean_roi_signal_tc, t10_blood, r1_blood, TR, baseline_time_points_aif)
    aif_time_tc = np.arange(len(mean_roi_signal_tc)) * TR
    return aif_time_tc, aif_conc_tc

def save_aif_roi_definition(roi_properties: dict, filepath: str):
    try:
        with open(filepath, 'w') as f: json.dump(roi_properties, f, indent=4)
    except (IOError, TypeError) as e: raise IOError(f"Error saving AIF ROI definition to {filepath}: {e}")

def load_aif_roi_definition(filepath: str) -> dict | None:
    required_keys = ["slice_index", "pos_x", "pos_y", "size_w", "size_h", "image_ref_name"]
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        if not isinstance(data, dict): raise ValueError("ROI definition file not a valid JSON object.")
        for key in required_keys:
            if key not in data: raise ValueError(f"Missing required key: '{key}'")
        if not isinstance(data["slice_index"], int): raise ValueError("slice_index must be an integer.")
        if not all(isinstance(data[k], (int, float)) for k in ["pos_x", "pos_y", "size_w", "size_h"]): raise ValueError("ROI position/size values must be numeric.")
        if not isinstance(data["image_ref_name"], str): raise ValueError("image_ref_name must be a string.")
        return data
    except FileNotFoundError: raise 
    except json.JSONDecodeError as e: raise ValueError(f"Error decoding JSON from AIF ROI file {filepath}: {e}")
    except IOError as e: raise IOError(f"Error reading AIF ROI definition from {filepath}: {e}")
