import numpy as np
import csv
import os
import json 
from ..core import conversion 

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
    This is a common simplification based on Parker et al. (2006), "Experimentally-derived
    functional form for a population-averaged high-temporal-resolution arterial input function
    for dynamic contrast-enhanced MRI". Magn Reson Med, 56(5), 993-1000.
    The parameters (A1, m1, A2, m2) are typically used when time is in minutes.
    The output concentration is scaled by D, which can incorporate dose and patient factors.

    The formula used:
    Cp(t) = D * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))

    Args:
        time_points (np.ndarray): Array of time points (units should be consistent
                                   with m1, m2 units, typically minutes).
        D (float, optional): Overall scaling factor (e.g., for dose or normalization). Defaults to 1.0.
        A1 (float, optional): Amplitude of the first exponential term. Defaults to 0.809 (mM min).
        m1 (float, optional): Decay rate of the first exponential term. Defaults to 0.171 (min⁻¹).
        A2 (float, optional): Amplitude of the second exponential term. Defaults to 0.330 (mM min).
        m2 (float, optional): Decay rate of the second exponential term. Defaults to 2.05 (min⁻¹).

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
    
    valid_time_points = np.maximum(time_points, 0) # Ensure time is not negative for exp
    term1 = A1 * np.exp(-m1 * valid_time_points)
    term2 = A2 * np.exp(-m2 * valid_time_points)
    Cp_t = D * (term1 + term2)
    return Cp_t

def weinmann_aif(time_points: np.ndarray, 
                 D_scaler: float = 1.0, 
                 A1: float = 3.99,  # Units: (mM * min) or similar depending on D_scaler and time units
                 m1: float = 0.144, # Units: min^-1
                 A2: float = 4.78,  # Units: (mM * min)
                 m2: float = 0.0111 # Units: min^-1
                ) -> np.ndarray:
    '''
    Weinmann population-averaged AIF.
    A common bi-exponential form: Cp(t) = D_scaler * (A1*exp(-m1*t) + A2*exp(-m2*t)).
    The default parameters (A1, m1, A2, m2) are based on Weinmann et al. (1982),
    "Characteristics of Gadolinium-DTPA Complex: A Potential NMR Contrast Agent". 
    Am J Roentgenol, 142(3), 619-624, often cited and adapted.
    These parameters are typically used when time is in minutes, and D_scaler
    can be used to adjust for dose (e.g., mmol/kg) and patient weight to yield mM.
    For generic use, D_scaler can be 1.0 and the output is proportional to concentration.

    Args:
        time_points (np.ndarray): Time points for AIF calculation (typically in minutes).
        D_scaler (float): General scaling factor (e.g., for dose/weight adjustment or normalization).
        A1 (float): Amplitude/proportion of the first exponential component.
        m1 (float): Rate constant of the first exponential component (typically min^-1).
        A2 (float): Amplitude/proportion of the second exponential component.
        m2 (float): Rate constant of the second exponential component (typically min^-1).
    Returns:
        np.ndarray: Concentration values for the AIF.
    '''
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 :
        raise ValueError("AIF parameters (D_scaler, A1, A2, m1, m2) must be non-negative.")
    
    valid_time_points = np.maximum(time_points, 0) # Ensure time is not negative
    
    term1 = A1 * np.exp(-m1 * valid_time_points)
    term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)


POPULATION_AIFS = {
    "parker": parker_aif,
    "weinmann": weinmann_aif,
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

def extract_aif_from_roi(
    dce_4d_data: np.ndarray, 
    roi_2d_coords: tuple, 
    slice_index_z: int, 
    t10_blood: float, 
    r1_blood: float, 
    TR: float,
    baseline_time_points_aif: int = 5 
) -> tuple[np.ndarray, np.ndarray]:
    if dce_4d_data.ndim != 4: raise ValueError("dce_4d_data must be a 4D array.")
    x_start, y_start, width, height = roi_2d_coords
    if not (0 <= x_start < dce_4d_data.shape[0] and 0 <= y_start < dce_4d_data.shape[1] and 0 <= slice_index_z < dce_4d_data.shape[2]):
        raise ValueError(f"ROI start coordinates or Z-slice index out of bounds for DCE data shape {dce_4d_data.shape}.")
    if not (x_start + width <= dce_4d_data.shape[0] and y_start + height <= dce_4d_data.shape[1]):
        raise ValueError(f"ROI dimensions exceed DCE data spatial bounds.")
    if width <= 0 or height <= 0: raise ValueError("ROI width and height must be positive.")

    roi_patch_3d = dce_4d_data[x_start : x_start + width, y_start : y_start + height, slice_index_z, :]
    if roi_patch_3d.size == 0: raise ValueError("ROI patch is empty.")
    mean_roi_signal_tc = np.mean(roi_patch_3d, axis=(0, 1)) 
    if len(mean_roi_signal_tc) == 0: raise ValueError("Mean ROI signal time course is empty.")
    aif_concentration_tc = conversion.signal_tc_to_concentration_tc(mean_roi_signal_tc, t10_blood, r1_blood, TR, baseline_time_points_aif)
    aif_time_tc = np.arange(len(mean_roi_signal_tc)) * TR
    return aif_time_tc, aif_concentration_tc

def save_aif_roi_definition(roi_properties: dict, filepath: str):
    try:
        with open(filepath, 'w') as f: json.dump(roi_properties, f, indent=4)
    except (IOError, TypeError) as e: raise IOError(f"Error saving AIF ROI definition to {filepath}: {e}")

def load_aif_roi_definition(filepath: str) -> dict | None:
    required_keys = ["slice_index", "pos_x", "pos_y", "size_w", "size_h", "image_ref_name"]
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        if not isinstance(data, dict): raise ValueError("ROI definition file does not contain a valid JSON object (dictionary).")
        for key in required_keys:
            if key not in data: raise ValueError(f"Missing required key in AIF ROI definition file: '{key}'")
        if not isinstance(data["slice_index"], int): raise ValueError("slice_index must be an integer.")
        if not all(isinstance(data[k], (int, float)) for k in ["pos_x", "pos_y", "size_w", "size_h"]):
            raise ValueError("ROI position/size values must be numeric.")
        if not isinstance(data["image_ref_name"], str): raise ValueError("image_ref_name must be a string.")
        return data
    except FileNotFoundError: raise 
    except json.JSONDecodeError as e: raise ValueError(f"Error decoding JSON from AIF ROI file {filepath}: {e}")
    except IOError as e: raise IOError(f"Error reading AIF ROI definition from {filepath}: {e}")
