import numpy as np
import csv
import os
import json # Added for saving/loading ROI definitions
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
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0 :
        raise ValueError("AIF parameters (D, A1, A2, m1, m2) must be non-negative.")
    term1 = A1 * np.exp(-m1 * time_points)
    term2 = A2 * np.exp(-m2 * time_points)
    Cp_t = D * (term1 + term2)
    return Cp_t

POPULATION_AIFS = {"parker": parker_aif}

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
    """
    Saves AIF ROI properties to a JSON file.

    Args:
        roi_properties (dict): Dictionary containing ROI properties. Expected keys:
                               "slice_index": int, "pos_x": float, "pos_y": float,
                               "size_w": float, "size_h": float, "image_ref_name": str.
        filepath (str): Path to save the JSON file.

    Raises:
        IOError: If there is an error writing the file.
        TypeError: If roi_properties is not JSON serializable.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(roi_properties, f, indent=4)
    except (IOError, TypeError) as e: # Catch more specific errors if json.dump can raise them
        raise IOError(f"Error saving AIF ROI definition to {filepath}: {e}")

def load_aif_roi_definition(filepath: str) -> dict | None:
    """
    Loads AIF ROI properties from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict | None: A dictionary containing the loaded ROI properties, or None if loading fails.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or missing required keys.
        IOError: If there is an error reading the file.
    """
    required_keys = ["slice_index", "pos_x", "pos_y", "size_w", "size_h", "image_ref_name"]
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("ROI definition file does not contain a valid JSON object (dictionary).")

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in AIF ROI definition file: '{key}'")
        
        # Optional type checks, though JSON loads numbers as float/int typically
        if not isinstance(data["slice_index"], int): raise ValueError("slice_index must be an integer.")
        if not all(isinstance(data[k], (int, float)) for k in ["pos_x", "pos_y", "size_w", "size_h"]):
            raise ValueError("ROI position/size values must be numeric.")
        if not isinstance(data["image_ref_name"], str): raise ValueError("image_ref_name must be a string.")

        return data
    except FileNotFoundError:
        # Let FileNotFoundError propagate or handle specifically if needed by UI
        raise 
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from AIF ROI file {filepath}: {e}")
    except IOError as e: # Catch read errors
        raise IOError(f"Error reading AIF ROI definition from {filepath}: {e}")
    # ValueError from key/type checks will also propagate
    # Return None for other unexpected errors, or let them propagate
    # For now, let specific exceptions propagate for clearer error messages in UI.
    # If a generic "return None" is preferred for all errors:
    # except Exception as e:
    #     print(f"Failed to load AIF ROI definition: {e}") # Or log to console
    #     return None
