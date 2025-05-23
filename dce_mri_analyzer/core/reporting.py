import numpy as np
import csv

def calculate_roi_statistics(data_map_slice: np.ndarray, roi_mask_slice: np.ndarray) -> dict | None:
    """
    Calculates basic statistics for values within an ROI on a 2D data slice.

    Args:
        data_map_slice (np.ndarray): The 2D NumPy array of the data slice.
        roi_mask_slice (np.ndarray): A 2D boolean NumPy array of the same shape as
                                     data_map_slice, where True indicates pixels
                                     within the ROI.

    Returns:
        dict | None: A dictionary containing statistics (N, Mean, StdDev, Median, Min, Max)
                     if the ROI contains valid data points. Returns a dict with N=0 and NaN for
                     stats if ROI is empty or all values are NaN. Returns None if inputs are invalid.
    """
    if not isinstance(data_map_slice, np.ndarray) or data_map_slice.ndim != 2:
        raise ValueError("data_map_slice must be a 2D NumPy array.")
    if not isinstance(roi_mask_slice, np.ndarray) or roi_mask_slice.ndim != 2:
        raise ValueError("roi_mask_slice must be a 2D NumPy array.")
    if data_map_slice.shape != roi_mask_slice.shape:
        raise ValueError("data_map_slice and roi_mask_slice must have the same shape.")

    # Ensure roi_mask_slice is boolean
    roi_mask_slice = roi_mask_slice.astype(bool)
    
    roi_values = data_map_slice[roi_mask_slice]

    if roi_values.size == 0:
        return {"N": 0, "Mean": np.nan, "StdDev": np.nan, 
                "Median": np.nan, "Min": np.nan, "Max": np.nan}
    
    # Calculate statistics, ignoring NaNs within the ROI
    # If all values in ROI are NaN, nan-functions will return NaN, which is appropriate.
    stats = {
        "N": roi_values.size, # Total number of pixels in ROI
        "N_valid": np.sum(~np.isnan(roi_values)), # Number of non-NaN pixels
        "Mean": np.nanmean(roi_values),
        "StdDev": np.nanstd(roi_values),
        "Median": np.nanmedian(roi_values),
        "Min": np.nanmin(roi_values),
        "Max": np.nanmax(roi_values)
    }
    return stats

def format_roi_statistics_to_string(stats_dict: dict | None, map_name: str, roi_name: str = "ROI") -> str:
    """
    Formats ROI statistics into a human-readable string.

    Args:
        stats_dict (dict | None): Dictionary of statistics from calculate_roi_statistics.
        map_name (str): Name of the map on which statistics were calculated.
        roi_name (str, optional): Name of the ROI. Defaults to "ROI".

    Returns:
        str: A formatted string of the statistics.
    """
    if stats_dict is None or stats_dict.get("N_valid", 0) == 0: # Check N_valid
        return f"No valid data in {roi_name} for '{map_name}'."

    lines = [f"Statistics for {roi_name} on '{map_name}':"]
    for key, value in stats_dict.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)

def save_roi_statistics_csv(stats_dict: dict, filepath: str, map_name: str, roi_name: str = "ROI"):
    """
    Saves ROI statistics to a CSV file.

    Args:
        stats_dict (dict): Dictionary of statistics.
        filepath (str): Path to save the CSV file.
        map_name (str): Name of the map.
        roi_name (str, optional): Name of the ROI. Defaults to "ROI".

    Raises:
        ValueError: If stats_dict is None or empty.
        IOError: If file writing fails.
    """
    if not stats_dict: # Handles None or empty dict
        raise ValueError("No statistics data to save.")

    fieldnames = ['MapName', 'ROIName', 'Statistic', 'Value']
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for stat_name, stat_value in stats_dict.items():
                writer.writerow({
                    'MapName': map_name, 
                    'ROIName': roi_name, 
                    'Statistic': stat_name, 
                    'Value': stat_value
                })
    except IOError as e:
        raise IOError(f"Error writing ROI statistics to CSV file {filepath}: {e}")
    except Exception as e: # Catch any other unexpected errors during CSV writing
        raise Exception(f"An unexpected error occurred while saving ROI statistics: {e}")
