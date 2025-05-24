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
        return {"N": 0, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, 
                "Median": np.nan, "Min": np.nan, "Max": np.nan}
    
    stats = {
        "N": roi_values.size, 
        "N_valid": np.sum(~np.isnan(roi_values)), 
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
    if stats_dict is None or stats_dict.get("N_valid", 0) == 0:
        return f"No valid data in {roi_name} for '{map_name}'."

    lines = [f"Statistics for {roi_name} on '{map_name}':"]
    for key, value in stats_dict.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)

def save_multiple_roi_statistics_csv(
    stats_results_list: list[tuple[str, int, str, dict]], 
    filepath: str
    ):
    '''
    Saves statistics for multiple ROIs to a single CSV file.
    Args:
        stats_results_list: A list of tuples, where each tuple is
                            (map_name, slice_index, roi_name, stats_dict).
        filepath: Path to the CSV file to save.
    '''
    if not stats_results_list:
        print("No statistics provided to save_multiple_roi_statistics_csv.")
        # Create an empty file with headers if desired, or just return
        # For now, just return to avoid empty file creation without explicit need
        return

    # Define fieldnames based on typical stats_dict keys plus context
    # Assuming all stats_dicts in the list have the same keys (N, Mean, StdDev, etc.)
    # Find the first valid stats_dict to determine keys
    first_valid_stats_dict = None
    for _, _, _, s_dict in stats_results_list:
        if s_dict and s_dict.get("N_valid", 0) > 0 : # Check N_valid for actual content
            first_valid_stats_dict = s_dict
            break
    
    if not first_valid_stats_dict: 
         # If no ROI has valid stats, use default keys or write an empty file.
         # For now, if all ROIs are empty/NaN, the output file will just have headers.
         stat_keys = ["N", "N_valid", "Mean", "StdDev", "Median", "Min", "Max"] # Default/expected keys
    else:
         stat_keys = list(first_valid_stats_dict.keys())

    fieldnames = ['MapName', 'SliceIndex', 'ROIName'] + stat_keys

    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for map_name, slice_idx, roi_name, stats_dict_for_roi in stats_results_list:
                if stats_dict_for_roi: # Ensure there are stats to write
                    row_data = {
                        'MapName': map_name,
                        'SliceIndex': slice_idx,
                        'ROIName': roi_name
                    }
                    # Ensure all stat_keys are present in row_data, defaulting to 'N/A' or np.nan if missing in this specific dict
                    for skey in stat_keys:
                        row_data[skey] = stats_dict_for_roi.get(skey, np.nan) # or "N/A" if preferred for CSV

                    writer.writerow(row_data)
                else: # Write basic info even if stats are null/empty for this particular ROI
                    # Create a row with N/A for stat values
                    empty_stat_row = {'MapName': map_name, 'SliceIndex': slice_idx, 'ROIName': roi_name}
                    for skey in stat_keys:
                        empty_stat_row[skey] = "N/A" # Or np.nan, but "N/A" is clearer in CSV
                    writer.writerow(empty_stat_row)
    except IOError as e:
        raise IOError(f"Error writing ROI statistics to CSV file {filepath}: {e}")
    except Exception as e: 
        raise Exception(f"An unexpected error occurred while saving ROI statistics: {e}")

# Keep the old function for now, or mark as deprecated.
# For this task, we'll just leave it. If it's not used, it can be removed later.
def save_roi_statistics_csv(stats_dict: dict, filepath: str, map_name: str, roi_name: str = "ROI"):
    """
    Saves ROI statistics to a CSV file. (Old version for single ROI)
    """
    if not stats_dict:
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
