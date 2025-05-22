import numpy as np
import os # For os.path.exists if needed, though try-except is better for opening

def load_text_mrsi_data(data_filepath, metadata_filepath=None):
    """
    Loads MRSI spectral data and metadata from text files.

    The spectral data is expected to be in a text file (e.g., CSV) where
    each column represents the spectrum for a single voxel and each row
    represents a spectral point.

    The metadata file (if provided) is expected to be a simple key-value
    text file, with each line formatted as 'key: value'.

    Args:
        data_filepath (str): Path to the text file containing spectral data.
        metadata_filepath (str, optional): Path to the text file containing
                                           metadata. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - spectral_data (numpy.ndarray): The loaded spectral data, shaped
                                             as (num_voxels, num_spectral_points).
                                             Returns None if loading fails.
            - metadata (dict): A dictionary containing the loaded metadata.
                               Returns an empty dict if no metadata file is
                               provided or if loading fails.

    Raises:
        FileNotFoundError: If the data_filepath does not exist.
                           (Metadata file not found is handled gracefully by
                            returning empty metadata).
        ValueError: If data parsing or metadata value conversion fails.
    """
    spectral_data = None
    metadata = {}

    # --- Load Spectral Data ---
    try:
        # Assuming data is points x voxels, so delimiter is usually comma or whitespace
        # For CSV: delimiter=','
        # For space-separated: delimiter=None or ' '
        # For tab-separated: delimiter='\t'
        # Using genfromtxt for more robustness with missing values if any, though loadtxt is fine.
        raw_data = np.loadtxt(data_filepath, delimiter=',')
        if raw_data.ndim == 0: # Single value file
            raise ValueError("Data file seems to contain only a single value.")
        elif raw_data.ndim == 1: # Single spectrum (interpreted as points) or single point for many voxels
            # Assuming it's a single spectrum: (points,) -> (1, points)
            spectral_data = raw_data.reshape(1, -1)
            print(f"Loaded data shape: {raw_data.shape}. Reshaped to: {spectral_data.shape}")
        elif raw_data.ndim == 2: # Expected: (points, voxels)
            # Transpose to get (voxels, points)
            spectral_data = raw_data.T
            print(f"Loaded data shape: {raw_data.shape}. Transposed to: {spectral_data.shape}")
        else:
            raise ValueError(f"Loaded data has unexpected number of dimensions: {raw_data.ndim}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_filepath}")
        raise
    except Exception as e:
        print(f"Error loading or processing spectral data from {data_filepath}: {e}")
        raise ValueError(f"Could not load data from {data_filepath}: {e}")


    # --- Load Metadata ---
    if metadata_filepath:
        try:
            with open(metadata_filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert known numeric fields
                    if key in ['SpectralWidth (Hz)', 'EchoTime (ms)', 'NumberOfPoints']:
                        try:
                            metadata[key] = int(value) if value.isdigit() else float(value)
                        except ValueError:
                            print(f"Warning: Could not convert metadata value '{value}' for key '{key}' to number. Storing as string.")
                            metadata[key] = value
                    elif key == 'VoxelSize (mm)': # Keep as string, or parse further if needed
                        metadata[key] = value
                    elif key == 'OriginalShape': # Keep as string, will be parsed later if needed
                        metadata[key] = value
                    else:
                        metadata[key] = value
            print(f"Loaded metadata from {metadata_filepath}")
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {metadata_filepath}. Continuing without metadata.")
        except Exception as e:
            print(f"Error loading metadata from {metadata_filepath}: {e}")
            # Keep metadata dict as is (potentially partially filled or empty)

    # --- Reshape data based on metadata if OriginalShape is present ---
    if spectral_data is not None and 'OriginalShape' in metadata:
        original_shape_str = metadata.get('OriginalShape')
        try:
            # Example: "8x8x1x1024"
            shape_dims = [int(d) for d in original_shape_str.split('x')]
            if not shape_dims:
                raise ValueError("OriginalShape string is empty or invalid.")

            num_points_meta = shape_dims[-1]
            num_voxels_meta = np.prod(shape_dims[:-1])

            # Current spectral_data is (presumed_voxels, presumed_points) from raw_data.T
            # or (1, points) if raw_data was 1D.
            
            # If raw_data was (points, voxels)
            # Example: raw_data (1024, 64) -> current spectral_data (64, 1024)
            # metadata 'OriginalShape': "8x8x1x1024" -> num_voxels_meta=64, num_points_meta=1024
            # This matches current shape (64, 1024) so no reshape needed
            # but good to validate.

            if spectral_data.shape[0] == num_voxels_meta and spectral_data.shape[1] == num_points_meta:
                print(f"Data shape {spectral_data.shape} matches OriginalShape {shape_dims} from metadata.")
            # This case handles if raw_data was (voxels, points) initially
            elif spectral_data.shape[1] == num_voxels_meta and spectral_data.shape[0] == num_points_meta:
                 print(f"Data shape {spectral_data.shape} seems to be (points, voxels) based on OriginalShape {shape_dims}. Transposing.")
                 spectral_data = spectral_data.T # Transpose to (voxels, points)
            # This case handles if data was loaded flat and needs full reshape
            elif spectral_data.size == num_voxels_meta * num_points_meta:
                print(f"Reshaping flat data of size {spectral_data.size} to ({num_voxels_meta}, {num_points_meta}) based on OriginalShape.")
                spectral_data = spectral_data.reshape(num_voxels_meta, num_points_meta)
            else:
                print(f"Warning: Loaded data shape {spectral_data.shape} (total elements {spectral_data.size}) "
                      f"does not conform to OriginalShape {original_shape_str} "
                      f"(expected {num_voxels_meta} voxels, {num_points_meta} points, total {num_voxels_meta*num_points_meta}). "
                      "Using loaded shape and metadata as is.")

        except Exception as e:
            print(f"Warning: Could not parse or apply OriginalShape metadata '{original_shape_str}': {e}")
            
    # --- Validate Data Dimensions (Optional but Recommended) ---
    if spectral_data is not None and metadata:
        if 'NumberOfPoints' in metadata:
            num_points_meta = metadata['NumberOfPoints']
            if spectral_data.shape[1] != num_points_meta:
                print(f"Warning: Number of points in data ({spectral_data.shape[1]}) "
                      f"does not match NumberOfPoints in metadata ({num_points_meta}).")
        
        # More complex validation if OriginalShape was used for reshaping
        if 'OriginalShape' in metadata and 'shape_dims' in locals(): # check if shape_dims was defined
            num_voxels_meta = np.prod(shape_dims[:-1])
            if spectral_data.shape[0] != num_voxels_meta:
                 print(f"Warning: Number of voxels in data ({spectral_data.shape[0]}) "
                       f"does not match calculated voxels from OriginalShape in metadata ({num_voxels_meta}).")

    if spectral_data is None and not metadata: # If both failed or data file wasn't even found.
        print("Load_text_mrsi_data: Failed to load any data or metadata.")
        # Depending on strictness, could return (None, {}) or raise an error earlier.
        # Current implementation will raise FileNotFoundError for data_filepath.
        # If data_filepath is found but data is malformed, spectral_data might be None.

    return spectral_data, metadata

if __name__ == '__main__':
    # Create dummy files for testing
    # This part is for local testing and won't be part of the actual plugin code normally.
    
    # Test Case 1: Data (points x voxels), Metadata present
    print("\\n--- Test Case 1: data_points_x_voxels.csv ---")
    dummy_data_pv_content = "10.1,12.3,11.5\\n15.2,17.8,16.1\\n20.0,22.0,21.0\\n25.5,27.5,26.5" # 4 points, 3 voxels
    with open("data_points_x_voxels.csv", "w") as f:
        f.write(dummy_data_pv_content.replace("\\\\n", "\\n"))
    
    dummy_metadata_content = "SpectralWidth (Hz): 2000\\nEchoTime (ms): 30\\nNumberOfPoints: 4\\nOriginalShape: 1x3x4"
    with open("metadata1.txt", "w") as f:
        f.write(dummy_metadata_content.replace("\\\\n", "\\n"))

    data1, meta1 = load_text_mrsi_data("data_points_x_voxels.csv", "metadata1.txt")
    if data1 is not None:
        print("Data1 shape:", data1.shape) # Expected: (3, 4)
        print("Data1 content:\\n", data1)
    print("Meta1:", meta1)
    print("-------------------------------------------\\n")

    # Test Case 2: Data (voxels x points), Metadata present, OriginalShape matches direct interpretation
    print("--- Test Case 2: data_voxels_x_points.csv ---")
    dummy_data_vp_content = "10,11,12,13\\n20,21,22,23\\n30,31,32,33" # 3 voxels, 4 points
    with open("data_voxels_x_points.csv", "w") as f:
        f.write(dummy_data_vp_content.replace("\\\\n", "\\n"))
    
    dummy_metadata_content2 = "SpectralWidth (Hz): 2000\\nEchoTime (ms): 30\\nNumberOfPoints: 4\\nOriginalShape: 3x1x4" # implies 3 voxels
    with open("metadata2.txt", "w") as f:
        f.write(dummy_metadata_content2.replace("\\\\n", "\\n"))

    # To make this test case work as intended with current logic (where loadtxt gives points x voxels)
    # the data file should actually be points x voxels, and then OriginalShape guides the final (voxels, points)
    # If data is truly voxels x points, loadtxt might interpret it as (voxels, points) directly.
    # Let's assume loadtxt gives (rows as read, cols as read).
    # If data_voxels_x_points.csv is (3,4), loadtxt reads it as (3,4).
    # Initial transpose makes it (4,3).
    # OriginalShape 3x1x4 -> 3 voxels, 4 points.
    # Reshaping logic: spectral_data.shape[0] (4) != num_voxels_meta (3)
    #                  spectral_data.shape[1] (3) != num_points_meta (4)
    #                  spectral_data.size (12) == num_voxels_meta * num_points_meta (12) -> reshape(3,4)
    # This scenario will be hit.

    data2, meta2 = load_text_mrsi_data("data_voxels_x_points.csv", "metadata2.txt")
    if data2 is not None:
        print("Data2 shape:", data2.shape) # Expected: (3, 4)
        print("Data2 content:\\n", data2)
    print("Meta2:", meta2)
    print("-------------------------------------------\\n")


    # Test Case 3: Single spectrum data (1D array from loadtxt)
    print("--- Test Case 3: single_spectrum.csv ---")
    dummy_single_spec_content = "10.1,15.2,20.0,25.5" # 1 spectrum, 4 points (read as a single line)
    with open("single_spectrum.csv", "w") as f:
        f.write(dummy_single_spec_content) # loadtxt will read this as 1D array if it's one line of CSV
    
    dummy_metadata3_content = "SpectralWidth (Hz): 1000\\nNumberOfPoints: 4\\nOriginalShape: 1x1x4"
    with open("metadata3.txt", "w") as f:
        f.write(dummy_metadata3_content.replace("\\\\n", "\\n"))
        
    data3, meta3 = load_text_mrsi_data("single_spectrum.csv", "metadata3.txt")
    if data3 is not None:
        print("Data3 shape:", data3.shape) # Expected: (1, 4)
        print("Data3 content:\\n", data3)
    print("Meta3:", meta3)
    print("-------------------------------------------\\n")

    # Test Case 4: Data file only, no metadata
    print("--- Test Case 4: data_only.csv ---")
    with open("data_only.csv", "w") as f: # Same as data_points_x_voxels.csv
        f.write(dummy_data_pv_content.replace("\\\\n", "\\n"))
    data4, meta4 = load_text_mrsi_data("data_only.csv")
    if data4 is not None:
        print("Data4 shape:", data4.shape) # Expected (3,4) after transpose of (4,3)
        print("Data4 content:\\n", data4)
    print("Meta4:", meta4) # Expected: {}
    print("-------------------------------------------\\n")

    # Test Case 5: Malformed metadata
    print("--- Test Case 5: Malformed metadata ---")
    malformed_meta_content = "SpectralWidth (Hz) 2000\\nInvalidLine\\nEchoTime (ms): thirty"
    with open("metadata_malformed.txt", "w") as f:
        f.write(malformed_meta_content.replace("\\\\n", "\\n"))
    data5, meta5 = load_text_mrsi_data("data_points_x_voxels.csv", "metadata_malformed.txt")
    if data5 is not None:
        print("Data5 shape:", data5.shape)
    print("Meta5:", meta5) # Check warnings and how data is stored
    print("-------------------------------------------\\n")

    # Test Case 6: Data file not found
    print("--- Test Case 6: Data file not found ---")
    try:
        load_text_mrsi_data("non_existent_data.csv")
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")
    print("-------------------------------------------\\n")

    # Test Case 7: Data file with shape mismatch vs OriginalShape
    print("--- Test Case 7: Data with shape mismatch vs OriginalShape ---")
    # data_points_x_voxels.csv is (4 points, 3 voxels)
    # metadata_shape_mismatch.txt will claim OriginalShape: 2x2x4 (4 voxels, 4 points)
    # This should trigger a warning as 3 voxels != 4 voxels
    mismatch_meta_content = "SpectralWidth (Hz): 2000\\nNumberOfPoints: 4\\nOriginalShape: 2x2x4"
    with open("metadata_shape_mismatch.txt", "w") as f:
        f.write(mismatch_meta_content.replace("\\\\n", "\\n"))
    data7, meta7 = load_text_mrsi_data("data_points_x_voxels.csv", "metadata_shape_mismatch.txt")
    if data7 is not None:
        print("Data7 shape:", data7.shape) # Should be (3,4) as loaded, with a warning
    print("Meta7:", meta7)
    print("-------------------------------------------\\n")
    
    # Cleanup dummy files
    files_to_delete = [
        "data_points_x_voxels.csv", "metadata1.txt",
        "data_voxels_x_points.csv", "metadata2.txt",
        "single_spectrum.csv", "metadata3.txt",
        "data_only.csv", "metadata_malformed.txt",
        "metadata_shape_mismatch.txt"
    ]
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            os.remove(f_path)
    print("Dummy files cleaned up.")
