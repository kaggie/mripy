import os
import nibabel as nib
import numpy as np

def load_nifti_file(filepath: str):
    """
    Loads a NIfTI file.

    Args:
        filepath (str): Path to the NIfTI file.

    Returns:
        nibabel.nifti1.Nifti1Image: The loaded NIfTI image object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid NIfTI file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI file not found at: {filepath}")
    try:
        img = nib.load(filepath)
        return img
    except Exception as e:
        raise ValueError(f"Invalid NIfTI file: {filepath}. Error: {e}")

def load_dce_series(filepath: str):
    """
    Loads a 4D DCE NIfTI series.

    Args:
        filepath (str): Path to the 4D NIfTI file.

    Returns:
        np.ndarray: The DCE series data as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 4D NIfTI file.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 4:
        raise ValueError("DCE series must be a 4D NIfTI image.")
    return img.get_fdata()

def load_t1_map(filepath: str, dce_shape: tuple = None):
    """
    Loads a 3D T1 map NIfTI file.

    Args:
        filepath (str): Path to the 3D NIfTI file.
        dce_shape (tuple, optional): Shape of the corresponding DCE series 
                                     (x, y, z, time) for dimension validation. 
                                     Defaults to None.

    Returns:
        np.ndarray: The T1 map data as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 3D NIfTI file or if dimensions 
                    do not match dce_shape.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 3:
        raise ValueError("T1 map must be a 3D NIfTI image.")
    if dce_shape is not None:
        if img.shape != dce_shape[:3]:
            raise ValueError(
                "T1 map dimensions do not match DCE series spatial dimensions."
            )
    return img.get_fdata()

def load_mask(filepath: str, reference_shape: tuple = None):
    """
    Loads a 3D mask NIfTI file and converts it to a boolean array.

    Args:
        filepath (str): Path to the 3D NIfTI mask file.
        reference_shape (tuple, optional): Spatial shape (x, y, z) of the 
                                           reference image (DCE or T1 map) for
                                           dimension validation. Defaults to None.

    Returns:
        np.ndarray: The mask data as a boolean NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 3D NIfTI file or if dimensions
                    do not match reference_shape.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 3:
        raise ValueError("Mask must be a 3D NIfTI image.")
    if reference_shape is not None:
        if img.shape != reference_shape:
            raise ValueError(
                "Mask dimensions do not match the reference image dimensions."
            )
    data = img.get_fdata()
    return data.astype(bool)

def save_nifti_map(data_map: np.ndarray, original_nifti_ref_path: str, output_filepath: str):
    """
    Saves a 3D data map as a NIfTI file, using an original NIfTI file for affine and header.

    Args:
        data_map (np.ndarray): The 3D NumPy array containing the parameter map data.
        original_nifti_ref_path (str): Path to an original NIfTI file (e.g., T1 map or
                                       one frame of DCE) to source affine and header.
        output_filepath (str): The path where the new NIfTI map file will be saved.

    Raises:
        FileNotFoundError: If the original_nifti_ref_path does not exist.
        Exception: Can re-raise exceptions from nibabel.load or nibabel.save.
    """
    if not os.path.exists(original_nifti_ref_path):
        raise FileNotFoundError(f"Reference NIfTI file not found at: {original_nifti_ref_path}")

    ref_nifti_img = nib.load(original_nifti_ref_path)

    # Ensure data_map is 3D and matches spatial dimensions of reference
    if data_map.ndim != 3:
        raise ValueError(f"data_map must be a 3D array. Got {data_map.ndim} dimensions.")
    if data_map.shape != ref_nifti_img.shape[:3]: # Compare with spatial part of ref shape
        raise ValueError(f"data_map shape {data_map.shape} does not match "
                         f"reference NIfTI spatial shape {ref_nifti_img.shape[:3]}.")

    # Create new NIfTI image
    # Use float32 for parameter maps typically.
    # Copy header from original and update data type information if necessary.
    new_header = ref_nifti_img.header.copy()
    new_header.set_data_dtype(np.float32) 
    # If original was 4D, ensure the new 3D map's header reflects 3D
    if len(ref_nifti_img.shape) == 4 and len(data_map.shape) == 3:
        new_header.set_data_shape(data_map.shape) # Update shape in header for 3D
        # Remove 4th dimension related info if any (e.g. pixdim[4], dim[4])
        # This is often handled by nibabel when creating Nifti1Image with 3D data
        # but explicitly setting shape in header is good practice.
        # new_header['dim'][0] = 3 # Number of dimensions
        # new_header['dim'][4] = 1 # Size of 4th dim
        # new_header['pixdim'][4] = 0 # Voxel size for 4th dim (or 1 if preferred)

    new_nifti_image = nib.Nifti1Image(data_map.astype(np.float32), ref_nifti_img.affine, header=new_header)
    
    nib.save(new_nifti_image, output_filepath)
    print(f"NIfTI map saved to: {output_filepath}") # For CLI confirmation, GUI will use log console
