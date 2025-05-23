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
