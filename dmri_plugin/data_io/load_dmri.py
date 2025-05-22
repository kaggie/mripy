import numpy as np
import nibabel as nib # Assuming nibabel is installed
import os # For __main__ test block

def load_nifti_dmri_data(nifti_filepath, bval_filepath, bvec_filepath):
    """
    Loads dMRI data from NIfTI image file, and corresponding b-values and b-vectors
    from text files.

    Args:
        nifti_filepath (str): Path to the NIfTI image file (usually 4D).
        bval_filepath (str): Path to the .bval file (text file with b-values).
        bvec_filepath (str): Path to the .bvec file (text file with b-vectors).

    Returns:
        tuple: Contains:
            - image_data (np.ndarray): 4D NumPy array (x, y, z, num_gradients_or_b0s).
            - b_values (np.ndarray): 1D NumPy array of b-values.
            - b_vectors (np.ndarray): 2D NumPy array of b-vectors (N, 3).
        Returns None, None, None if any error occurs during loading or validation.
    """
    try:
        # Load NIfTI image
        print(f"Loading NIfTI image from: {nifti_filepath}")
        img = nib.load(nifti_filepath)
        image_data = img.get_fdata(dtype=np.float32) # Load as float32 for consistency

        if image_data.ndim != 4:
            print(f"Error: NIfTI image data at {nifti_filepath} is not 4D. Found {image_data.ndim} dimensions.")
            return None, None, None

        num_volumes = image_data.shape[3]
        print(f"NIfTI data loaded: shape={image_data.shape}")

        # Load b-values
        print(f"Loading b-values from: {bval_filepath}")
        # b-values are typically space or comma separated in a single row or column
        b_values_str = np.loadtxt(bval_filepath, ndmin=1) # ndmin=1 to handle single value case
        b_values = b_values_str.astype(np.float32)
        if b_values.ndim > 1: # If loaded as multiple rows/cols, try to flatten
             b_values = b_values.flatten()
        print(f"b-values loaded: count={len(b_values)}")


        # Load b-vectors
        print(f"Loading b-vectors from: {bvec_filepath}")
        # b-vectors are typically 3xN (3 rows, N columns) or Nx3. Need to ensure (N, 3)
        b_vectors_raw = np.loadtxt(bvec_filepath)
        if b_vectors_raw.shape[0] == 3 and b_vectors_raw.shape[1] == num_volumes:
            b_vectors = b_vectors_raw.T.astype(np.float32)  # Transpose to get (N, 3)
            print(f"b-vectors loaded and transposed: shape={b_vectors.shape}")
        elif b_vectors_raw.shape[0] == num_volumes and b_vectors_raw.shape[1] == 3:
            b_vectors = b_vectors_raw.astype(np.float32)
            print(f"b-vectors loaded: shape={b_vectors.shape}")
        else:
            print(f"Error: b-vectors file {bvec_filepath} has unexpected shape {b_vectors_raw.shape}. Expected 3x{num_volumes} or {num_volumes}x3.")
            return None, None, None

        # Validation
        if len(b_values) != num_volumes:
            print(f"Error: Number of b-values ({len(b_values)}) does not match number of image volumes ({num_volumes}).")
            return None, None, None
        if b_vectors.shape[0] != num_volumes:
            print(f"Error: Number of b-vectors ({b_vectors.shape[0]}) does not match number of image volumes ({num_volumes}).")
            return None, None, None
        if b_vectors.shape[1] != 3: # Should be caught by the shape check above, but good for clarity
            print(f"Error: b-vectors do not have 3 components (shape is {b_vectors.shape}).")
            return None, None, None

        print("Successfully loaded and validated dMRI data, b-values, and b-vectors.")
        return image_data, b_values, b_vectors

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during dMRI data loading: {e}")
        return None, None, None

if __name__ == '__main__':
    # Create dummy files for testing
    def create_dummy_dmri_data(base_name="dummy_dwi", shape=(10,10,10,5), bval_count=5, bvec_dims=3):
        # NIfTI
        nifti_data = np.random.rand(*shape).astype(np.float32)
        nifti_img = nib.Nifti1Image(nifti_data, np.eye(4))
        nib.save(nifti_img, f"{base_name}.nii.gz")
        print(f"Created dummy NIfTI: {base_name}.nii.gz")

        # bval
        b_vals = np.random.randint(0, 2, bval_count) * 1000 # Example b0 and b1000
        b_vals[0] = 0 # Ensure at least one b0
        np.savetxt(f"{base_name}.bval", b_vals, fmt='%d', newline=' ')
        print(f"Created dummy bval: {base_name}.bval with values: {b_vals}")

        # bvec
        b_vecs = np.random.rand(bvec_dims, bval_count)
        b_vecs[:, b_vals == 0] = 0 # Zero vectors for b0 images
        
        # Normalize non-zero vectors
        for i in range(bval_count):
            if b_vals[i] != 0:
                norm = np.linalg.norm(b_vecs[:, i])
                if norm > 1e-6: # Avoid division by zero for nearly zero vectors
                    b_vecs[:, i] = b_vecs[:, i] / norm
                else: # If a non-b0 vector is accidentally zero, set a default (e.g. [1,0,0])
                    b_vecs[:,i] = np.array([1.0, 0.0, 0.0]) if bvec_dims == 3 else np.zeros(bvec_dims)
                    b_vecs[0,i] = 1.0


        np.savetxt(f"{base_name}.bvec", b_vecs, fmt='%.8f')
        print(f"Created dummy bvec: {base_name}.bvec")
        return f"{base_name}.nii.gz", f"{base_name}.bval", f"{base_name}.bvec"

    print("\n--- Testing dMRI data loading function ---")
    
    # Test Case 1: Ideal case
    print("\n--- Test Case 1: Ideal Case ---")
    nii_path1, bval_path1, bvec_path1 = create_dummy_dmri_data("test_dwi1", shape=(3,3,3,5), bval_count=5)
    data1, bvals1, bvecs1 = load_nifti_dmri_data(nii_path1, bval_path1, bvec_path1)
    if data1 is not None:
        print(f"Test Case 1 Success: Loaded data shape: {data1.shape}, bvals len: {len(bvals1)}, bvecs shape: {bvecs1.shape}")
        assert data1.shape == (3,3,3,5)
        assert len(bvals1) == 5
        assert bvecs1.shape == (5,3)
    else:
        print("Test Case 1 Failed.")

    # Test Case 2: bvecs shape needs transpose (bvecs are 5x3 initially)
    print("\n--- Test Case 2: bvecs shape 5x3 (num_volumes x 3) ---")
    nii_path2, bval_path2, _ = create_dummy_dmri_data("test_dwi2", shape=(4,4,4,6), bval_count=6)
    # Create a bvec file with shape (num_volumes, 3)
    b_vecs_alt_data = np.random.rand(6, 3) # 6 volumes, 3 components
    b_vals_temp = np.loadtxt(bval_path2) # Load bvals to zero out corresponding b0 vectors
    b_vecs_alt_data[b_vals_temp == 0, :] = 0
    for i in range(6): # Normalize
        if b_vals_temp[i] != 0:
            norm = np.linalg.norm(b_vecs_alt_data[i,:])
            if norm > 1e-6: b_vecs_alt_data[i,:] /= norm
            else: b_vecs_alt_data[i,:] = np.array([1.0,0.0,0.0])
    
    np.savetxt("test_dwi2.bvec", b_vecs_alt_data, fmt='%.8f') # Save as (N,3)
    print(f"Created bvec for Test Case 2: test_dwi2.bvec with shape {b_vecs_alt_data.shape}")
    
    data2, bvals2, bvecs2 = load_nifti_dmri_data(nii_path2, bval_path2, "test_dwi2.bvec")
    if data2 is not None:
        print(f"Test Case 2 Success (bvec shape {b_vecs_alt_data.shape} handled): Loaded data shape: {data2.shape}, bvecs shape: {bvecs2.shape}")
        assert data2.shape == (4,4,4,6)
        assert bvecs2.shape == (6,3)
    else:
        print("Test Case 2 Failed.")
        
    # Test Case 3: Mismatch bval count
    print("\n--- Test Case 3: Mismatch bval count ---")
    nii_path3, _, bvec_path3 = create_dummy_dmri_data("test_dwi3", shape=(2,2,2,4), bval_count=4) # NIfTI has 4 volumes
    np.savetxt("test_dwi3.bval", np.array([0, 1000, 1000]), fmt='%d', newline=' ') # Only 3 bvals
    print(f"Created bval for Test Case 3: test_dwi3.bval with 3 values")
    data3, bvals3, bvecs3 = load_nifti_dmri_data(nii_path3, "test_dwi3.bval", bvec_path3)
    if data3 is None:
        print("Test Case 3 Success (Mismatch bval count handled).")
    else:
        print(f"Test Case 3 Failed. Loaded data shape: {data3.shape if data3 is not None else 'None'}")

    # Test Case 4: Non-4D NIfTI
    print("\n--- Test Case 4: Non-4D NIfTI ---")
    nifti_3d_data = np.random.rand(3,3,3).astype(np.float32)
    nifti_3d_img = nib.Nifti1Image(nifti_3d_data, np.eye(4))
    nib.save(nifti_3d_img, "test_dwi4.nii.gz")
    print(f"Created 3D NIfTI for Test Case 4: test_dwi4.nii.gz")
    # Create dummy bval/bvec that would match if NIfTI was, say, 1 volume (but it's 3D)
    _, bval_path4, bvec_path4 = create_dummy_dmri_data("test_dwi4_meta", shape=(3,3,3,1), bval_count=1) 
    data4, bvals4, bvecs4 = load_nifti_dmri_data("test_dwi4.nii.gz", bval_path4, bvec_path4)
    if data4 is None:
        print("Test Case 4 Success (Non-4D NIfTI handled).")
    else:
        print(f"Test Case 4 Failed. Loaded data shape: {data4.shape if data4 is not None else 'None'}")
        
    # Cleanup dummy files
    print("\nCleaning up dummy files...")
    test_files = [
        "test_dwi1.nii.gz", "test_dwi1.bval", "test_dwi1.bvec",
        "test_dwi2.nii.gz", "test_dwi2.bval", "test_dwi2.bvec",
        "test_dwi3.nii.gz", "test_dwi3.bval", "test_dwi3.bvec",
        "test_dwi4.nii.gz", "test_dwi4_meta.nii.gz", "test_dwi4_meta.bval", "test_dwi4_meta.bvec"
    ]
    for f_path in test_files:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception as e_clean:
                print(f"Warning: Could not remove file {f_path}: {e_clean}")
    print("Dummy files cleanup attempt finished.")
    print("\n--- dMRI loader tests complete ---")
```
