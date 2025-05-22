import numpy as np
# Assumes dti.py is in dmri_plugin.models. Adjust import if structure is different.
try:
    from ..models.dti import calculate_D_elements, tensor_elements_to_matrix, calculate_dti_metrics
except ImportError: # Fallback for direct execution
    import sys
    import os
    # Add parent of dmri_plugin to path if this file is in dmri_plugin/fitting
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from dmri_plugin.models.dti import calculate_D_elements, tensor_elements_to_matrix, calculate_dti_metrics


def _construct_b_design_matrix(b_values_dw, b_vectors_dw):
    """
    Constructs the B design matrix for DTI fitting.
    Each row corresponds to a diffusion-weighted gradient direction i.
    B_i = [b_i*gx_i^2, b_i*gy_i^2, b_i*gz_i^2, 
           2*b_i*gx_i*gy_i, 2*b_i*gx_i*gz_i, 2*b_i*gy_i*gz_i]

    Args:
        b_values_dw (np.ndarray): 1D array of b-values (N,) for diffusion-weighted (b>0) measurements.
        b_vectors_dw (np.ndarray): 2D array of b-vectors (N, 3) for diffusion-weighted measurements,
                                   assumed to be unit vectors.

    Returns:
        np.ndarray: The B design matrix (N, 6), or None if inputs are invalid.
    """
    if not isinstance(b_values_dw, np.ndarray) or not isinstance(b_vectors_dw, np.ndarray):
        print("Error: b_values_dw and b_vectors_dw must be NumPy arrays in _construct_b_design_matrix.")
        return None
    if b_values_dw.ndim != 1 or b_vectors_dw.ndim != 2 or \
       b_values_dw.shape[0] != b_vectors_dw.shape[0] or b_vectors_dw.shape[1] != 3:
        print("Error: Invalid input dimensions for b_values_dw or b_vectors_dw in _construct_b_design_matrix.")
        print(f"  b_values_dw shape: {b_values_dw.shape}, expected (N,)")
        print(f"  b_vectors_dw shape: {b_vectors_dw.shape}, expected (N, 3)")
        return None
    if np.any(b_values_dw <= 0): # Design matrix is for b>0 data
        print("Error: b_values_dw should only contain values > 0 for design matrix construction.")
        return None


    N = len(b_values_dw)
    b_design_matrix = np.zeros((N, 6), dtype=np.float64) # Use float64 for precision

    gx = b_vectors_dw[:, 0]
    gy = b_vectors_dw[:, 1]
    gz = b_vectors_dw[:, 2]

    b_design_matrix[:, 0] = b_values_dw * gx**2
    b_design_matrix[:, 1] = b_values_dw * gy**2
    b_design_matrix[:, 2] = b_values_dw * gz**2
    b_design_matrix[:, 3] = 2 * b_values_dw * gx * gy
    b_design_matrix[:, 4] = 2 * b_values_dw * gx * gz
    b_design_matrix[:, 5] = 2 * b_values_dw * gy * gz
    
    return b_design_matrix


def fit_dti_voxel(voxel_signals_dw, S0_voxel, b_values_dw, b_vectors_dw, min_signal_threshold=1e-6):
    """
    Fits the DTI model to the diffusion signals from a single voxel.
    Assumes inputs (voxel_signals_dw, b_values_dw, b_vectors_dw) correspond to
    diffusion-weighted (b>0) measurements only.

    Args:
        voxel_signals_dw (np.ndarray): 1D array of diffusion-weighted signal intensities (S_i)
                                       for the voxel from N gradient directions (b>0).
        S0_voxel (float): Signal intensity for b=0 for the voxel.
        b_values_dw (np.ndarray): 1D array of b-values (N,) for the diffusion-weighted signals.
        b_vectors_dw (np.ndarray): 2D array of b-vectors (N, 3) for the diffusion-weighted signals.
        min_signal_threshold (float): Minimum signal value to use for S0 and S_dw to avoid log(0) or negative logs.

    Returns:
        tuple: (MD, FA, AD, RD, D_tensor_matrix (3x3), sorted_eigenvalues (3,), sorted_eigenvectors (3x3))
               Returns (None, ..., None) if fitting fails or input is invalid.
    """
    if S0_voxel <= min_signal_threshold: 
        # print(f"Warning: S0_voxel ({S0_voxel:.2e}) is at or below threshold ({min_signal_threshold:.1e}), cannot fit DTI reliably.")
        return None, None, None, None, None, None, None
    
    # Ensure no zero/negative signals in dw signals to avoid log errors
    processed_voxel_signals_dw = np.maximum(voxel_signals_dw, min_signal_threshold)
    
    # Calculate log signal ratio S_i / S0
    # If S_i > S0 (due to noise), log can be positive. This is usually okay for lstsq.
    log_S_ratio = np.log(processed_voxel_signals_dw / S0_voxel)
    
    # Handle potential -inf or nan if processed_voxel_signals_dw was still problematic relative to S0
    # For instance, if S0_voxel was also very small.
    log_S_ratio = np.nan_to_num(log_S_ratio, nan=0.0, posinf=0.0, neginf=-10.0) # Cap extreme neg values


    b_design_matrix = _construct_b_design_matrix(b_values_dw, b_vectors_dw)
    if b_design_matrix is None:
        # print("Warning: Failed to construct b_design_matrix in fit_dti_voxel.")
        return None, None, None, None, None, None, None

    D_elements = calculate_D_elements(log_S_ratio, b_design_matrix)
    if D_elements is None:
        # print("Warning: Failed to calculate D_elements in fit_dti_voxel.")
        return None, None, None, None, None, None, None

    D_tensor_matrix = tensor_elements_to_matrix(D_elements)
    if D_tensor_matrix is None:
        # print("Warning: Failed to convert D_elements to matrix in fit_dti_voxel.")
        return None, None, None, None, None, None, None
        
    MD, FA, AD, RD, sorted_eigenvalues, sorted_eigenvectors_matrix = calculate_dti_metrics(D_tensor_matrix)
    if MD is None: # Check if metrics calculation failed
        # print("Warning: Failed to calculate DTI metrics in fit_dti_voxel, but tensor might be valid.")
        return None, None, None, None, D_tensor_matrix, None, None # Still return tensor if that part worked

    return MD, FA, AD, RD, D_tensor_matrix, sorted_eigenvalues, sorted_eigenvectors_matrix


def fit_dti_volume(image_data_4d, b_values, b_vectors, b0_threshold=50.0, min_S0_intensity_threshold=1.0):
    """
    Fits the DTI model voxel-wise to a 4D dMRI dataset.

    Args:
        image_data_4d (np.ndarray): 4D dMRI data (x, y, z, num_gradients_or_b0s).
        b_values (np.ndarray): 1D array of all b-values.
        b_vectors (np.ndarray): 2D array of all b-vectors (N_total, 3).
        b0_threshold (float): b-values below or equal to this are considered b0 images.
        min_S0_intensity_threshold (float): Minimum S0 intensity for a voxel to be processed.
                                            Voxels below this (after S0 averaging) are skipped.

    Returns:
        dict: A dictionary of 3D parameter maps (e.g., "MD": md_map, "FA": fa_map, ...),
              and "D_tensor_map" (x,y,z,3,3). Values for skipped or failed voxels will be NaN.
              Returns None if critical errors occur (e.g., no b0 or dw images).
    """
    if not isinstance(image_data_4d, np.ndarray) or image_data_4d.ndim != 4:
        print("Error: image_data_4d must be a 4D NumPy array.")
        return None
    if not isinstance(b_values, np.ndarray) or b_values.ndim != 1 or len(b_values) != image_data_4d.shape[3]:
        print("Error: b_values must be a 1D NumPy array matching the 4th dim of image_data_4d.")
        return None
    if not isinstance(b_vectors, np.ndarray) or b_vectors.ndim != 2 or \
       b_vectors.shape[0] != image_data_4d.shape[3] or b_vectors.shape[1] != 3:
        print("Error: b_vectors must be a 2D NumPy array (N_total, 3) matching the 4th dim of image_data_4d.")
        return None


    spatial_dims = image_data_4d.shape[:3]
    md_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    fa_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    ad_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    rd_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    d_tensor_map_3x3 = np.full(spatial_dims + (3,3), np.nan, dtype=np.float32) 
    
    b0_indices = np.where(b_values <= b0_threshold)[0]
    dw_indices = np.where(b_values > b0_threshold)[0]

    if len(b0_indices) == 0:
        print("Error: No b0 images found based on b0_threshold. Cannot proceed with DTI fitting.")
        return None
    if len(dw_indices) == 0:
        print("Error: No diffusion-weighted images found based on b0_threshold. Cannot proceed.")
        return None

    # Average S0 images if multiple exist
    S0_volume = np.mean(image_data_4d[..., b0_indices], axis=3, dtype=np.float32)
    
    dw_signals_volume = image_data_4d[..., dw_indices].astype(np.float32)
    b_values_dw = b_values[dw_indices].astype(np.float32)
    b_vectors_dw = b_vectors[dw_indices, :].astype(np.float32)
    
    print(f"Starting DTI fitting for volume of shape {spatial_dims}...")
    total_voxels = np.prod(spatial_dims)
    processed_voxels = 0
    fitted_voxels_count = 0

    for x in range(spatial_dims[0]):
        for y in range(spatial_dims[1]):
            for z in range(spatial_dims[2]):
                S0_voxel = S0_volume[x, y, z]
                
                # Masking based on S0 intensity
                if S0_voxel < min_S0_intensity_threshold:
                    processed_voxels += 1
                    if total_voxels > 0 and processed_voxels % (total_voxels // 100 or 1) == 0:
                        print(f"DTI Fitting Progress: { (processed_voxels/total_voxels)*100 :.1f}%", end='\r')
                    continue 
                
                voxel_signals_dw = dw_signals_volume[x, y, z, :]
                
                MD, FA, AD, RD, D_matrix, _, _ = fit_dti_voxel(
                    voxel_signals_dw, S0_voxel, b_values_dw, b_vectors_dw
                )
                
                if MD is not None: # If fit was successful (fit_dti_voxel returns None for some metrics if only D_matrix is good)
                    md_map[x,y,z] = MD
                    fa_map[x,y,z] = FA if FA is not None else np.nan # Ensure FA is NaN if None
                    ad_map[x,y,z] = AD if AD is not None else np.nan
                    rd_map[x,y,z] = RD if RD is not None else np.nan
                    if D_matrix is not None:
                        d_tensor_map_3x3[x,y,z,:,:] = D_matrix
                    fitted_voxels_count +=1
                # If fit_dti_voxel returned Nones, maps retain their np.nan initialization.
                
                processed_voxels += 1
                if total_voxels > 0 and processed_voxels % (total_voxels // 100 or 1) == 0:
                    print(f"DTI Fitting Progress: { (processed_voxels/total_voxels)*100 :.1f}% ({fitted_voxels_count} fitted)", end='\r')
    
    print(f"\nDTI Fitting complete. Processed {processed_voxels} voxels, successfully fitted {fitted_voxels_count} voxels.")

    return {
        "MD": md_map, "FA": fa_map, "AD": ad_map, "RD": rd_map,
        "D_tensor_map": d_tensor_map_3x3
    }

if __name__ == '__main__':
    print("Testing DTI fitter functions...")
    # Create some dummy 4D data, bvals, bvecs
    shape_4d = (5, 5, 2, 10) # Small volume for quick test: x,y,z,gradients
    # Ensure data is positive and has some structure for S0 vs DWI
    base_signal = np.random.rand(*shape_4d[:3]) * 500 + 100 # Base S0 signal per voxel
    img_data = np.zeros(shape_4d, dtype=np.float32)
    
    bvals = np.zeros(shape_4d[3], dtype=np.float32)
    bvals[0] = 0 # One b0 image
    bvals[1:] = 1000 # 9 DWIs with b=1000
    
    bvecs = np.random.rand(shape_4d[3], 3).astype(np.float32)
    bvecs[0,:] = 0 # b0 has zero vector
    
    for i in range(1, shape_4d[3]): # Normalize non-b0 bvecs
        norm = np.linalg.norm(bvecs[i,:])
        if norm > 1e-6: bvecs[i,:] /= norm
        else: bvecs[i,:] = np.array([1.0, 0.0, 0.0])

    # Simulate signal decay for DWIs
    # S_i = S0 * exp(-b_i * D_apparent_i)
    # For simplicity, let's use a fixed D_apparent for simulation
    D_sim = 0.7e-3 
    for x_ in range(shape_4d[0]):
        for y_ in range(shape_4d[1]):
            for z_ in range(shape_4d[2]):
                s0_val = base_signal[x_,y_,z_]
                img_data[x_,y_,z_,0] = s0_val # b0 image
                for grad_idx in range(1, shape_4d[3]):
                    # Simplified decay based on b-value and one component of b-vector
                    # True DTI decay is S0 * exp(- (b_val * (gx^2 Dxx + gy^2 Dyy + ...)))
                    # This is just to get some non-random DWI signals for testing structure
                    decay_factor = np.exp(-bvals[grad_idx] * D_sim * (bvecs[grad_idx,0]**2 + 0.5*bvecs[grad_idx,1]**2 + 0.2*bvecs[grad_idx,2]**2 ))
                    img_data[x_,y_,z_,grad_idx] = s0_val * decay_factor + np.random.rand()*10 # Add some noise


    print(f"\n--- Test _construct_b_design_matrix ---")
    dw_indices_test = np.where(bvals > 50)[0]
    bvals_dw_test = bvals[dw_indices_test]
    bvecs_dw_test = bvecs[dw_indices_test,:]
    design_matrix_test = _construct_b_design_matrix(bvals_dw_test, bvecs_dw_test)
    if design_matrix_test is not None:
        print(f"Constructed design matrix shape: {design_matrix_test.shape}")
        assert design_matrix_test.shape == (len(dw_indices_test), 6)
        print("_construct_b_design_matrix test PASSED.")
    else:
        print("_construct_b_design_matrix test FAILED.")

    print(f"\n--- Test fit_dti_voxel (voxel 0,0,0) ---")
    S0_voxel_test = np.mean(img_data[0,0,0, np.where(bvals <= 50)[0]])
    voxel_signals_dw_test = img_data[0,0,0, dw_indices_test]
    
    print(f"S0 for voxel (0,0,0): {S0_voxel_test:.2f}")
    # print(f"DW signals for voxel (0,0,0): {voxel_signals_dw_test}")
    # print(f"b-values for DWIs: {bvals_dw_test}")
    # print(f"b-vectors for DWIs (first 3):\n{bvecs_dw_test[:3,:]}")

    MD_v, FA_v, AD_v, RD_v, Dmat_v, evals_v, evecs_v = fit_dti_voxel(
        voxel_signals_dw_test, S0_voxel_test, bvals_dw_test, bvecs_dw_test
    )
    if MD_v is not None:
        print(f"Voxel fit results: MD={MD_v:.3e}, FA={FA_v:.3f}, AD={AD_v:.3e}, RD={RD_v:.3e}")
        # print(f"Tensor:\n{Dmat_v}")
        print("fit_dti_voxel test PASSED (execution).")
    else:
        print("fit_dti_voxel test FAILED (MD is None).")

    print(f"\n--- Test fit_dti_volume ---")
    dti_maps = fit_dti_volume(img_data, bvals, bvecs, b0_threshold=50, min_S0_intensity_threshold=50.0)
    if dti_maps is not None:
        print(f"Volume fitting complete. Got maps: {list(dti_maps.keys())}")
        print(f"MD map shape: {dti_maps['MD'].shape}, FA map non-NaNs: {np.sum(~np.isnan(dti_maps['FA']))}")
        assert dti_maps['MD'].shape == shape_4d[:3]
        assert dti_maps['D_tensor_map'].shape == shape_4d[:3] + (3,3)
        
        # Check if the first voxel's result from volume fitting matches single voxel fit
        # Need to handle NaN for voxels that might have been skipped or failed
        if MD_v is not None and not np.isnan(dti_maps['MD'][0,0,0]):
             assert np.allclose(dti_maps['MD'][0,0,0], MD_v), \
                f"MD map value {dti_maps['MD'][0,0,0]} does not match single voxel fit {MD_v}"
        elif np.isnan(MD_v) != np.isnan(dti_maps['MD'][0,0,0]): # One is NaN, other is not
            print(f"Warning: MD for voxel (0,0,0) mismatch: map is {dti_maps['MD'][0,0,0]}, single fit is {MD_v}")


        print("fit_dti_volume test PASSED (execution and basic checks).")
    else:
        print("fit_dti_volume test FAILED (returned None).")
    
    print("\nAll DTI fitter function tests finished.")

```
