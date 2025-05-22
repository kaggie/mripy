import numpy as np

def calculate_D_elements(log_S_ratio, b_design_matrix):
    """
    Estimates the 6 unique diffusion tensor elements using linear least squares.
    The model is: log(S_i / S0) = - (b_xx_i*D_xx + b_yy_i*D_yy + ... + 2*b_xy_i*D_xy + ...)
    Or, more simply, log(S_i / S0) = - B_i . D_vec
    where D_vec = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    and B_i = [b_xx_i, b_yy_i, b_zz_i, 2*b_xy_i, 2*b_xz_i, 2*b_yz_i] for each gradient direction i.

    Args:
        log_S_ratio (np.ndarray): 1D array of log(S_i / S0) for N gradient directions.
                                  Shape (N,).
        b_design_matrix (np.ndarray): Design matrix derived from b-values and b-vectors.
                                     Each row corresponds to a gradient direction.
                                     Shape (N, 6). Columns are typically ordered as
                                     [b_xx, b_yy, b_zz, 2*b_xy, 2*b_xz, 2*b_yz].

    Returns:
        np.ndarray: 1D array of the 6 fitted diffusion tensor elements (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz),
                    or None if fitting fails.
    """
    if not isinstance(log_S_ratio, np.ndarray) or not isinstance(b_design_matrix, np.ndarray):
        print("Error: Inputs log_S_ratio and b_design_matrix must be NumPy arrays.")
        return None
        
    if log_S_ratio.ndim != 1 or b_design_matrix.ndim != 2 or \
       log_S_ratio.shape[0] != b_design_matrix.shape[0] or \
       b_design_matrix.shape[1] != 6:
        print("Error: Input dimensions are incorrect for DTI fitting.")
        print(f"  log_S_ratio shape: {log_S_ratio.shape}, expected (N,)")
        print(f"  b_design_matrix shape: {b_design_matrix.shape}, expected (N, 6)")
        print(f"  N from log_S_ratio: {log_S_ratio.shape[0]}, N from b_design_matrix: {b_design_matrix.shape[0]}")
        return None
    
    if log_S_ratio.shape[0] < 6 : # Need at least 6 measurements for 6 unknowns
        print(f"Error: Insufficient measurements ({log_S_ratio.shape[0]}) for DTI fitting (need at least 6).")
        return None

    try:
        # We are solving B.D_vec = -log_S_ratio
        # D_vec = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
        D_elements, residuals, rank, singular_values = np.linalg.lstsq(b_design_matrix, -log_S_ratio, rcond=None)
        
        # Check rank to ensure the solution is well-determined
        if rank < 6:
            print(f"Warning: Rank of b_design_matrix ({rank}) is less than 6. Solution may be ill-determined.")
            # Depending on policy, one might return None or the potentially unstable solution.
            # For now, return the solution but with a warning.
            
        return D_elements
    except np.linalg.LinAlgError as e:
        print(f"Error during DTI fitting (least squares linear algebra error): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in calculate_D_elements: {e}")
        return None

def tensor_elements_to_matrix(D_elements):
    """
    Converts the 6 unique diffusion tensor elements into a 3x3 symmetric matrix.

    Args:
        D_elements (np.ndarray): 1D array of 6 tensor elements
                                 (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz).

    Returns:
        np.ndarray: 3x3 symmetric diffusion tensor matrix, or None if input is invalid.
    """
    if not isinstance(D_elements, np.ndarray):
        print("Error: D_elements must be a NumPy array.")
        return None
    if D_elements.shape != (6,):
        print(f"Error: Invalid input shape for D_elements. Expected (6,), got {D_elements.shape}.")
        return None
    
    D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = D_elements
    D_tensor = np.array([
        [D_xx, D_xy, D_xz],
        [D_xy, D_yy, D_yz],
        [D_xz, D_yz, D_zz]
    ], dtype=np.float64) # Use float64 for precision in eigenvalue decomposition
    return D_tensor

def calculate_dti_metrics(D_tensor_matrix):
    """
    Calculates DTI metrics (MD, FA, AD, RD, eigenvalues, eigenvectors)
    from a 3x3 symmetric diffusion tensor matrix.

    Args:
        D_tensor_matrix (np.ndarray): 3x3 symmetric diffusion tensor.

    Returns:
        tuple: Contains (MD, FA, AD, RD, sorted_eigenvalues, sorted_eigenvectors_matrix),
               or (None, None, None, None, None, None) if calculation fails.
               Eigenvectors are columns in the returned matrix, sorted corresponding to eigenvalues.
    """
    if not isinstance(D_tensor_matrix, np.ndarray):
        print("Error: D_tensor_matrix must be a NumPy array.")
        return None, None, None, None, None, None
        
    if D_tensor_matrix.shape != (3, 3):
        print(f"Error: Invalid input diffusion tensor matrix shape. Expected (3,3), got {D_tensor_matrix.shape}.")
        return None, None, None, None, None, None

    try:
        # eigh for symmetric matrices; returns eigenvalues and eigenvectors.
        # Eigenvalues are not guaranteed to be sorted by eigh.
        eigenvalues, eigenvectors = np.linalg.eigh(D_tensor_matrix) 

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1] # Get indices that would sort in ascending, then reverse
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors_matrix = eigenvectors[:, sorted_indices] # Reorder columns of eigenvectors

        # Clamp small negative eigenvalues to zero for physical plausibility and stability of metrics.
        # This is a common practice.
        clamped_eigenvalues = np.maximum(sorted_eigenvalues, 0)
        
        lambda1, lambda2, lambda3 = clamped_eigenvalues[0], clamped_eigenvalues[1], clamped_eigenvalues[2]

        # Mean Diffusivity (MD)
        MD = np.mean(clamped_eigenvalues) # (lambda1 + lambda2 + lambda3) / 3.0

        # Fractional Anisotropy (FA)
        # FA = sqrt(1/2) * sqrt( (l1-l2)^2 + (l2-l3)^2 + (l3-l1)^2 ) / sqrt(l1^2+l2^2+l3^2)
        # Denominator: sqrt(sum of squared eigenvalues)
        sum_sq_evals = np.sum(clamped_eigenvalues**2)
        
        if sum_sq_evals < 1e-20: # Threshold to consider all eigenvalues effectively zero
            FA = 0.0 # If all eigenvalues are zero, tensor is isotropic with no diffusion, FA is 0.
        else:
            # Numerator term for FA: (lambda1-lambda2)^2 + (lambda2-lambda3)^2 + (lambda3-lambda1)^2
            numerator_fa_sq = (lambda1 - lambda2)**2 + (lambda2 - lambda3)**2 + (lambda3 - lambda1)**2
            FA = np.sqrt(0.5 * numerator_fa_sq / sum_sq_evals)
        
        # Ensure FA is between 0 and 1, handle potential floating point inaccuracies
        FA = np.clip(FA, 0.0, 1.0)
        if not np.isfinite(FA): # Should be caught by sum_sq_evals check, but as a safeguard
            FA = 0.0

        # Axial Diffusivity (AD) - principal eigenvalue
        AD = lambda1

        # Radial Diffusivity (RD) - average of the two smaller eigenvalues
        RD = (lambda2 + lambda3) / 2.0

        return MD, FA, AD, RD, sorted_eigenvalues, sorted_eigenvectors_matrix

    except np.linalg.LinAlgError as e:
        print(f"Error during eigenvalue calculation: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred in calculate_dti_metrics: {e}")
        return None, None, None, None, None, None

if __name__ == '__main__':
    print("Testing DTI model functions...")

    # Test calculate_D_elements
    print("\n--- Testing calculate_D_elements ---")
    b_design = np.array([
        [1000, 0, 0, 0, 0, 0],      # Corresponds to b_xx for gx=(1,0,0), b=1000
        [0, 1000, 0, 0, 0, 0],      # Corresponds to b_yy for gy=(0,1,0), b=1000
        [0, 0, 1000, 0, 0, 0],      # Corresponds to b_zz for gz=(0,0,1), b=1000
        [500, 500, 0, 1000, 0, 0],  # Example b_xx,b_yy,2b_xy for g=(1/sqrt(2),1/sqrt(2),0), b=1000
        [500, 0, 500, 0, 1000, 0],  # Example b_xx,b_zz,2b_xz for g=(1/sqrt(2),0,1/sqrt(2)), b=1000
        [0, 500, 500, 0, 0, 1000]   # Example b_yy,b_zz,2b_yz for g=(0,1/sqrt(2),1/sqrt(2)), b=1000
    ]) 
    true_D_elements = np.array([1.5e-3, 0.5e-3, 0.3e-3, 0.1e-3, 0.2e-3, 0.05e-3])
    log_S_ratio_test = -np.dot(b_design, true_D_elements)
    
    fitted_D_elements = calculate_D_elements(log_S_ratio_test, b_design)
    if fitted_D_elements is not None:
        print(f"True D elements:   {np.array2string(true_D_elements, precision=2, floatmode='scientific')}")
        print(f"Fitted D elements: {np.array2string(fitted_D_elements, precision=2, floatmode='scientific')}")
        assert np.allclose(fitted_D_elements, true_D_elements, atol=1e-6), "Fitted D elements don't match true D"
        print("calculate_D_elements test PASSED.")
    else:
        print("calculate_D_elements test FAILED.")

    # Test tensor_elements_to_matrix
    print("\n--- Testing tensor_elements_to_matrix ---")
    D_matrix = tensor_elements_to_matrix(fitted_D_elements if fitted_D_elements is not None else true_D_elements)
    if D_matrix is not None:
        print(f"Constructed D tensor matrix:\n{D_matrix}")
        expected_D_matrix = np.array([
            [true_D_elements[0], true_D_elements[3], true_D_elements[4]],
            [true_D_elements[3], true_D_elements[1], true_D_elements[5]],
            [true_D_elements[4], true_D_elements[5], true_D_elements[2]]
        ])
        assert np.allclose(D_matrix, expected_D_matrix), "Constructed D matrix incorrect"
        print("tensor_elements_to_matrix test PASSED.")
    else:
        print("tensor_elements_to_matrix test FAILED.")

    # Test calculate_dti_metrics
    print("\n--- Testing calculate_dti_metrics ---")
    # Example: Isotropic diffusion Dxx=Dyy=Dzz=1e-3, others 0
    iso_D_elements = np.array([1e-3, 1e-3, 1e-3, 0, 0, 0])
    iso_D_matrix = tensor_elements_to_matrix(iso_D_elements)
    MD, FA, AD, RD, evals, evecs = calculate_dti_metrics(iso_D_matrix)
    if MD is not None:
        print(f"Metrics for isotropic tensor: MD={MD:.2e}, FA={FA:.3f}, AD={AD:.2e}, RD={RD:.2e}")
        print(f"Eigenvalues (sorted): {np.array2string(evals, precision=2, floatmode='scientific')}")
        assert np.isclose(MD, 1e-3), "MD incorrect for isotropic"
        assert np.isclose(FA, 0.0), "FA incorrect for isotropic"
        assert np.isclose(AD, 1e-3), "AD incorrect for isotropic"
        assert np.isclose(RD, 1e-3), "RD incorrect for isotropic"
        print("calculate_dti_metrics test for isotropic PASSED.")
    else:
        print("calculate_dti_metrics test for isotropic FAILED.")

    # Example: Anisotropic diffusion (using true_D_elements from before, but with Dxy,Dxz,Dyz=0 for easier eigenvalue check)
    aniso_D_elements_diag = np.array([1.5e-3, 0.5e-3, 0.3e-3, 0, 0, 0])
    aniso_D_matrix_diag = tensor_elements_to_matrix(aniso_D_elements_diag)
    MD_a, FA_a, AD_a, RD_a, evals_a, evecs_a = calculate_dti_metrics(aniso_D_matrix_diag)
    if MD_a is not None:
        print(f"Metrics for anisotropic (diagonal) tensor: MD={MD_a:.2e}, FA={FA_a:.3f}, AD={AD_a:.2e}, RD={RD_a:.2e}")
        print(f"Eigenvalues (sorted): {np.array2string(evals_a, precision=2, floatmode='scientific')}")
        
        expected_evals_a = np.sort([1.5e-3, 0.5e-3, 0.3e-3])[::-1]
        assert np.allclose(evals_a, expected_evals_a), "Eigenvalues incorrect for anisotropic (diagonal)"
        
        expected_MD_a = np.mean(expected_evals_a)
        assert np.isclose(MD_a, expected_MD_a), "MD incorrect for anisotropic (diagonal)"
        
        expected_AD_a = expected_evals_a[0]
        assert np.isclose(AD_a, expected_AD_a), "AD incorrect for anisotropic (diagonal)"
        
        expected_RD_a = (expected_evals_a[1] + expected_evals_a[2]) / 2.0
        assert np.isclose(RD_a, expected_RD_a), "RD incorrect for anisotropic (diagonal)"

        # FA calculation check
        l1, l2, l3 = expected_evals_a
        sum_sq_evals_a = np.sum(expected_evals_a**2)
        if sum_sq_evals_a > 1e-20:
            numerator_fa_sq_a = (l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2
            expected_FA_a = np.sqrt(0.5 * numerator_fa_sq_a / sum_sq_evals_a)
            assert np.isclose(FA_a, expected_FA_a), f"FA incorrect for anisotropic (diagonal). Got {FA_a:.3f}, expected {expected_FA_a:.3f}"
        else:
            assert np.isclose(FA_a, 0.0), "FA should be 0 if all eigenvalues are zero"

        print("calculate_dti_metrics test for anisotropic (diagonal) PASSED.")
    else:
        print("calculate_dti_metrics test for anisotropic (diagonal) FAILED.")
        
    print("\nAll DTI model function tests finished.")

```
