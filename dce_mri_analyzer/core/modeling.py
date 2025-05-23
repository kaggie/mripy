import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# from scipy.signal import convolve # Using np.convolve for now
from PyQt5.QtWidgets import QApplication # For processEvents in long loops

def _convolve_Cp_with_exp(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func) -> np.ndarray:
    """
    Helper function to convolve the AIF (Cp(t)) with Ktrans * exp(-(Ktrans/ve)*t).
    Assumes t is uniformly spaced.

    Args:
        t (np.ndarray): Array of time points, assumed to be uniformly spaced and sorted.
        Ktrans (float): Volume transfer constant.
        ve (float): Fractional volume of extravascular extracellular space (EES).
        Cp_t_interp_func: Interpolated AIF function Cp(t').

    Returns:
        np.ndarray: The result of the convolution, representing the Tofts component.
    """
    if len(t) < 2:
        return np.zeros_like(t)
    
    dt = t[1] - t[0]
    if dt <= 0: 
        raise ValueError("Time points 't' for convolution must be sorted and strictly increasing, implying dt > 0.")

    k_exp = Ktrans / (ve + 1e-9) 
    exp_decay_kernel = np.exp(-k_exp * t) 
    Cp_values_at_t = Cp_t_interp_func(t)
    convolution_result = np.convolve(Cp_values_at_t, exp_decay_kernel, mode='full')[:len(t)] * dt
    
    return Ktrans * convolution_result

def standard_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func) -> np.ndarray:
    """
    Standard Tofts model using convolution.
    Ct(t) = Ktrans * integral_0^t ( Cp(tau) * exp(-(Ktrans/ve)*(t-tau)) ) dtau

    Args:
        t (np.ndarray): Array of time points. (Assumed to be uniformly spaced for np.convolve)
        Ktrans (float): Volume transfer constant.
        ve (float): Fractional volume of EES.
        Cp_t_interp_func: Interpolated AIF function Cp(t').

    Returns:
        np.ndarray: Modeled tissue concentration curve.
    """
    if Ktrans < 0 or ve < 0: 
        return np.full_like(t, np.inf)

    return _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)

def extended_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, vp: float, Cp_t_interp_func) -> np.ndarray:
    """
    Extended Tofts model using convolution.
    Ct(t) = vp * Cp(t) + Ktrans * integral_0^t ( Cp(tau) * exp(-(Ktrans/ve)*(t-tau)) ) dtau

    Args:
        t (np.ndarray): Array of time points. (Assumed to be uniformly spaced for np.convolve)
        Ktrans (float): Volume transfer constant.
        ve (float): Fractional volume of EES.
        vp (float): Fractional plasma volume.
        Cp_t_interp_func: Interpolated AIF function Cp(t').

    Returns:
        np.ndarray: Modeled tissue concentration curve.
    """
    if Ktrans < 0 or ve < 0 or vp < 0:
        return np.full_like(t, np.inf) 

    vp_component = vp * Cp_t_interp_func(t)
    tofts_component = _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)
    
    return vp_component + tofts_component

# Modified to accept Cp_interp_func
def fit_standard_tofts(
    t_tissue: np.ndarray, 
    Ct_tissue: np.ndarray, 
    Cp_interp_func, # Changed from t_aif, Cp_aif
    initial_params=(0.1, 0.2), 
    bounds_params=([0, 0], [1.0, 1.0]) 
) -> tuple[tuple[float, float] | None, np.ndarray | None]:
    """
    Fits the Standard Tofts model to a single voxel's tissue concentration curve.

    Args:
        t_tissue (np.ndarray): Time points for tissue curve (must be uniformly spaced).
        Ct_tissue (np.ndarray): Tissue concentrations for the voxel.
        Cp_interp_func: Interpolated AIF function.
        initial_params (tuple): Initial guess for (Ktrans, ve).
        bounds_params (tuple): Bounds for (Ktrans, ve).

    Returns:
        tuple: ((fitted_Ktrans, fitted_ve), fitted_curve) or (None, None) if fitting fails.
    """
    if not (len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return None, None
    # dt check removed as it's implicitly handled by _convolve_Cp_with_exp or should be checked before calling voxel-wise
    
    try:
        objective_func = lambda t_obj, Ktrans, ve: standard_tofts_model_conv(t_obj, Ktrans, ve, Cp_interp_func)
        
        popt, pcov = curve_fit(
            objective_func, 
            t_tissue, 
            Ct_tissue, 
            p0=initial_params, 
            bounds=bounds_params, 
            method='trf',
            ftol=1e-4, xtol=1e-4, gtol=1e-4 # Adjusted tolerances slightly
        )
        
        fitted_Ktrans, fitted_ve = popt
        # Avoid re-calculating fitted_curve here if not immediately needed by voxel-wise,
        # but it's useful for single voxel test.
        fitted_curve = standard_tofts_model_conv(t_tissue, fitted_Ktrans, fitted_ve, Cp_interp_func) 
        
        return (fitted_Ktrans, fitted_ve), fitted_curve
    
    except RuntimeError:
        return None, None
    except ValueError: # Can happen from bad inputs to model during fit
        return None, None
    except Exception: 
        return None, None

# Modified to accept Cp_interp_func
def fit_extended_tofts(
    t_tissue: np.ndarray, 
    Ct_tissue: np.ndarray, 
    Cp_interp_func, # Changed from t_aif, Cp_aif
    initial_params=(0.1, 0.2, 0.05), 
    bounds_params=([0, 0, 0], [1.0, 1.0, 0.5])
) -> tuple[tuple[float, float, float] | None, np.ndarray | None]:
    """
    Fits the Extended Tofts model to a single voxel's tissue concentration data.
    Args:
        Cp_interp_func: Interpolated AIF function.
    (Other args similar to fit_standard_tofts)
    Returns:
        tuple: ((fitted_Ktrans, fitted_ve, fitted_vp), fitted_curve) or (None, None) if fitting fails.
    """
    if not (len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return None, None

    try:
        objective_func = lambda t_obj, Ktrans, ve, vp: extended_tofts_model_conv(t_obj, Ktrans, ve, vp, Cp_interp_func)
        
        popt, pcov = curve_fit(
            objective_func, 
            t_tissue, 
            Ct_tissue, 
            p0=initial_params, 
            bounds=bounds_params, 
            method='trf',
            ftol=1e-4, xtol=1e-4, gtol=1e-4
        )
        
        fitted_Ktrans, fitted_ve, fitted_vp = popt
        fitted_curve = extended_tofts_model_conv(t_tissue, fitted_Ktrans, fitted_ve, fitted_vp, Cp_interp_func)
        
        return (fitted_Ktrans, fitted_ve, fitted_vp), fitted_curve

    except RuntimeError:
        return None, None
    except ValueError:
        return None, None
    except Exception:
        return None, None

def fit_standard_tofts_voxelwise(
    Ct_data: np.ndarray, 
    t_tissue: np.ndarray, 
    Cp_interp_func, # Changed from t_aif, Cp_aif
    mask: np.ndarray = None, 
    initial_params=(0.1, 0.2), 
    bounds_params=([0.001, 0.001], [1.0, 1.0]) # Ensure Ktrans, ve > 0
):
    """
    Fits the Standard Tofts model voxel-wise to 4D concentration data.
    """
    if Ct_data.ndim != 4:
        raise ValueError("Ct_data must be a 4D array (x, y, z, time).")
    if t_tissue.ndim != 1 or Ct_data.shape[3] != len(t_tissue):
        raise ValueError("t_tissue must be a 1D array matching time dimension of Ct_data.")

    spatial_dims = Ct_data.shape[:3]
    ktrans_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    ve_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    
    total_voxels = np.prod(spatial_dims) if mask is None else np.sum(mask)
    processed_voxels = 0
    voxels_to_process_for_update = max(1, total_voxels // 100) # Update roughly every 1%

    print(f"Starting Standard Tofts voxel-wise fitting for {total_voxels} voxels...")

    for x in range(spatial_dims[0]):
        for y in range(spatial_dims[1]):
            for z in range(spatial_dims[2]):
                if mask is not None and not mask[x, y, z]:
                    continue

                Ct_voxel = Ct_data[x, y, z, :]
                if np.all(np.isnan(Ct_voxel)) or np.all(Ct_voxel == 0) or np.sum(~np.isnan(Ct_voxel)) < 3 : # Need a few points to fit
                    continue
                
                # Remove NaNs from this voxel's curve if any (e.g. from S0=0)
                valid_indices = ~np.isnan(Ct_voxel)
                Ct_voxel_clean = Ct_voxel[valid_indices]
                t_tissue_clean = t_tissue[valid_indices]

                if len(Ct_voxel_clean) < len(initial_params) * 2: # Need enough points
                    continue

                fit_result, _ = fit_standard_tofts(
                    t_tissue_clean, Ct_voxel_clean, Cp_interp_func, initial_params, bounds_params
                )
                
                if fit_result:
                    ktrans_map[x, y, z] = fit_result[0]
                    ve_map[x, y, z] = fit_result[1]
                
                processed_voxels += 1
                if processed_voxels % voxels_to_process_for_update == 0:
                    progress = 100.0 * processed_voxels / total_voxels
                    print(f"Standard Tofts fitting progress: {progress:.1f}% ({processed_voxels}/{total_voxels})")
                    QApplication.processEvents() # Allow GUI to update if in main thread (caution)
        # print(f"Finished slice z={z} for Standard Tofts.") # Optional: slice-wise progress

    print("Standard Tofts voxel-wise fitting completed.")
    return {"Ktrans": ktrans_map, "ve": ve_map}


def fit_extended_tofts_voxelwise(
    Ct_data: np.ndarray, 
    t_tissue: np.ndarray, 
    Cp_interp_func, # Changed from t_aif, Cp_aif
    mask: np.ndarray = None, 
    initial_params=(0.1, 0.2, 0.05), 
    bounds_params=([0.001, 0.001, 0.001], [1.0, 1.0, 0.5]) # Ensure params > 0
):
    """
    Fits the Extended Tofts model voxel-wise to 4D concentration data.
    """
    if Ct_data.ndim != 4:
        raise ValueError("Ct_data must be a 4D array (x, y, z, time).")
    if t_tissue.ndim != 1 or Ct_data.shape[3] != len(t_tissue):
        raise ValueError("t_tissue must be a 1D array matching time dimension of Ct_data.")

    spatial_dims = Ct_data.shape[:3]
    ktrans_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    ve_map = np.full(spatial_dims, np.nan, dtype=np.float32)
    vp_map = np.full(spatial_dims, np.nan, dtype=np.float32)

    total_voxels = np.prod(spatial_dims) if mask is None else np.sum(mask)
    processed_voxels = 0
    voxels_to_process_for_update = max(1, total_voxels // 100) # Update roughly every 1%

    print(f"Starting Extended Tofts voxel-wise fitting for {total_voxels} voxels...")

    for x in range(spatial_dims[0]):
        for y in range(spatial_dims[1]):
            for z in range(spatial_dims[2]):
                if mask is not None and not mask[x, y, z]:
                    continue

                Ct_voxel = Ct_data[x, y, z, :]
                if np.all(np.isnan(Ct_voxel)) or np.all(Ct_voxel == 0) or np.sum(~np.isnan(Ct_voxel)) < 3:
                    continue

                valid_indices = ~np.isnan(Ct_voxel)
                Ct_voxel_clean = Ct_voxel[valid_indices]
                t_tissue_clean = t_tissue[valid_indices]

                if len(Ct_voxel_clean) < len(initial_params) * 2:
                    continue
                
                fit_result, _ = fit_extended_tofts(
                    t_tissue_clean, Ct_voxel_clean, Cp_interp_func, initial_params, bounds_params
                )
                
                if fit_result:
                    ktrans_map[x, y, z] = fit_result[0]
                    ve_map[x, y, z] = fit_result[1]
                    vp_map[x, y, z] = fit_result[2]

                processed_voxels += 1
                if processed_voxels % voxels_to_process_for_update == 0:
                    progress = 100.0 * processed_voxels / total_voxels
                    print(f"Extended Tofts fitting progress: {progress:.1f}% ({processed_voxels}/{total_voxels})")
                    QApplication.processEvents() # Allow GUI to update if in main thread
        # print(f"Finished slice z={z} for Extended Tofts.")

    print("Extended Tofts voxel-wise fitting completed.")
    return {"Ktrans": ktrans_map, "ve": ve_map, "vp": vp_map}
