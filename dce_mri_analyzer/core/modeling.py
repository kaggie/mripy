import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz # Added for Patlak model
import multiprocessing
import os
from functools import partial

# --- Model Definitions (Convolution-based and Patlak) ---
def _convolve_Cp_with_exp(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func) -> np.ndarray:
    if len(t) < 2: return np.zeros_like(t)
    dt = t[1] - t[0]
    if dt <= 0: raise ValueError("Time points 't' for convolution must be sorted and strictly increasing.")
    k_exp = Ktrans / (ve + 1e-9) 
    exp_decay_kernel = np.exp(-k_exp * t) 
    Cp_values_at_t = Cp_t_interp_func(t)
    convolution_result = np.convolve(Cp_values_at_t, exp_decay_kernel, mode='full')[:len(t)] * dt
    return Ktrans * convolution_result

def standard_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func) -> np.ndarray:
    if Ktrans < 0 or ve < 0: return np.full_like(t, np.inf)
    return _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)

def extended_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, vp: float, Cp_t_interp_func) -> np.ndarray:
    if Ktrans < 0 or ve < 0 or vp < 0: return np.full_like(t, np.inf) 
    vp_component = vp * Cp_t_interp_func(t)
    tofts_component = _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)
    return vp_component + tofts_component

def patlak_model(t_points: np.ndarray, Ktrans: float, vp: float, 
                 Cp_t_interp_func, integral_Cp_dt_interp_func) -> np.ndarray:
    """
    Patlak model.
    Ct(t) = Ktrans * integral(Cp(tau)d(tau))_0^t + vp * Cp(t)

    Args:
        t_points (np.ndarray): Array of time points.
        Ktrans (float): Patlak Ktrans (slope).
        vp (float): Fractional plasma volume (intercept/Cp(t) when plotting Ct(t)/Cp(t) vs integral(Cp)/Cp(t)).
        Cp_t_interp_func: Interpolated AIF function Cp(t').
        integral_Cp_dt_interp_func: Interpolated function for integral_0^t Cp(tau)d(tau).

    Returns:
        np.ndarray: Modeled tissue concentration curve.
    """
    if Ktrans < 0 or vp < 0: # Basic validation
        return np.full_like(t_points, np.inf)

    Cp_values = Cp_t_interp_func(t_points)
    integral_Cp_values = integral_Cp_dt_interp_func(t_points)
    
    return Ktrans * integral_Cp_values + vp * Cp_values


# --- Single-Voxel Fitting Functions ---
def fit_standard_tofts(
    t_tissue: np.ndarray, 
    Ct_tissue: np.ndarray, 
    Cp_interp_func, 
    initial_params=(0.1, 0.2), 
    bounds_params=([0, 0], [1.0, 1.0]) 
) -> tuple[tuple[float, float] | None, np.ndarray | None]:
    if not (len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)): return None, None
    try:
        objective_func = lambda t_obj, Ktrans, ve: standard_tofts_model_conv(t_obj, Ktrans, ve, Cp_interp_func)
        popt, pcov = curve_fit(objective_func, t_tissue, Ct_tissue, p0=initial_params, bounds=bounds_params, method='trf', ftol=1e-4, xtol=1e-4, gtol=1e-4)
        fitted_Ktrans, fitted_ve = popt
        fitted_curve = standard_tofts_model_conv(t_tissue, fitted_Ktrans, fitted_ve, Cp_interp_func) 
        return (fitted_Ktrans, fitted_ve), fitted_curve
    except Exception: return None, None 

def fit_extended_tofts(
    t_tissue: np.ndarray, 
    Ct_tissue: np.ndarray, 
    Cp_interp_func, 
    initial_params=(0.1, 0.2, 0.05), 
    bounds_params=([0, 0, 0], [1.0, 1.0, 0.5])
) -> tuple[tuple[float, float, float] | None, np.ndarray | None]:
    if not (len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)): return None, None
    try:
        objective_func = lambda t_obj, Ktrans, ve, vp: extended_tofts_model_conv(t_obj, Ktrans, ve, vp, Cp_interp_func)
        popt, pcov = curve_fit(objective_func, t_tissue, Ct_tissue, p0=initial_params, bounds=bounds_params, method='trf', ftol=1e-4, xtol=1e-4, gtol=1e-4)
        fitted_Ktrans, fitted_ve, fitted_vp = popt
        fitted_curve = extended_tofts_model_conv(t_tissue, fitted_Ktrans, fitted_ve, fitted_vp, Cp_interp_func)
        return (fitted_Ktrans, fitted_ve, fitted_vp), fitted_curve
    except Exception: return None, None

def fit_patlak_model(
    t_tissue: np.ndarray, 
    Ct_tissue: np.ndarray, 
    Cp_interp_func, 
    integral_Cp_dt_interp_func,
    initial_params=(0.05, 0.05), 
    bounds_params=([0, 0], [1.0, 0.5]) # Ktrans_patlak, vp_patlak
) -> tuple[tuple[float, float] | None, np.ndarray | None]:
    """
    Fits the Patlak model to a single voxel's tissue concentration curve.
    """
    if not (len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)): return None, None
    
    # Patlak analysis is typically linear for t > t*, where t* is some time after which
    # backflux from EES to plasma is negligible or steady state is achieved.
    # For simplicity, we fit to all points here but this could be refined.
    # Or, ensure the time range for fitting (t_tissue) is appropriate.
    
    def objective_func(t_obj, Ktrans_patlak, vp_patlak):
        return patlak_model(t_obj, Ktrans_patlak, vp_patlak, Cp_interp_func, integral_Cp_dt_interp_func)

    try:
        popt, pcov = curve_fit(objective_func, t_tissue, Ct_tissue, 
                               p0=initial_params, bounds=bounds_params, method='trf',
                               xtol=1e-4, ftol=1e-4, gtol=1e-4)
        fitted_Ktrans, fitted_vp = popt
        fitted_curve = patlak_model(t_tissue, fitted_Ktrans, fitted_vp, 
                                    Cp_interp_func, integral_Cp_dt_interp_func)
        return (fitted_Ktrans, fitted_vp), fitted_curve
    except RuntimeError: return (np.nan, np.nan), np.full_like(t_tissue, np.nan)
    except ValueError: return (np.nan, np.nan), np.full_like(t_tissue, np.nan)
    except Exception: return (np.nan, np.nan), np.full_like(t_tissue, np.nan)


# --- Multiprocessing Worker Function (Top-Level) ---
def _fit_voxel_worker(args_tuple):
    voxel_idx_xyz, Ct_voxel, t_tissue, t_aif, Cp_aif, model_name, \
    initial_params_for_model, bounds_params_for_model = args_tuple

    if np.all(np.isnan(Ct_voxel)) or np.all(Ct_voxel == 0) or np.sum(~np.isnan(Ct_voxel)) < 3:
        return voxel_idx_xyz, model_name, {"error": "Skipped (NaN, zero, or insufficient data points)"}
    valid_indices = ~np.isnan(Ct_voxel)
    Ct_voxel_clean = Ct_voxel[valid_indices]; t_tissue_clean = t_tissue[valid_indices]
    min_points_needed = len(initial_params_for_model) * 2 
    if len(Ct_voxel_clean) < min_points_needed:
        return voxel_idx_xyz, model_name, {"error": f"Skipped (insufficient valid data: got {len(Ct_voxel_clean)}, need {min_points_needed})"}

    try:
        Cp_interp_func = interp1d(t_aif, Cp_aif, kind='linear', bounds_error=False, fill_value=0.0)
        params_tuple = None; param_names = []

        if model_name == "Standard Tofts":
            params_tuple, _ = fit_standard_tofts(t_tissue_clean, Ct_voxel_clean, Cp_interp_func, initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans", "ve"]
        elif model_name == "Extended Tofts":
            params_tuple, _ = fit_extended_tofts(t_tissue_clean, Ct_voxel_clean, Cp_interp_func, initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans", "ve", "vp"]
        elif model_name == "Patlak":
            # Create integral interpolator for Patlak inside worker
            integral_Cp_dt_aif = cumtrapz(Cp_aif, t_aif, initial=0)
            integral_Cp_dt_interp_func = interp1d(t_aif, integral_Cp_dt_aif, kind='linear', bounds_error=False, fill_value=0.0)
            params_tuple, _ = fit_patlak_model(t_tissue_clean, Ct_voxel_clean, Cp_interp_func, integral_Cp_dt_interp_func, initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans_patlak", "vp_patlak"]
        else:
            return voxel_idx_xyz, model_name, {"error": f"Unknown model: {model_name}"}
        
        if params_tuple is None or np.any(np.isnan(params_tuple)): 
            return voxel_idx_xyz, model_name, {"error": "Fit failed (no parameters or NaN returned)"}
        return voxel_idx_xyz, model_name, dict(zip(param_names, params_tuple))
    except Exception as e: return voxel_idx_xyz, model_name, {"error": f"Unexpected error in worker: {e}"}


# --- Voxel-wise Fitting Functions (Parallelized) ---
def _base_fit_voxelwise(
    Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
    model_name: str, param_names_map: dict, # e.g. {"Ktrans":"Ktrans_patlak", "vp":"vp_patlak"}
    initial_params: tuple, bounds_params: tuple,
    mask: np.ndarray = None, num_processes: int = None
):
    if Ct_data.ndim != 4: raise ValueError("Ct_data must be a 4D array.")
    if t_tissue.ndim != 1 or Ct_data.shape[3] != len(t_tissue): raise ValueError("t_tissue must match time dim of Ct_data.")
    if t_aif.ndim != 1 or Cp_aif.ndim != 1 or len(t_aif) != len(Cp_aif): raise ValueError("t_aif and Cp_aif must be 1D arrays of same length.")

    num_proc_to_use = num_processes if num_processes and num_processes > 0 else os.cpu_count()
    if num_proc_to_use is None: num_proc_to_use = 1

    spatial_dims = Ct_data.shape[:3]
    # Initialize parameter maps based on param_names_map keys
    result_maps = {map_key: np.full(spatial_dims, np.nan, dtype=np.float32) for map_key in param_names_map.values()}
    
    tasks_args_list = []
    for x in range(spatial_dims[0]):
        for y in range(spatial_dims[1]):
            for z in range(spatial_dims[2]):
                if mask is not None and not mask[x, y, z]: continue
                Ct_voxel = Ct_data[x, y, z, :]
                tasks_args_list.append(((x,y,z), Ct_voxel, t_tissue, t_aif, Cp_aif, model_name, initial_params, bounds_params))

    if not tasks_args_list: print(f"No voxels to process for {model_name} fitting."); return result_maps

    print(f"Starting {model_name} fitting for {len(tasks_args_list)} voxels using up to {num_proc_to_use} processes...")
    
    results_list = []
    if num_proc_to_use > 1 and len(tasks_args_list) > 1:
        try:
            with multiprocessing.Pool(processes=num_proc_to_use) as pool: results_list = pool.map(_fit_voxel_worker, tasks_args_list)
        except Exception as e: print(f"Error during multiprocessing pool for {model_name}: {e}. Falling back to serial."); num_proc_to_use = 1
            
    if not results_list or num_proc_to_use == 1:
        print(f"Processing {model_name} serially..."); results_list = [_fit_voxel_worker(args) for args in tasks_args_list]

    for result_item in results_list:
        if result_item is None: continue
        voxel_idx_xyz, model_name_out, result_dict = result_item
        if model_name_out == model_name and "error" not in result_dict:
            for p_name_orig, p_name_map in param_names_map.items():
                result_maps[p_name_map][voxel_idx_xyz] = result_dict.get(p_name_orig, np.nan)
        # Optional: Log errors here if needed
            
    print(f"{model_name} voxel-wise fitting completed.")
    return result_maps


def fit_standard_tofts_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, mask=None, initial_params=(0.1, 0.2), bounds_params=([0.001, 0.001], [1.0, 1.0]), num_processes=None):
    param_names_map = {"Ktrans": "Ktrans", "ve": "ve"}
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, "Standard Tofts", param_names_map, initial_params, bounds_params, mask, num_processes)

def fit_extended_tofts_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, mask=None, initial_params=(0.1, 0.2, 0.05), bounds_params=([0.001, 0.001, 0.001], [1.0, 1.0, 0.5]), num_processes=None):
    param_names_map = {"Ktrans": "Ktrans", "ve": "ve", "vp": "vp"}
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, "Extended Tofts", param_names_map, initial_params, bounds_params, mask, num_processes)

def fit_patlak_model_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, mask=None, initial_params=(0.05, 0.05), bounds_params=([0, 0], [1.0, 0.5]), num_processes=None):
    # Use distinct names for Patlak parameters to avoid key collisions if used together
    param_names_map = {"Ktrans_patlak": "Ktrans_patlak", "vp_patlak": "vp_patlak"}
    # The worker maps its internal "Ktrans_patlak", "vp_patlak" to these keys in the result_dict
    # So, _base_fit_voxelwise needs to look for these in result_dict
    # This means the param_names_map should be {"Ktrans_patlak":"Ktrans_patlak", "vp_patlak":"vp_patlak"}
    # if the worker returns "Ktrans_patlak" and "vp_patlak".
    # Or, the worker returns generic "Ktrans", "vp" and the mapping happens here.
    # The current _fit_voxel_worker uses "Ktrans_patlak", "vp_patlak" as keys in its returned dict.
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif, "Patlak", param_names_map, initial_params, bounds_params, mask, num_processes)
