import unittest
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz # For Patlak tests
from unittest.mock import patch, MagicMock, ANY # ANY for some mock calls
import os
import sys

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import modeling

class TestModelingFunctions(unittest.TestCase): # Renamed to be more general
    def setUp(self):
        # Time vector for AIF and tissue
        self.t_points = np.linspace(0, 1.0, 61) # Time in minutes for consistency with typical AIF params
        
        # Sample AIF (simple boxcar for testing, not realistic for real data but predictable)
        self.Cp_aif = np.zeros_like(self.t_points)
        self.Cp_aif[5:15] = 5.0 # Concentration in mM, assuming time in minutes
        
        self.Cp_interp_func = interp1d(self.t_points, self.Cp_aif, kind='linear', bounds_error=False, fill_value=0.0)
        
        # Integral of Cp for Patlak
        self.integral_Cp_dt_aif = cumtrapz(self.Cp_aif, self.t_points, initial=0)
        self.integral_Cp_dt_interp_func = interp1d(self.t_points, self.integral_Cp_dt_aif, kind='linear', bounds_error=False, fill_value=0.0)

    def test_patlak_model_function(self):
        """Test the Patlak model function with known inputs."""
        Ktrans_patlak = 0.1 # min^-1
        vp_patlak = 0.05    # dimensionless
        test_times = np.array([0, 0.1, 0.2, 0.3]) # Time in minutes
        # Cp(0.1) = 5.0, integral(Cp(0.1)) = (0.1-5/60)*5.0 = (0.1-0.0833)*5 = 0.0166*5 = 0.083 (approx)
        # Cp(0.2) = 0 (bolus ends at 14/60 = 0.233 min). integral(Cp(0.2)) = (14/60-5/60)*5 = (9/60)*5 = 0.15*5 = 0.75
        
        calculated_Ct = modeling.patlak_model(test_times, Ktrans_patlak, vp_patlak, 
                                              self.Cp_interp_func, self.integral_Cp_dt_interp_func)
        
        expected_Ct_manual = np.zeros_like(test_times)
        for i, t_val in enumerate(test_times):
            cp_val = self.Cp_interp_func(t_val)
            int_cp_val = self.integral_Cp_dt_interp_func(t_val)
            expected_Ct_manual[i] = Ktrans_patlak * int_cp_val + vp_patlak * cp_val
            
        np.testing.assert_array_almost_equal(calculated_Ct, expected_Ct_manual, decimal=5)


    def test_fit_patlak_model_single_voxel(self):
        """Test fitting the Patlak model to synthetic single-voxel data."""
        Ktrans_true = 0.15; vp_true = 0.03
        Ct_tissue_true = modeling.patlak_model(self.t_points, Ktrans_true, vp_true, self.Cp_interp_func, self.integral_Cp_dt_interp_func)
        noise_level = 0.01 * np.max(Ct_tissue_true) if np.max(Ct_tissue_true) > 0 else 0.01
        Ct_tissue_noisy = Ct_tissue_true + np.random.normal(0, noise_level, size=Ct_tissue_true.shape)
        initial_params = (0.1, 0.02); bounds = ([0, 0], [0.5, 0.3])
        
        params_fitted, _ = modeling.fit_patlak_model(
            self.t_points, Ct_tissue_noisy, 
            self.Cp_interp_func, self.integral_Cp_dt_interp_func,
            initial_params=initial_params, bounds_params=bounds
        )
        self.assertIsNotNone(params_fitted, "Patlak fitting failed to return parameters.")
        if params_fitted is not None and not np.all(np.isnan(params_fitted)):
            Ktrans_fit, vp_fit = params_fitted
            np.testing.assert_allclose(Ktrans_fit, Ktrans_true, rtol=0.3, atol=0.02) 
            np.testing.assert_allclose(vp_fit, vp_true, rtol=0.4, atol=0.02)     

    def test_fit_patlak_model_single_voxel_invalid_data(self):
        """Test Patlak fitting with invalid (e.g., all NaN) Ct_tissue."""
        Ct_tissue_nan = np.full_like(self.t_points, np.nan)
        params_fitted, curve_fitted = modeling.fit_patlak_model(
            self.t_points, Ct_tissue_nan, self.Cp_interp_func, self.integral_Cp_dt_interp_func
        )
        self.assertTrue(np.all(np.isnan(params_fitted)), "Expected NaN parameters for all-NaN input.")
        self.assertTrue(np.all(np.isnan(curve_fitted)), "Expected NaN curve for all-NaN input.")

    # --- Tests for 2CXM ---
    def test_ode_system_2cxm(self):
        """Test the 2CXM ODE system definition."""
        t_test = 0.1 # Example time point
        y_test = [0.5, 0.2] # C_p_tis, C_e_tis
        Fp, PS, vp, ve = 0.5, 0.05, 0.1, 0.2 # Example parameters (units depend on time unit)
        
        mock_Cp_aif_val = 1.0
        def mock_aif_interp(t): return mock_Cp_aif_val

        dC_p_tis_dt_expected = (Fp / vp) * (mock_Cp_aif_val - y_test[0]) - (PS / vp) * (y_test[0] - y_test[1])
        dC_e_tis_dt_expected = (PS / ve) * (y_test[0] - y_test[1])
        
        derivatives = modeling._ode_system_2cxm(t_test, y_test, Fp, PS, vp, ve, mock_aif_interp)
        
        self.assertAlmostEqual(derivatives[0], dC_p_tis_dt_expected)
        self.assertAlmostEqual(derivatives[1], dC_e_tis_dt_expected)

    def test_solve_2cxm_ode_model_runs(self):
        """Basic test that solve_2cxm_ode_model runs and returns correct shape."""
        Fp, PS, vp, ve = 0.6, 0.04, 0.15, 0.25 # Example parameters
        
        # Use a simple constant AIF for this test
        const_aif_val = 1.0
        cp_interp_const = lambda t: const_aif_val 
        
        Ct_model = modeling.solve_2cxm_ode_model(self.t_points, Fp, PS, vp, ve, cp_interp_const, t_span_max=self.t_points[-1])
        
        self.assertEqual(Ct_model.shape, self.t_points.shape)
        self.assertTrue(np.all(Ct_model >= 0), "Concentrations should be non-negative for positive inputs.")
        self.assertFalse(np.any(np.isinf(Ct_model)), "Model returned Inf, check parameters or solver.")
        self.assertFalse(np.any(np.isnan(Ct_model)), "Model returned NaN, check parameters or solver.")


    def test_fit_2cxm_model_single_voxel_synthetic(self):
        """Test fitting the 2CXM model to synthetic single-voxel data."""
        Fp_true, PS_true, vp_true, ve_true = 0.5, 0.03, 0.1, 0.2
        
        # Generate synthetic tissue data using the 2CXM model
        # Use the AIF from setUp
        Ct_tissue_true = modeling.solve_2cxm_ode_model(self.t_points, Fp_true, PS_true, vp_true, ve_true, 
                                                      self.Cp_interp_func, t_span_max=self.t_points[-1])
        
        self.assertFalse(np.any(np.isinf(Ct_tissue_true)|np.isnan(Ct_tissue_true)), "True data generation failed for 2CXM fit test.")

        # Add some noise (careful with noise level for complex models)
        noise_level = 0.005 * np.max(Ct_tissue_true) if np.max(Ct_tissue_true) > 0 else 0.005
        Ct_tissue_noisy = Ct_tissue_true + np.random.normal(0, noise_level, size=Ct_tissue_true.shape)
        
        initial_params = (0.4, 0.02, 0.08, 0.18) # Fp, PS, vp, ve
        # Bounds need to be reasonable, e.g., vp and ve cannot be zero.
        bounds = ([0, 0, 1e-3, 1e-3], [1.0, 0.5, 0.3, 0.5]) 
        
        params_fitted, _ = modeling.fit_2cxm_model(
            self.t_points, Ct_tissue_noisy, 
            self.Cp_interp_func, t_aif_max=self.t_points[-1], 
            initial_params=initial_params, bounds_params=bounds
        )
        
        self.assertIsNotNone(params_fitted, "2CXM fitting failed to return parameters.")
        if params_fitted is not None and not np.all(np.isnan(params_fitted)):
            Fp_fit, PS_fit, vp_fit, ve_fit = params_fitted
            # Tolerances might need to be quite loose for a 4-parameter fit
            # especially if the AIF is not very informative or time range is short.
            np.testing.assert_allclose(Fp_fit, Fp_true, rtol=0.5, atol=0.1) 
            np.testing.assert_allclose(PS_fit, PS_true, rtol=0.5, atol=0.02)
            np.testing.assert_allclose(vp_fit, vp_true, rtol=0.5, atol=0.05)
            np.testing.assert_allclose(ve_fit, ve_true, rtol=0.5, atol=0.05)

    @patch('core.modeling.fit_2cxm_model') # Add 2CXM to the mocks
    @patch('core.modeling.fit_patlak_model')
    @patch('core.modeling.fit_extended_tofts')
    @patch('core.modeling.fit_standard_tofts')
    def test_fit_voxel_worker_model_calls(self, mock_fit_std, mock_fit_ext, mock_fit_patlak, mock_fit_2cxm):
        """Test that _fit_voxel_worker calls the correct model fitting logic."""
        voxel_idx = (0,0,0); Ct_voxel = np.array([0.1,0.2,0.3,0.4,0.5]); t_tissue = np.arange(5)
        t_aif_worker = np.arange(6); Cp_aif_worker = np.zeros(6); Cp_aif_worker[1:3]=1.0

        # Standard Tofts
        mock_fit_std.return_value = ((0.2, 0.3), MagicMock()); args_std = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.1), ([0,0],[1,1])); _, _, res_std = modeling._fit_voxel_worker(args_std); mock_fit_std.assert_called_once(); self.assertEqual(res_std.get("Ktrans"), 0.2)
        # Extended Tofts
        mock_fit_ext.return_value = ((0.25, 0.35, 0.05), MagicMock()); args_ext = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Extended Tofts", (0.1,0.1,0.01), ([0,0,0],[1,1,1])); _, _, res_ext = modeling._fit_voxel_worker(args_ext); mock_fit_ext.assert_called_once(); self.assertEqual(res_ext.get("vp"), 0.05)
        # Patlak
        mock_fit_patlak.return_value = ((0.1, 0.05), MagicMock()); args_patlak = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Patlak", (0.05,0.02), ([0,0],[0.3,0.2])); _, _, res_patlak = modeling._fit_voxel_worker(args_patlak); mock_fit_patlak.assert_called_once(); self.assertEqual(res_patlak.get("Ktrans_patlak"), 0.1)
        # 2CXM
        mock_fit_2cxm.return_value = ((0.6,0.04,0.15,0.25), MagicMock()); args_2cxm = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "2CXM", (0.5,0.03,0.1,0.2), ([0,0,0,0],[1,1,1,1])); _, _, res_2cxm = modeling._fit_voxel_worker(args_2cxm); mock_fit_2cxm.assert_called_once(); self.assertEqual(res_2cxm.get("Fp_2cxm"), 0.6)
        # Check that t_aif_max was passed to fit_2cxm_model (ANY used as it's an internal detail of how worker calls it)
        mock_fit_2cxm.assert_called_with(ANY, ANY, ANY, t_aif_worker[-1], ANY, ANY)


    def test_fit_voxel_worker_skip_nan_data(self): # Unchanged
        voxel_idx = (1,1,1); Ct_voxel_nan = np.full(5, np.nan); t_tissue = np.arange(5)
        t_aif_worker = np.arange(6); Cp_aif_worker = np.zeros(6)
        args_tuple = (voxel_idx, Ct_voxel_nan, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.2), ([0,0],[1,1]))
        idx_out, model_out, result_dict = modeling._fit_voxel_worker(args_tuple)
        self.assertEqual(idx_out, voxel_idx); self.assertIn("error", result_dict); self.assertIn("Skipped", result_dict["error"])
        
    def test_fit_voxel_worker_skip_insufficient_data(self): # Unchanged
        voxel_idx = (1,1,1); Ct_voxel_short = np.array([0.1, np.nan, 0.3, np.nan, np.nan]); t_tissue = np.arange(5)
        t_aif_worker = np.arange(6); Cp_aif_worker = np.zeros(6)
        args_tuple = (voxel_idx, Ct_voxel_short, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.2), ([0,0],[1,1]))
        idx_out, model_out, result_dict = modeling._fit_voxel_worker(args_tuple)
        self.assertEqual(idx_out, voxel_idx); self.assertIn("error", result_dict); self.assertIn("insufficient valid data points", result_dict["error"])

    @patch('core.modeling._fit_voxel_worker') 
    def _run_voxelwise_parallel_test(self, mock_worker, fit_function_name, model_name_in_worker, param_keys_expected, default_initial_params, default_bounds):
        """Helper to test voxel-wise parallel calls for different models."""
        def side_effect_func(args):
            idx, _, _, _, _, model_name_call, _, _ = args
            if model_name_call == model_name_in_worker:
                return_params = {key: idx[0]*0.1 + idx[1]*0.01 + (param_keys_expected.index(key)*0.001) for key in param_keys_expected}
                return idx, model_name_call, return_params
            return idx, model_name_call, {"error": "Wrong model for mock"}
        mock_worker.side_effect = side_effect_func

        Ct_data = np.random.rand(2,2,1,10) 
        t_tissue = np.arange(10); t_aif = np.arange(10); Cp_aif = np.zeros(10); Cp_aif[1:3]=1.0
        fit_function = getattr(modeling, fit_function_name)

        results_serial = fit_function(Ct_data, t_tissue, t_aif, Cp_aif, initial_params=default_initial_params, bounds_params=default_bounds, num_processes=1)
        self.assertEqual(mock_worker.call_count, 4) 
        for i, key in enumerate(param_keys_expected):
            self.assertAlmostEqual(results_serial[key][0,0,0], 0*0.1 + 0*0.01 + i*0.001)
        
        mock_worker.reset_mock()
        with patch('multiprocessing.Pool') as mock_pool_constructor:
            mock_pool_instance = MagicMock()
            simulated_map_results = []
            for x in range(2):
                for y in range(2):
                    for z in range(1): 
                        args_tuple_sim = ((x,y,z), Ct_data[x,y,z,:], t_tissue, t_aif, Cp_aif, model_name_in_worker, default_initial_params, default_bounds)
                        simulated_map_results.append(side_effect_func(args_tuple_sim))
            mock_pool_instance.map.return_value = simulated_map_results
            mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance
            results_parallel = fit_function(Ct_data, t_tissue, t_aif, Cp_aif, initial_params=default_initial_params, bounds_params=default_bounds, num_processes=2)
            if len(Ct_data.reshape(-1, Ct_data.shape[-1])) > 1: 
                 mock_pool_constructor.assert_called_with(processes=2)
                 mock_pool_instance.map.assert_called_once()
            for key in param_keys_expected: np.testing.assert_array_almost_equal(results_parallel[key], results_serial[key])

    def test_fit_standard_tofts_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(fit_function_name="fit_standard_tofts_voxelwise", model_name_in_worker="Standard Tofts", param_keys_expected=["Ktrans", "ve"], default_initial_params=(0.1,0.2), default_bounds=([0,0],[1,1]))
    def test_fit_extended_tofts_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(fit_function_name="fit_extended_tofts_voxelwise", model_name_in_worker="Extended Tofts", param_keys_expected=["Ktrans", "ve", "vp"], default_initial_params=(0.1,0.2,0.05), default_bounds=([0,0,0],[1,1,1]))
    def test_fit_patlak_model_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(fit_function_name="fit_patlak_model_voxelwise", model_name_in_worker="Patlak", param_keys_expected=["Ktrans_patlak", "vp_patlak"], default_initial_params=(0.05,0.05), default_bounds=([0,0],[1,0.5]))
    def test_fit_2cxm_model_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(fit_function_name="fit_2cxm_model_voxelwise", model_name_in_worker="2CXM", param_keys_expected=["Fp_2cxm", "PS_2cxm", "vp_2cxm", "ve_2cxm"], default_initial_params=(0.1,0.05,0.05,0.1), default_bounds=([0,0,1e-3,1e-3],[2,1,0.5,0.7]))


if __name__ == '__main__':
    unittest.main()
