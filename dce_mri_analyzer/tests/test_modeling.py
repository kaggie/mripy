import unittest
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import modeling

class TestModelingFunctions(unittest.TestCase):
    def setUp(self):
        # Time vector for AIF and tissue
        self.t_points = np.linspace(0, 60, 61) # 61 points, 1 per second for 1 minute
        
        # Sample AIF (simple boxcar for testing, not realistic)
        self.Cp_aif = np.zeros_like(self.t_points)
        self.Cp_aif[5:15] = 1.0 # Bolus from t=5 to t=14
        
        self.Cp_interp_func = interp1d(self.t_points, self.Cp_aif, kind='linear', bounds_error=False, fill_value=0.0)
        
        # Integral of Cp for Patlak
        self.integral_Cp_dt_aif = cumtrapz(self.Cp_aif, self.t_points, initial=0)
        self.integral_Cp_dt_interp_func = interp1d(self.t_points, self.integral_Cp_dt_aif, kind='linear', bounds_error=False, fill_value=0.0)

    def test_patlak_model_function(self):
        """Test the Patlak model function with known inputs."""
        Ktrans_patlak = 0.1 
        vp_patlak = 0.05    
        test_times = np.array([0, 10, 20])
        expected_Ct = np.array([
            Ktrans_patlak * self.integral_Cp_dt_interp_func(0) + vp_patlak * self.Cp_interp_func(0),
            Ktrans_patlak * self.integral_Cp_dt_interp_func(10) + vp_patlak * self.Cp_interp_func(10),
            Ktrans_patlak * self.integral_Cp_dt_interp_func(20) + vp_patlak * self.Cp_interp_func(20),
        ])
        calculated_Ct = modeling.patlak_model(test_times, Ktrans_patlak, vp_patlak, 
                                              self.Cp_interp_func, self.integral_Cp_dt_interp_func)
        np.testing.assert_array_almost_equal(calculated_Ct, expected_Ct, decimal=5)

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

    @patch('core.modeling.fit_patlak_model')
    @patch('core.modeling.fit_extended_tofts')
    @patch('core.modeling.fit_standard_tofts')
    def test_fit_voxel_worker_model_calls(self, mock_fit_std, mock_fit_ext, mock_fit_patlak):
        """Test that _fit_voxel_worker calls the correct model fitting logic."""
        voxel_idx = (0,0,0)
        Ct_voxel = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) 
        t_tissue = np.arange(5)
        t_aif_worker = np.arange(6)
        Cp_aif_worker = np.zeros(6); Cp_aif_worker[1:3]=1.0 # Dummy AIF

        # Test Standard Tofts
        mock_fit_std.return_value = ((0.2, 0.3), np.array([1,2,3])) # Ktrans, ve
        args_std = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.1), ([0,0],[1,1]))
        _, _, res_std = modeling._fit_voxel_worker(args_std)
        mock_fit_std.assert_called_once()
        self.assertEqual(res_std.get("Ktrans"), 0.2)
        self.assertEqual(res_std.get("ve"), 0.3)

        # Test Extended Tofts
        mock_fit_ext.return_value = ((0.25, 0.35, 0.05), np.array([1,2,3])) # Ktrans, ve, vp
        args_ext = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Extended Tofts", (0.1,0.1,0.01), ([0,0,0],[1,1,1]))
        _, _, res_ext = modeling._fit_voxel_worker(args_ext)
        mock_fit_ext.assert_called_once()
        self.assertEqual(res_ext.get("Ktrans"), 0.25)
        self.assertEqual(res_ext.get("vp"), 0.05)
        
        # Test Patlak
        mock_fit_patlak.return_value = ((0.1, 0.05), np.array([1,2,3])) # Ktrans_patlak, vp_patlak
        args_patlak = (voxel_idx, Ct_voxel, t_tissue, t_aif_worker, Cp_aif_worker, "Patlak", (0.05,0.02), ([0,0],[0.3,0.2]))
        _, _, res_patlak = modeling._fit_voxel_worker(args_patlak)
        mock_fit_patlak.assert_called_once() 
        self.assertEqual(res_patlak.get("Ktrans_patlak"), 0.1)
        self.assertEqual(res_patlak.get("vp_patlak"), 0.05)


    def test_fit_voxel_worker_skip_nan_data(self):
        """Test _fit_voxel_worker skipping NaN data."""
        voxel_idx = (1,1,1); Ct_voxel_nan = np.full(5, np.nan); t_tissue = np.arange(5)
        t_aif_worker = np.arange(6); Cp_aif_worker = np.zeros(6)
        args_tuple = (voxel_idx, Ct_voxel_nan, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.2), ([0,0],[1,1]))
        idx_out, model_out, result_dict = modeling._fit_voxel_worker(args_tuple)
        self.assertEqual(idx_out, voxel_idx); self.assertIn("error", result_dict); self.assertIn("Skipped", result_dict["error"])
        
    def test_fit_voxel_worker_skip_insufficient_data(self):
        """Test _fit_voxel_worker skipping data with too few valid points."""
        voxel_idx = (1,1,1); Ct_voxel_short = np.array([0.1, np.nan, 0.3, np.nan, np.nan]); t_tissue = np.arange(5)
        t_aif_worker = np.arange(6); Cp_aif_worker = np.zeros(6)
        args_tuple = (voxel_idx, Ct_voxel_short, t_tissue, t_aif_worker, Cp_aif_worker, "Standard Tofts", (0.1,0.2), ([0,0],[1,1]))
        idx_out, model_out, result_dict = modeling._fit_voxel_worker(args_tuple)
        self.assertEqual(idx_out, voxel_idx); self.assertIn("error", result_dict); self.assertIn("insufficient valid data points", result_dict["error"])

    @patch('core.modeling._fit_voxel_worker') 
    def _run_voxelwise_parallel_test(self, mock_worker, fit_function_name, model_name_in_worker, param_keys_expected):
        """Helper to test voxel-wise parallel calls for different models."""
        def side_effect_func(args):
            idx, _, _, _, _, model_name_call, _, _ = args
            if model_name_call == model_name_in_worker:
                # Return a dict with keys matching param_keys_expected
                # Values are simple functions of index to make them unique and predictable
                return_params = {key: idx[0]*0.1 + idx[1]*0.01 + (param_keys_expected.index(key)*0.001) for key in param_keys_expected}
                return idx, model_name_call, return_params
            return idx, model_name_call, {"error": "Wrong model for mock"}
        mock_worker.side_effect = side_effect_func

        Ct_data = np.random.rand(2,2,1,10) 
        t_tissue = np.arange(10); t_aif = np.arange(10); Cp_aif = np.zeros(10); Cp_aif[1:3]=1.0
        
        fit_function = getattr(modeling, fit_function_name)

        # Test with num_processes=1 (serial path)
        results_serial = fit_function(Ct_data, t_tissue, t_aif, Cp_aif, num_processes=1)
        self.assertEqual(mock_worker.call_count, 4) 
        for i, key in enumerate(param_keys_expected):
            self.assertAlmostEqual(results_serial[key][0,0,0], 0*0.1 + 0*0.01 + i*0.001)
            self.assertAlmostEqual(results_serial[key][0,1,0], 0*0.1 + 1*0.01 + i*0.001)
        
        mock_worker.reset_mock()
        with patch('multiprocessing.Pool') as mock_pool_constructor:
            mock_pool_instance = MagicMock()
            simulated_map_results = []
            for x in range(2):
                for y in range(2):
                    for z in range(1): 
                        # Use dummy params for initial/bounds as they are not used by the mock worker here
                        args_tuple_sim = ((x,y,z), Ct_data[x,y,z,:], t_tissue, t_aif, Cp_aif, model_name_in_worker, (), ((),()))
                        simulated_map_results.append(side_effect_func(args_tuple_sim))
            
            mock_pool_instance.map.return_value = simulated_map_results
            mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance

            results_parallel = fit_function(Ct_data, t_tissue, t_aif, Cp_aif, num_processes=2)
            if len(Ct_data.reshape(-1, Ct_data.shape[-1])) > 1: # If more than one voxel to process
                 mock_pool_constructor.assert_called_with(processes=2)
                 mock_pool_instance.map.assert_called_once()
            
            for key in param_keys_expected:
                np.testing.assert_array_almost_equal(results_parallel[key], results_serial[key])

    def test_fit_standard_tofts_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(
            fit_function_name="fit_standard_tofts_voxelwise",
            model_name_in_worker="Standard Tofts",
            param_keys_expected=["Ktrans", "ve"]
        )

    def test_fit_extended_tofts_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(
            fit_function_name="fit_extended_tofts_voxelwise",
            model_name_in_worker="Extended Tofts",
            param_keys_expected=["Ktrans", "ve", "vp"]
        )

    def test_fit_patlak_model_voxelwise_parallel_calls(self):
        self._run_voxelwise_parallel_test(
            fit_function_name="fit_patlak_model_voxelwise",
            model_name_in_worker="Patlak",
            param_keys_expected=["Ktrans_patlak", "vp_patlak"]
        )

if __name__ == '__main__':
    unittest.main()
