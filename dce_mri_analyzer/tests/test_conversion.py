import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises_regex
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import conversion

class TestSignalToConcentration(unittest.TestCase):
    def setUp(self):
        """Prepare common test data for 4D signal_to_concentration."""
        self.dce_shape = (2, 2, 2, 10)  # Small x,y,z for manageable data
        self.spatial_shape = self.dce_shape[:3]
        
        self.t10_map_data = np.full(self.spatial_shape, 1.0)  # T10 of 1s
        self.r1 = 4.5  # s^-1 mM^-1
        self.TR = 0.005  # 5 ms (seconds)
        
        self.dce_series_data = np.ones(self.dce_shape) * 100  # Baseline signal 100
        self.dce_series_data[..., 5:] *= 1.5 
        self.baseline_time_points = 5

    def test_conversion_basic_case(self):
        """Test basic signal to concentration conversion for 4D data."""
        Ct_data = conversion.signal_to_concentration(
            self.dce_series_data, 
            self.t10_map_data, 
            self.r1, 
            self.TR, 
            baseline_time_points=self.baseline_time_points
        )
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        S_pre_expected = 100.0
        R1_0_expected = 1.0 / self.t10_map_data[0,0,0] 
        assert_array_almost_equal(Ct_data[0,0,0,:self.baseline_time_points], 0, decimal=5)
        S_t_enhanced = 150.0
        signal_ratio_term_enhanced = S_t_enhanced / S_pre_expected
        E1_0_term = (1.0 - np.exp(-self.TR * R1_0_expected)) 
        log_arg_enhanced = 1.0 - (signal_ratio_term_enhanced * E1_0_term) 
        R1_t_enhanced = (-1.0 / self.TR) * np.log(log_arg_enhanced) 
        delta_R1_t_enhanced = R1_t_enhanced - R1_0_expected 
        Ct_t_enhanced_expected = delta_R1_t_enhanced / self.r1 
        assert_array_almost_equal(Ct_data[0,0,0,self.baseline_time_points:], Ct_t_enhanced_expected, decimal=5)

    def test_input_validation_dimensions(self):
        """Test input validation for incorrect data dimensions for 4D data."""
        with assert_raises_regex(self, ValueError, "dce_series_data must be a 4D NumPy array."):
            conversion.signal_to_concentration(np.zeros((2,2,10)), self.t10_map_data, self.r1, self.TR)
        with assert_raises_regex(self, ValueError, "t10_map_data must be a 3D NumPy array."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((2,2,2,10)), self.r1, self.TR)
        with assert_raises_regex(self, ValueError, "Spatial dimensions of dce_series_data .* must match t10_map_data."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((3,3,3)), self.r1, self.TR)

    def test_input_validation_parameters(self):
        """Test input validation for incorrect parameter values for 4D data."""
        with assert_raises_regex(self, ValueError, "TR must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, 0)
        with assert_raises_regex(self, ValueError, "r1_relaxivity must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, 0, self.TR)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=0)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.dce_shape[3])

    def test_zero_t10_handling(self):
        """Test handling of T10=0 for 4D data."""
        t10_map_data_with_zero = self.t10_map_data.copy(); t10_map_data_with_zero[0,0,0] = 0
        Ct_data = conversion.signal_to_concentration(self.dce_series_data, t10_map_data_with_zero, self.r1, self.TR, baseline_time_points=self.baseline_time_points)
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        self.assertTrue(np.isfinite(Ct_data[0,0,0,0]))

    def test_zero_S_pre_handling(self):
        """Test handling of S_pre=0 for 4D data."""
        dce_data_zero_baseline = np.zeros_like(self.dce_series_data)
        dce_data_zero_baseline[..., self.baseline_time_points:] = 50 
        Ct_data = conversion.signal_to_concentration(dce_data_zero_baseline, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.baseline_time_points)
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        self.assertTrue(np.all(np.isfinite(Ct_data)))

    def test_log_arg_clipping(self):
        """Test log_arg clipping for 4D data."""
        dce_high_signal = self.dce_series_data.copy()
        dce_high_signal[0,0,0,:self.baseline_time_points] = 10 
        dce_high_signal[0,0,0,self.baseline_time_points] = 10 * 2100 
        t10_for_clipping_test = self.t10_map_data.copy(); t10_for_clipping_test[0,0,0] = 10.0
        Ct_data = conversion.signal_to_concentration(dce_high_signal, t10_for_clipping_test, self.r1, self.TR, baseline_time_points=self.baseline_time_points)
        self.assertTrue(np.isfinite(Ct_data[0,0,0,self.baseline_time_points]))
        self.assertGreater(Ct_data[0,0,0,self.baseline_time_points], 0) 

class TestSignalTcToConcentrationTc(unittest.TestCase):
    def setUp(self):
        """Prepare common test data for 1D signal_tc_to_concentration_tc."""
        self.num_time_points = 10
        self.signal_tc = np.ones(self.num_time_points) * 100  # Baseline signal 100
        self.signal_tc[5:] *= 1.5  # Signal increase after 5th timepoint
        self.t10_scalar = 1.0  # T10 of 1s
        self.r1_scalar = 4.5  # s^-1 mM^-1
        self.TR_scalar = 0.005  # 5 ms (seconds)
        self.baseline_pts_tc = 5

    def test_tc_conversion_basic(self):
        """Test basic signal to concentration conversion for 1D TC data."""
        Ct_tc = conversion.signal_tc_to_concentration_tc(
            self.signal_tc, self.t10_scalar, self.r1_scalar, self.TR_scalar, self.baseline_pts_tc
        )
        self.assertEqual(Ct_tc.shape, self.signal_tc.shape)
        
        S_pre_tc_expected = 100.0
        R1_0_tc_expected = 1.0 / self.t10_scalar
        
        # Baseline points
        assert_array_almost_equal(Ct_tc[:self.baseline_pts_tc], 0, decimal=5)
        
        # Enhanced points
        S_t_enhanced = 150.0
        signal_ratio_term_enhanced = S_t_enhanced / S_pre_tc_expected
        E1_0_term = (1.0 - np.exp(-self.TR_scalar * R1_0_tc_expected))
        log_arg_enhanced = 1.0 - (signal_ratio_term_enhanced * E1_0_term)
        R1_t_enhanced = (-1.0 / self.TR_scalar) * np.log(log_arg_enhanced)
        delta_R1_t_enhanced = R1_t_enhanced - R1_0_tc_expected
        Ct_t_enhanced_expected = delta_R1_t_enhanced / self.r1_scalar
        
        assert_array_almost_equal(Ct_tc[self.baseline_pts_tc:], Ct_t_enhanced_expected, decimal=5)

    def test_tc_conversion_input_validation(self):
        """Test input validation for 1D TC conversion."""
        # Non-1D signal_tc
        with assert_raises_regex(self, ValueError, "signal_tc must be a 1D NumPy array."):
            conversion.signal_tc_to_concentration_tc(np.zeros((2,2)), self.t10_scalar, self.r1_scalar, self.TR_scalar)
        # T10 <= 0
        with assert_raises_regex(self, ValueError, "t10_scalar must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc, 0, self.r1_scalar, self.TR_scalar)
        # r1 <= 0
        with assert_raises_regex(self, ValueError, "r1_relaxivity must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc, self.t10_scalar, 0, self.TR_scalar)
        # TR <= 0
        with assert_raises_regex(self, ValueError, "TR must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc, self.t10_scalar, self.r1_scalar, 0)
        # baseline_time_points <= 0
        with assert_raises_regex(self, ValueError, "baseline_time_points must be a positive integer."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc, self.t10_scalar, self.r1_scalar, self.TR_scalar, 0)
        # baseline_time_points >= len(signal_tc)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_tc_to_concentration_tc(self.signal_tc, self.t10_scalar, self.r1_scalar, self.TR_scalar, len(self.signal_tc))
        # Empty signal_tc
        with assert_raises_regex(self, ValueError, "signal_tc cannot be empty."):
            conversion.signal_tc_to_concentration_tc(np.array([]), self.t10_scalar, self.r1_scalar, self.TR_scalar, 1)


    def test_tc_conversion_edge_cases(self):
        """Test edge cases for 1D TC conversion."""
        # Zero baseline signal
        signal_tc_zero_baseline = np.zeros_like(self.signal_tc)
        signal_tc_zero_baseline[self.baseline_pts_tc:] = 50 # Some post-baseline signal
        
        Ct_tc_zero_baseline = conversion.signal_tc_to_concentration_tc(
            signal_tc_zero_baseline, self.t10_scalar, self.r1_scalar, self.TR_scalar, self.baseline_pts_tc
        )
        self.assertTrue(np.all(np.isfinite(Ct_tc_zero_baseline)))
        # Expected value for baseline where S_t=0 and S_pre=epsilon: Ct = (-R1_0)/r1
        expected_ct_baseline = (- (1.0/self.t10_scalar)) / self.r1_scalar
        assert_array_almost_equal(Ct_tc_zero_baseline[:self.baseline_pts_tc], expected_ct_baseline, decimal=5)


        # Clipping of log_arg
        signal_tc_high = np.ones_like(self.signal_tc) * 10 # S_pre will be 10
        # Make S_t/S_pre very large to force log_arg to be negative before clipping
        # R1_0 = 1/1 = 1. TR = 0.005. E1_0_term = 1 - exp(-0.005) ~ 0.004987
        # We need S_t/S_pre * E1_0_term > 1.  S_t/S_pre > 1/0.004987 ~ 200.5
        signal_tc_high[self.baseline_pts_tc] = 10 * 210 # S_t/S_pre = 210
        
        t10_for_clip = 1.0 # R1_0 = 1.0
        Ct_tc_high_signal = conversion.signal_tc_to_concentration_tc(
            signal_tc_high, t10_for_clip, self.r1_scalar, self.TR_scalar, self.baseline_pts_tc
        )
        self.assertTrue(np.isfinite(Ct_tc_high_signal[self.baseline_pts_tc]))
        # R1_t = -ln(1e-9)/TR = 20.723 / 0.005 ~ 4144.6
        # delta_R1 = 4144.6 - 1.0 = 4143.6
        # Ct = 4143.6 / 4.5 ~ 920.8
        self.assertAlmostEqual(Ct_tc_high_signal[self.baseline_pts_tc], 920.8, places=1)


if __name__ == '__main__':
    unittest.main()
