import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import conversion

class TestSignalToConcentration(unittest.TestCase):
    def setUp(self):
        """Prepare common test data."""
        self.dce_shape = (2, 2, 2, 10)  # Small x,y,z for manageable data
        self.spatial_shape = self.dce_shape[:3]
        
        self.t10_map_data = np.full(self.spatial_shape, 1.0)  # T10 of 1s
        self.r1 = 4.5  # s^-1 mM^-1
        self.TR = 0.005  # 5 ms (seconds)
        
        self.dce_series_data = np.ones(self.dce_shape) * 100  # Baseline signal 100
        # Simulate signal increase: 50% increase after 5th timepoint (index 4)
        # So, indices 0,1,2,3,4 are baseline. Indices 5,6,7,8,9 are enhanced.
        self.dce_series_data[..., 5:] *= 1.5 
        self.baseline_time_points = 5 # Use first 5 points for baseline

    def test_conversion_basic_case(self):
        """Test basic signal to concentration conversion."""
        Ct_data = conversion.signal_to_concentration(
            self.dce_series_data, 
            self.t10_map_data, 
            self.r1, 
            self.TR, 
            baseline_time_points=self.baseline_time_points
        )
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)

        # --- Manual calculation for one voxel ---
        S_pre_expected = 100.0
        R1_0_expected = 1.0 / self.t10_map_data[0,0,0] # Should be 1.0

        # For baseline time points (e.g., t=2, index 1)
        # S_t = 100
        # signal_ratio_term = S_t / S_pre_expected = 100.0 / 100.0 = 1.0
        # E1_0_term = (1.0 - np.exp(-self.TR * R1_0_expected)) approx 0.00498752079
        # log_arg = 1.0 - signal_ratio_term * E1_0_term approx 1.0 - 1.0 * 0.00498752079 = 0.995012479
        # R1_t = (-1.0 / self.TR) * np.log(log_arg) approx (-1/0.005) * np.log(0.995012479) = 1.0
        # delta_R1_t = R1_t - R1_0_expected approx 1.0 - 1.0 = 0
        # Ct_t = delta_R1_t / self.r1 approx 0
        assert_array_almost_equal(Ct_data[0,0,0,:self.baseline_time_points], 0, decimal=5)

        # For enhanced time points (e.g., t=7, index 6)
        S_t_enhanced = 150.0
        signal_ratio_term_enhanced = S_t_enhanced / S_pre_expected # 1.5
        E1_0_term = (1.0 - np.exp(-self.TR * R1_0_expected)) # approx 0.00498752079
        log_arg_enhanced = 1.0 - (signal_ratio_term_enhanced * E1_0_term) # approx 1.0 - 1.5 * 0.00498752079 = 0.9925187188
        R1_t_enhanced = (-1.0 / self.TR) * np.log(log_arg_enhanced) # approx (-1/0.005) * log(0.9925187188) = 1.503768
        delta_R1_t_enhanced = R1_t_enhanced - R1_0_expected # approx 1.503768 - 1.0 = 0.503768
        Ct_t_enhanced_expected = delta_R1_t_enhanced / self.r1 # approx 0.503768 / 4.5 = 0.111948
        
        assert_array_almost_equal(Ct_data[0,0,0,self.baseline_time_points:], Ct_t_enhanced_expected, decimal=5)

    def test_input_validation_dimensions(self):
        """Test input validation for incorrect data dimensions."""
        with self.assertRaisesRegex(ValueError, "dce_series_data must be a 4D NumPy array."):
            conversion.signal_to_concentration(np.zeros((2,2,10)), self.t10_map_data, self.r1, self.TR)
        
        with self.assertRaisesRegex(ValueError, "t10_map_data must be a 3D NumPy array."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((2,2,2,10)), self.r1, self.TR)
        
        with self.assertRaisesRegex(ValueError, "Spatial dimensions of dce_series_data .* must match t10_map_data."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((3,3,3)), self.r1, self.TR)

    def test_input_validation_parameters(self):
        """Test input validation for incorrect parameter values."""
        with self.assertRaisesRegex(ValueError, "TR must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, 0)
        with self.assertRaisesRegex(ValueError, "TR must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, -1)

        with self.assertRaisesRegex(ValueError, "r1_relaxivity must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, 0, self.TR)
        with self.assertRaisesRegex(ValueError, "r1_relaxivity must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, -1, self.TR)

        with self.assertRaisesRegex(ValueError, "baseline_time_points must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=0)
        
        with self.assertRaisesRegex(ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.dce_shape[3])
        with self.assertRaisesRegex(ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.dce_shape[3]+1)

    def test_zero_t10_handling(self):
        """Test handling of T10=0 (relies on epsilon addition in main code)."""
        t10_map_data_with_zero = self.t10_map_data.copy()
        t10_map_data_with_zero[0,0,0] = 0 # Set one voxel's T10 to zero
        
        try:
            Ct_data = conversion.signal_to_concentration(
                self.dce_series_data, 
                t10_map_data_with_zero, 
                self.r1, 
                self.TR,
                baseline_time_points=self.baseline_time_points
            )
            self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
            # The voxel with T10=0 will have R1_0 approx 1e9.
            # S_pre = 100. E1_0_term will be approx 1.0.
            # Baseline S_t = 100. signal_ratio_term = 1.
            # log_arg = 1 - 1*1 = 0. Clipped to 1e-9. R1_t = -ln(1e-9)/TR approx 4144.
            # delta_R1_t = 4144 - 1e9, which is very negative. Ct will be very negative.
            self.assertTrue(np.isfinite(Ct_data[0,0,0,0])) # Should compute to a finite number.
            self.assertLess(Ct_data[0,0,0,0], -1e6) # Expect a large negative number
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised unexpectedly for T10=0 case.")

    def test_zero_S_pre_handling(self):
        """Test handling of S_pre=0 (relies on epsilon addition)."""
        dce_data_zero_baseline = np.zeros_like(self.dce_series_data)
        # Make some post-baseline signal to avoid all Ct being zero due to S_t=0
        dce_data_zero_baseline[..., self.baseline_time_points:] = 50 
        
        try:
            Ct_data = conversion.signal_to_concentration(
                dce_data_zero_baseline, 
                self.t10_map_data, 
                self.r1, 
                self.TR,
                baseline_time_points=self.baseline_time_points
            )
            self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
            # S_pre is 0 (+epsilon). For baseline points S_t=0, so signal_ratio_term=0, log_arg=1, R1_t=0, delta_R1_t = -R1_0, Ct = -R1_0/r1.
            # For post-baseline S_t=50. signal_ratio_term is large (50/1e-9).
            # log_arg = 1 - large_ratio * E1_0_term. This will be very negative.
            # Clipped to 1e-9. R1_t will be large positive (approx 4144 if E1_0_term leads to it).
            # Ct_t = (large_R1_t - R1_0) / r1. Expect large positive.
            expected_ct_baseline = (- (1.0/self.t10_map_data[0,0,0])) / self.r1 # around -1.0/4.5 = -0.222
            assert_array_almost_equal(Ct_data[0,0,0,:self.baseline_time_points], expected_ct_baseline, decimal=5)
            self.assertTrue(np.all(Ct_data[0,0,0,self.baseline_time_points:] > 0)) # Expect positive Ct for enhanced points
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised unexpectedly for S_pre=0 case.")


    def test_log_arg_clipping(self):
        """Test that log_arg clipping prevents math errors for np.log."""
        # We need (S_t / S_pre) * (1.0 - np.exp(-TR * R1_0)) >= 1
        # Let S_pre = 100 (from self.dce_series_data[0,0,0,0])
        # Let T10 = 10s (so R1_0 = 0.1)
        # TR = 0.005
        # E1_0_term = 1 - exp(-TR * R1_0) = 1 - exp(-0.005 * 0.1) = 1 - exp(-0.0005) approx 0.000499875
        # We need S_t/S_pre * E1_0_term >= 1  => S_t/S_pre >= 1/E1_0_term approx 1/0.000499875 approx 2000.5
        
        dce_high_signal = self.dce_series_data.copy()
        # S_pre is 100 for this voxel from setup.
        # Set S_t for one time point to be much larger than S_pre, making S_t/S_pre = 2100
        idx_x, idx_y, idx_z = 0, 0, 0
        time_point_high_signal = self.baseline_time_points # First post-baseline point
        
        dce_high_signal[idx_x, idx_y, idx_z, time_point_high_signal] = \
            self.dce_series_data[idx_x, idx_y, idx_z, 0] * 2100 # S_pre * 2100

        t10_for_clipping_test = self.t10_map_data.copy()
        t10_for_clipping_test[idx_x, idx_y, idx_z] = 10.0 # R1_0 = 0.1

        try:
            Ct_data = conversion.signal_to_concentration(
                dce_high_signal, 
                t10_for_clipping_test, 
                self.r1, 
                self.TR,
                baseline_time_points=self.baseline_time_points
            )
            # If log_arg was not clipped, np.log of a non-positive would raise error or warning.
            # The result for Ct_data[idx_x, idx_y, idx_z, time_point_high_signal] will be calculated
            # using the clipped log_arg (1e-9).
            # R1_t = (-1/TR) * np.log(1e-9) approx (-1/0.005) * (-20.7232658369) approx 4144.65
            # R1_0 = 0.1
            # delta_R1_t = 4144.65 - 0.1 = 4144.55
            # Ct = 4144.55 / 4.5 approx 920.01
            self.assertTrue(np.isfinite(Ct_data[idx_x, idx_y, idx_z, time_point_high_signal]))
            self.assertAlmostEqual(Ct_data[idx_x, idx_y, idx_z, time_point_high_signal], 920.01, places=1) 

        except FloatingPointError: # Catch potential warnings raised to errors by numpy
            self.fail("FloatingPointError (e.g. log of non-positive) raised unexpectedly.")


if __name__ == '__main__':
    unittest.main()
