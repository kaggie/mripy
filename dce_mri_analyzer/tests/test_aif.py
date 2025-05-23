import unittest
import numpy as np
import os
import tempfile
import shutil
import csv
from unittest.mock import patch, MagicMock

# Add project root for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import aif
from core import conversion # Needed for test_extract_aif_from_roi

class TestAifFunctions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_aif_from_file_csv_success(self):
        """Test successful loading of AIF from a CSV file."""
        filepath = os.path.join(self.test_dir, "test_aif.csv")
        times = np.array([0, 1, 2, 3], dtype=float)
        concentrations = np.array([0.0, 0.1, 0.2, 0.15], dtype=float)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Concentration (mM)"]) # Header
            for t, c in zip(times, concentrations):
                writer.writerow([t, c])
        
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_success(self):
        """Test successful loading of AIF from a TXT file (tab-separated)."""
        filepath = os.path.join(self.test_dir, "test_aif.txt")
        times = np.array([10, 20, 30], dtype=float)
        concentrations = np.array([0.5, 1.5, 1.0], dtype=float)
        with open(filepath, 'w') as f:
            f.write("Time\tConcentration\n") # Header
            for t, c in zip(times, concentrations):
                f.write(f"{t}\t{c}\n")
        
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_space_separated_success(self):
        """Test successful loading from a TXT file (space-separated)."""
        filepath = os.path.join(self.test_dir, "test_aif_space.txt")
        times = np.array([0, 5, 10, 15], dtype=float)
        concentrations = np.array([0.0, 0.2, 0.4, 0.3], dtype=float)
        with open(filepath, 'w') as f:
            f.write("TimeConcentration\n") # No real delimiter in header
            for t, c in zip(times, concentrations):
                f.write(f"{t} {c}\n")
        
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)


    def test_load_aif_from_file_not_found(self):
        """Test FileNotFoundError for non-existent AIF file."""
        filepath = os.path.join(self.test_dir, "non_existent_aif.txt")
        with self.assertRaises(FileNotFoundError):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_bad_format_columns(self):
        """Test ValueError for AIF file with incorrect number of columns."""
        filepath = os.path.join(self.test_dir, "bad_format_cols.txt")
        with open(filepath, 'w') as f:
            f.write("Time\tConcentration\tExtra\n")
            f.write("0\t0.1\t0.05\n")
        with self.assertRaisesRegex(ValueError, "Expected 2 columns"):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_bad_format_data(self):
        """Test ValueError for AIF file with non-numeric data."""
        filepath = os.path.join(self.test_dir, "bad_format_data.txt")
        with open(filepath, 'w') as f:
            f.write("Time\tConcentration\n")
            f.write("0\ttext_not_number\n")
        with self.assertRaisesRegex(ValueError, "Non-numeric data found"):
            aif.load_aif_from_file(filepath)
            
    def test_load_aif_from_file_empty(self):
        """Test ValueError for empty AIF file."""
        filepath = os.path.join(self.test_dir, "empty_aif.txt")
        with open(filepath, 'w') as f:
            pass # Empty file
        with self.assertRaisesRegex(ValueError, "AIF file is empty"):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_header_only(self):
        """Test ValueError for AIF file with only a header."""
        filepath = os.path.join(self.test_dir, "header_only_aif.txt")
        with open(filepath, 'w') as f:
            f.write("Time\tConcentration\n")
        with self.assertRaisesRegex(ValueError, "No numeric data found"): # Specific error depends on parsing path
            aif.load_aif_from_file(filepath)


    def test_parker_aif(self):
        """Test Parker AIF generation for specific time points."""
        time_points = np.array([0, 0.1, 0.5, 1.0]) # Example time points in minutes
        # Parker parameters: D=1.0, A1=0.809, m1=0.171, A2=0.330, m2=2.05
        # Cp(t) = D * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))
        # t=0: Cp(0) = 1.0 * (0.809 * 1 + 0.330 * 1) = 1.139
        # t=0.1: Cp(0.1) = 1.0 * (0.809 * exp(-0.171*0.1) + 0.330 * exp(-2.05*0.1)) 
        #              = (0.809 * 0.983047) + (0.330 * 0.81465) 
        #              = 0.79528 + 0.26883 = 1.06411
        expected_concs = np.array([
            1.139, 
            (0.809 * np.exp(-0.171*0.1) + 0.330 * np.exp(-2.05*0.1)),
            (0.809 * np.exp(-0.171*0.5) + 0.330 * np.exp(-2.05*0.5)),
            (0.809 * np.exp(-0.171*1.0) + 0.330 * np.exp(-2.05*1.0))
        ])
        generated_concs = aif.parker_aif(time_points)
        np.testing.assert_array_almost_equal(generated_concs, expected_concs, decimal=5)

    def test_generate_population_aif(self):
        """Test generation of population AIFs."""
        time_points = np.array([0, 1, 2])
        # Test Parker AIF
        parker_concs = aif.generate_population_aif("parker", time_points)
        expected_parker = aif.parker_aif(time_points)
        np.testing.assert_array_almost_equal(parker_concs, expected_parker)

        # Test non-existent AIF
        non_existent_aif = aif.generate_population_aif("non_existent_model", time_points)
        self.assertIsNone(non_existent_aif)

    @patch('core.conversion.signal_tc_to_concentration_tc') # Mock the conversion function
    def test_extract_aif_from_roi(self, mock_signal_tc_to_concentration_tc):
        """Test AIF extraction from ROI, focusing on ROI averaging and time vector."""
        # Define mock return value for the conversion
        mock_expected_concentration_tc = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_signal_tc_to_concentration_tc.return_value = mock_expected_concentration_tc

        dce_data_shape = (10, 10, 5, 5) # X, Y, Z, Time
        dce_4d_data = np.ones(dce_data_shape) 
        # Make a gradient in the ROI to check averaging
        # ROI: x=2-3, y=2-3, z=1. Signal values:
        # dce_4d_data[2,2,1,:] = 10
        # dce_4d_data[3,2,1,:] = 20
        # dce_4d_data[2,3,1,:] = 30
        # dce_4d_data[3,3,1,:] = 40
        # Mean signal in ROI should be (10+20+30+40)/4 = 25 for all time points
        dce_4d_data[2,2,1,:] = 10
        dce_4d_data[3,2,1,:] = 20
        dce_4d_data[2,3,1,:] = 30
        dce_4d_data[3,3,1,:] = 40
        
        expected_mean_signal_tc = np.full(dce_data_shape[3], 25.0)

        roi_coords = (2, 2, 2, 2) # x_start, y_start, width, height
        slice_index_z = 1
        t10_blood = 1.4
        r1_blood = 4.5
        TR = 0.005 # seconds
        baseline_pts = 2

        aif_time_tc, aif_concentration_tc = aif.extract_aif_from_roi(
            dce_4d_data, roi_coords, slice_index_z, t10_blood, r1_blood, TR, baseline_pts
        )

        # Check that the mock was called with the correct mean signal
        mock_signal_tc_to_concentration_tc.assert_called_once()
        call_args = mock_signal_tc_to_concentration_tc.call_args[0] # Get positional args
        called_signal_tc = call_args[0]
        np.testing.assert_array_almost_equal(called_signal_tc, expected_mean_signal_tc)
        self.assertEqual(call_args[1], t10_blood)
        self.assertEqual(call_args[2], r1_blood)
        self.assertEqual(call_args[3], TR)
        self.assertEqual(call_args[4], baseline_pts)


        # Check returned time vector
        expected_time_vector = np.arange(dce_data_shape[3]) * TR
        np.testing.assert_array_almost_equal(aif_time_tc, expected_time_vector)

        # Check returned concentration (this comes from the mock)
        np.testing.assert_array_almost_equal(aif_concentration_tc, mock_expected_concentration_tc)

    def test_extract_aif_from_roi_value_errors(self):
        """Test ValueError for invalid ROI coordinates in extract_aif_from_roi."""
        dce_4d_data = np.ones((5,5,3,10))
        with self.assertRaisesRegex(ValueError, "ROI start coordinates or Z-slice index out of bounds"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,1,1), 5, 1.4, 4.5, 0.005) # Z too large
        with self.assertRaisesRegex(ValueError, "ROI dimensions .* exceed DCE data spatial bounds"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,6,1), 0, 1.4, 4.5, 0.005) # Width too large
        with self.assertRaisesRegex(ValueError, "ROI width and height must be positive"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,0,1), 0, 1.4, 4.5, 0.005) # Width is zero


if __name__ == '__main__':
    unittest.main()
