import unittest
import numpy as np
import os
import tempfile
import shutil
import csv
import json # For TestAifRoiSaveLoad
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
            f.write("TimeConcentration\n") 
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
        with open(filepath, 'w') as f: pass 
        with self.assertRaisesRegex(ValueError, "AIF file is empty"):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_header_only(self):
        """Test ValueError for AIF file with only a header."""
        filepath = os.path.join(self.test_dir, "header_only_aif.txt")
        with open(filepath, 'w') as f: f.write("Time\tConcentration\n")
        with self.assertRaisesRegex(ValueError, "No numeric data found"): 
            aif.load_aif_from_file(filepath)


    def test_parker_aif(self):
        """Test Parker AIF generation for specific time points."""
        time_points = np.array([0, 0.1, 0.5, 1.0]) 
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
        parker_concs = aif.generate_population_aif("parker", time_points)
        expected_parker = aif.parker_aif(time_points)
        np.testing.assert_array_almost_equal(parker_concs, expected_parker)
        non_existent_aif = aif.generate_population_aif("non_existent_model", time_points)
        self.assertIsNone(non_existent_aif)

    @patch('core.conversion.signal_tc_to_concentration_tc') 
    def test_extract_aif_from_roi(self, mock_signal_tc_to_concentration_tc):
        """Test AIF extraction from ROI, focusing on ROI averaging and time vector."""
        mock_expected_concentration_tc = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_signal_tc_to_concentration_tc.return_value = mock_expected_concentration_tc
        dce_data_shape = (10, 10, 5, 5); dce_4d_data = np.ones(dce_data_shape) 
        dce_4d_data[2,2,1,:] = 10; dce_4d_data[3,2,1,:] = 20
        dce_4d_data[2,3,1,:] = 30; dce_4d_data[3,3,1,:] = 40
        expected_mean_signal_tc = np.full(dce_data_shape[3], 25.0)
        roi_coords = (2, 2, 2, 2); slice_index_z = 1
        t10_blood = 1.4; r1_blood = 4.5; TR = 0.005; baseline_pts = 2

        aif_time_tc, aif_concentration_tc = aif.extract_aif_from_roi(
            dce_4d_data, roi_coords, slice_index_z, t10_blood, r1_blood, TR, baseline_pts
        )
        mock_signal_tc_to_concentration_tc.assert_called_once()
        call_args = mock_signal_tc_to_concentration_tc.call_args[0] 
        called_signal_tc = call_args[0]
        np.testing.assert_array_almost_equal(called_signal_tc, expected_mean_signal_tc)
        self.assertEqual(call_args[1], t10_blood); self.assertEqual(call_args[2], r1_blood)
        self.assertEqual(call_args[3], TR); self.assertEqual(call_args[4], baseline_pts)
        expected_time_vector = np.arange(dce_data_shape[3]) * TR
        np.testing.assert_array_almost_equal(aif_time_tc, expected_time_vector)
        np.testing.assert_array_almost_equal(aif_concentration_tc, mock_expected_concentration_tc)

    def test_extract_aif_from_roi_value_errors(self):
        """Test ValueError for invalid ROI coordinates in extract_aif_from_roi."""
        dce_4d_data = np.ones((5,5,3,10))
        with self.assertRaisesRegex(ValueError, "ROI start coordinates or Z-slice index out of bounds"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,1,1), 5, 1.4, 4.5, 0.005) 
        with self.assertRaisesRegex(ValueError, "ROI dimensions .* exceed DCE data spatial bounds"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,6,1), 0, 1.4, 4.5, 0.005) 
        with self.assertRaisesRegex(ValueError, "ROI width and height must be positive"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,0,1), 0, 1.4, 4.5, 0.005) 


class TestAifRoiSaveLoad(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_roi_props = {
            "slice_index": 10, 
            "pos_x": 20.5, "pos_y": 30.0,
            "size_w": 15.2, "size_h": 10.8,
            "image_ref_name": "Mean DCE"
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_load_aif_roi_definition(self):
        """Test saving and then loading an AIF ROI definition."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=self.test_dir, mode='w')
        tmp_file_path = tmp_file.name
        tmp_file.close() # Close it so save_aif_roi_definition can open it

        aif.save_aif_roi_definition(self.sample_roi_props, tmp_file_path)
        loaded_props = aif.load_aif_roi_definition(tmp_file_path)
        
        self.assertIsNotNone(loaded_props)
        self.assertEqual(loaded_props, self.sample_roi_props)
        os.remove(tmp_file_path)

    def test_load_aif_roi_definition_not_found(self):
        """Test FileNotFoundError when loading a non-existent ROI definition file."""
        non_existent_path = os.path.join(self.test_dir, "no_such_roi.json")
        with self.assertRaises(FileNotFoundError):
            aif.load_aif_roi_definition(non_existent_path)

    def test_load_aif_roi_definition_bad_json(self):
        """Test ValueError when loading a malformed JSON ROI definition file."""
        tmp_file_path = os.path.join(self.test_dir, "bad_roi.json")
        with open(tmp_file_path, 'w') as f:
            f.write("{'slice_index': 10, 'pos_x': 20.5, ...") # Malformed JSON
        
        with self.assertRaisesRegex(ValueError, "Error decoding JSON"):
            aif.load_aif_roi_definition(tmp_file_path)

    def test_load_aif_roi_definition_missing_keys(self):
        """Test ValueError when loading an ROI definition file with missing keys."""
        tmp_file_path = os.path.join(self.test_dir, "missing_keys_roi.json")
        bad_props = {"slice_index": 5, "pos_x": 10.0} # Missing other keys
        with open(tmp_file_path, 'w') as f:
            json.dump(bad_props, f)
        
        with self.assertRaisesRegex(ValueError, "Missing required key"):
            aif.load_aif_roi_definition(tmp_file_path)
            
    def test_load_aif_roi_definition_wrong_types(self):
        """Test ValueError when loading an ROI definition file with incorrect data types for keys."""
        tmp_file_path = os.path.join(self.test_dir, "wrong_types_roi.json")
        wrong_type_props = self.sample_roi_props.copy()
        wrong_type_props["slice_index"] = "not_an_integer" # Slice index should be int
        with open(tmp_file_path, 'w') as f:
            json.dump(wrong_type_props, f)
        
        with self.assertRaisesRegex(ValueError, "slice_index must be an integer"):
            aif.load_aif_roi_definition(tmp_file_path)

        wrong_type_props_2 = self.sample_roi_props.copy()
        wrong_type_props_2["pos_x"] = "not_a_float" # pos_x should be numeric
        with open(tmp_file_path, 'w') as f: # Overwrite
            json.dump(wrong_type_props_2, f)
        with self.assertRaisesRegex(ValueError, "ROI position/size values must be numeric"):
            aif.load_aif_roi_definition(tmp_file_path)


if __name__ == '__main__':
    unittest.main()
