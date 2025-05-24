import unittest
import numpy as np
import os
import tempfile
import shutil
import csv
import json 
from unittest.mock import patch, MagicMock

# Add project root for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import aif
from core import conversion 

class TestAifFunctions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Time in minutes for typical AIF parameter interpretation
        self.time_points_minutes = np.array([0, 0.1, 0.5, 1.0, 2.0, 5.0])


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_aif_from_file_csv_success(self):
        filepath = os.path.join(self.test_dir, "test_aif.csv")
        times = np.array([0, 1, 2, 3], dtype=float)
        concentrations = np.array([0.0, 0.1, 0.2, 0.15], dtype=float)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(["Time (s)", "Concentration (mM)"]) 
            for t, c in zip(times, concentrations): writer.writerow([t, c])
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_success(self):
        filepath = os.path.join(self.test_dir, "test_aif.txt")
        times = np.array([10, 20, 30], dtype=float)
        concentrations = np.array([0.5, 1.5, 1.0], dtype=float)
        with open(filepath, 'w') as f:
            f.write("Time\tConcentration\n"); 
            for t, c in zip(times, concentrations): f.write(f"{t}\t{c}\n")
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_space_separated_success(self):
        filepath = os.path.join(self.test_dir, "test_aif_space.txt")
        times = np.array([0, 5, 10, 15], dtype=float)
        concentrations = np.array([0.0, 0.2, 0.4, 0.3], dtype=float)
        with open(filepath, 'w') as f:
            f.write("TimeConcentration\n"); 
            for t, c in zip(times, concentrations): f.write(f"{t} {c}\n")
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_not_found(self):
        filepath = os.path.join(self.test_dir, "non_existent_aif.txt")
        with self.assertRaises(FileNotFoundError): aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_bad_format_columns(self):
        filepath = os.path.join(self.test_dir, "bad_format_cols.txt")
        with open(filepath, 'w') as f: f.write("Time\tConcentration\tExtra\n0\t0.1\t0.05\n")
        with self.assertRaisesRegex(ValueError, "Expected 2 columns"): aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_bad_format_data(self):
        filepath = os.path.join(self.test_dir, "bad_format_data.txt")
        with open(filepath, 'w') as f: f.write("Time\tConcentration\n0\ttext_not_number\n")
        with self.assertRaisesRegex(ValueError, "Non-numeric data found"): aif.load_aif_from_file(filepath)
            
    def test_load_aif_from_file_empty(self):
        filepath = os.path.join(self.test_dir, "empty_aif.txt"); 
        with open(filepath, 'w') as f: pass 
        with self.assertRaisesRegex(ValueError, "AIF file is empty"): aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_header_only(self):
        filepath = os.path.join(self.test_dir, "header_only_aif.txt"); 
        with open(filepath, 'w') as f: f.write("Time\tConcentration\n")
        with self.assertRaisesRegex(ValueError, "No numeric data found"): aif.load_aif_from_file(filepath)

    def test_save_aif_curve(self):
        """Test saving AIF curve to CSV and TXT files."""
        times = np.array([0, 1, 2, 3.5], dtype=float)
        concentrations = np.array([0.0, 0.15, 0.25, 0.1], dtype=float)

        for ext, delimiter in [(".csv", ","), (".txt", "\t")]:
            with self.subTest(extension=ext):
                tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=self.test_dir, mode='w')
                tmp_file_path = tmp_file.name
                tmp_file.close()

                aif.save_aif_curve(times, concentrations, tmp_file_path)

                with open(tmp_file_path, 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    header = next(reader)
                    self.assertEqual(header, ['Time', 'Concentration'])
                    loaded_data = list(reader)
                    self.assertEqual(len(loaded_data), len(times))
                    for i, row in enumerate(loaded_data):
                        self.assertAlmostEqual(float(row[0]), times[i])
                        self.assertAlmostEqual(float(row[1]), concentrations[i])
                os.remove(tmp_file_path)
        
        # Test error for unequal length arrays
        with self.assertRaises(ValueError):
            aif.save_aif_curve(np.array([1,2]), np.array([1,2,3]), "test.csv")


    def test_parker_aif(self):
        """Test Parker AIF generation for specific time points (time in minutes)."""
        expected_concs = np.array([
            1.0 * (0.809 * np.exp(-0.171*0.0) + 0.330 * np.exp(-2.05*0.0)), 
            1.0 * (0.809 * np.exp(-0.171*0.1) + 0.330 * np.exp(-2.05*0.1)), 
            1.0 * (0.809 * np.exp(-0.171*0.5) + 0.330 * np.exp(-2.05*0.5)), 
            1.0 * (0.809 * np.exp(-0.171*1.0) + 0.330 * np.exp(-2.05*1.0)), 
            1.0 * (0.809 * np.exp(-0.171*2.0) + 0.330 * np.exp(-2.05*2.0)), 
            1.0 * (0.809 * np.exp(-0.171*5.0) + 0.330 * np.exp(-2.05*5.0)), 
        ])
        generated_concs = aif.parker_aif(self.time_points_minutes)
        np.testing.assert_array_almost_equal(generated_concs, expected_concs, decimal=5)

    def test_weinmann_aif(self):
        """Test Weinmann AIF generation for specific time points (time in minutes)."""
        expected_concs = np.array([
            1.0 * (3.99 * np.exp(-0.144*0.0) + 4.78 * np.exp(-0.0111*0.0)), 
            1.0 * (3.99 * np.exp(-0.144*0.1) + 4.78 * np.exp(-0.0111*0.1)), 
            1.0 * (3.99 * np.exp(-0.144*0.5) + 4.78 * np.exp(-0.0111*0.5)), 
            1.0 * (3.99 * np.exp(-0.144*1.0) + 4.78 * np.exp(-0.0111*1.0)), 
            1.0 * (3.99 * np.exp(-0.144*2.0) + 4.78 * np.exp(-0.0111*2.0)), 
            1.0 * (3.99 * np.exp(-0.144*5.0) + 4.78 * np.exp(-0.0111*5.0)), 
        ])
        generated_concs = aif.weinmann_aif(self.time_points_minutes)
        np.testing.assert_array_almost_equal(generated_concs, expected_concs, decimal=5)

    def test_fast_biexponential_aif(self):
        """Test Fast Bi-exponential AIF generation (time in minutes)."""
        # Default params: D_scaler=1.0, A1=0.6, m1=3.0, A2=0.4, m2=0.3
        expected_concs = np.array([
            1.0 * (0.6 * np.exp(-3.0*0.0) + 0.4 * np.exp(-0.3*0.0)), # t=0
            1.0 * (0.6 * np.exp(-3.0*0.1) + 0.4 * np.exp(-0.3*0.1)), # t=0.1
            1.0 * (0.6 * np.exp(-3.0*0.5) + 0.4 * np.exp(-0.3*0.5)), # t=0.5
            1.0 * (0.6 * np.exp(-3.0*1.0) + 0.4 * np.exp(-0.3*1.0)), # t=1.0
        ])
        generated_concs = aif.fast_biexponential_aif(self.time_points_minutes[:4]) # Use first 4 points
        np.testing.assert_array_almost_equal(generated_concs, expected_concs, decimal=5)

        # Test with custom parameters
        custom_params = {'D_scaler': 0.8, 'A1': 0.5, 'm1': 2.5, 'A2': 0.5, 'm2': 0.25}
        expected_custom = np.array([
            0.8 * (0.5 * np.exp(-2.5*0.0) + 0.5 * np.exp(-0.25*0.0)), 
            0.8 * (0.5 * np.exp(-2.5*0.1) + 0.5 * np.exp(-0.25*0.1)),
        ])
        generated_custom = aif.fast_biexponential_aif(self.time_points_minutes[:2], **custom_params)
        np.testing.assert_array_almost_equal(generated_custom, expected_custom, decimal=5)


    def test_aif_parameters_documented_and_positive(self):
        """Check AIF model parameters (non-negativity and basic doc presence)."""
        for model_name, model_func in aif.POPULATION_AIFS.items():
            with self.subTest(model=model_name):
                self.assertTrue(model_func.__doc__ is not None and len(model_func.__doc__) > 20, f"{model_name} docstring missing or too short.")
                # Get param names from metadata to test negative values
                param_meta = aif.AIF_PARAMETER_METADATA.get(model_name, [])
                if not param_meta: continue # Skip if no metadata for some reason
                
                first_param_name = param_meta[0][0] # e.g. D_scaler
                test_params_negative = {first_param_name: -1.0}
                with self.assertRaises(ValueError, msg=f"{model_name} did not raise ValueError for negative {first_param_name}"):
                     model_func(self.time_points_minutes, **test_params_negative)


    def test_population_aifs_with_custom_params(self):
        """Test that population AIFs use custom parameters correctly."""
        time_points = np.array([0.5, 1.0]) # Minutes

        # Parker
        parker_default_t05 = aif.parker_aif(time_points)[0]
        parker_custom_A1 = aif.parker_aif(time_points, A1=0.5)[0]
        self.assertNotAlmostEqual(parker_custom_A1, parker_default_t05, 
                                  msg="Parker AIF with custom A1 yielded same as default.")
        # Expected: D_scaler * (0.5 * exp(-m1*t) + A2 * exp(-m2*t))
        expected_parker_custom = 1.0 * (0.5 * np.exp(-0.171*0.5) + 0.330 * np.exp(-2.05*0.5))
        self.assertAlmostEqual(parker_custom_A1, expected_parker_custom, decimal=5)

        # Weinmann
        weinmann_default_t05 = aif.weinmann_aif(time_points)[0]
        weinmann_custom_m1 = aif.weinmann_aif(time_points, m1=0.2)[0]
        self.assertNotAlmostEqual(weinmann_custom_m1, weinmann_default_t05,
                                  msg="Weinmann AIF with custom m1 yielded same as default.")
        expected_weinmann_custom = 1.0 * (3.99 * np.exp(-0.2*0.5) + 4.78 * np.exp(-0.0111*0.5))
        self.assertAlmostEqual(weinmann_custom_m1, expected_weinmann_custom, decimal=5)

        # Fast Bi-exponential
        fast_default_t05 = aif.fast_biexponential_aif(time_points)[0]
        fast_custom_D = aif.fast_biexponential_aif(time_points, D_scaler=2.0)[0]
        self.assertNotAlmostEqual(fast_custom_D, fast_default_t05,
                                  msg="Fast Bi-exp AIF with custom D_scaler yielded same as default.")
        self.assertAlmostEqual(fast_custom_D, fast_default_t05 * 2.0, decimal=5)


    def test_generate_population_aif(self): 
        """Test generation of population AIFs, including with custom params."""
        # Parker AIF - default
        parker_concs = aif.generate_population_aif("parker", self.time_points_minutes)
        expected_parker = aif.parker_aif(self.time_points_minutes)
        np.testing.assert_array_almost_equal(parker_concs, expected_parker)
        # Weinmann AIF - with custom params
        custom_weinmann_params = {'D_scaler': 0.75, 'A1': 3.0}
        weinmann_concs_custom = aif.generate_population_aif("weinmann", self.time_points_minutes, params=custom_weinmann_params)
        expected_weinmann_custom = aif.weinmann_aif(self.time_points_minutes, **custom_weinmann_params)
        np.testing.assert_array_almost_equal(weinmann_concs_custom, expected_weinmann_custom)

        non_existent_aif = aif.generate_population_aif("non_existent_model", self.time_points_minutes)
        self.assertIsNone(non_existent_aif)

    @patch('core.conversion.signal_tc_to_concentration_tc') 
    def test_extract_aif_from_roi(self, mock_signal_tc_to_concentration_tc): # Unchanged
        mock_expected_concentration_tc = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_signal_tc_to_concentration_tc.return_value = mock_expected_concentration_tc
        dce_data_shape = (10, 10, 5, 5); dce_4d_data = np.ones(dce_data_shape) 
        dce_4d_data[2,2,1,:] = 10; dce_4d_data[3,2,1,:] = 20; dce_4d_data[2,3,1,:] = 30; dce_4d_data[3,3,1,:] = 40
        expected_mean_signal_tc = np.full(dce_data_shape[3], 25.0)
        roi_coords = (2, 2, 2, 2); slice_index_z = 1; t10_blood = 1.4; r1_blood = 4.5; TR = 0.005; baseline_pts = 2
        aif_time_tc, aif_concentration_tc = aif.extract_aif_from_roi(dce_4d_data, roi_coords, slice_index_z, t10_blood, r1_blood, TR, baseline_pts)
        mock_signal_tc_to_concentration_tc.assert_called_once(); call_args = mock_signal_tc_to_concentration_tc.call_args[0] 
        called_signal_tc = call_args[0]; np.testing.assert_array_almost_equal(called_signal_tc, expected_mean_signal_tc)
        self.assertEqual(call_args[1], t10_blood); self.assertEqual(call_args[2], r1_blood); self.assertEqual(call_args[3], TR); self.assertEqual(call_args[4], baseline_pts)
        expected_time_vector = np.arange(dce_data_shape[3]) * TR
        np.testing.assert_array_almost_equal(aif_time_tc, expected_time_vector)
        np.testing.assert_array_almost_equal(aif_concentration_tc, mock_expected_concentration_tc)

    def test_extract_aif_from_roi_value_errors(self): # Unchanged
        dce_4d_data = np.ones((5,5,3,10))
        with self.assertRaisesRegex(ValueError, "ROI start coordinates or Z-slice index out of bounds"): aif.extract_aif_from_roi(dce_4d_data, (0,0,1,1), 5, 1.4, 4.5, 0.005) 
        with self.assertRaisesRegex(ValueError, "ROI dimensions .* exceed DCE data spatial bounds"): aif.extract_aif_from_roi(dce_4d_data, (0,0,6,1), 0, 1.4, 4.5, 0.005) 
        with self.assertRaisesRegex(ValueError, "ROI width and height must be positive"): aif.extract_aif_from_roi(dce_4d_data, (0,0,0,1), 0, 1.4, 4.5, 0.005) 


class TestAifRoiSaveLoad(unittest.TestCase): # Unchanged
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_roi_props = {"slice_index": 10, "pos_x": 20.5, "pos_y": 30.0, "size_w": 15.2, "size_h": 10.8, "image_ref_name": "Mean DCE"}
    def tearDown(self): shutil.rmtree(self.test_dir)
    def test_save_load_aif_roi_definition(self):
        tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=self.test_dir, mode='w'); tmp_file_path = tmp_file.name; tmp_file.close() 
        aif.save_aif_roi_definition(self.sample_roi_props, tmp_file_path); loaded_props = aif.load_aif_roi_definition(tmp_file_path)
        self.assertIsNotNone(loaded_props); self.assertEqual(loaded_props, self.sample_roi_props); os.remove(tmp_file_path)
    def test_load_aif_roi_definition_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_roi.json")
        with self.assertRaises(FileNotFoundError): aif.load_aif_roi_definition(non_existent_path)
    def test_load_aif_roi_definition_bad_json(self):
        tmp_file_path = os.path.join(self.test_dir, "bad_roi.json")
        with open(tmp_file_path, 'w') as f: f.write("{'slice_index': 10, ...") 
        with self.assertRaisesRegex(ValueError, "Error decoding JSON"): aif.load_aif_roi_definition(tmp_file_path)
    def test_load_aif_roi_definition_missing_keys(self):
        tmp_file_path = os.path.join(self.test_dir, "missing_keys_roi.json"); bad_props = {"slice_index": 5, "pos_x": 10.0} 
        with open(tmp_file_path, 'w') as f: json.dump(bad_props, f)
        with self.assertRaisesRegex(ValueError, "Missing required key"): aif.load_aif_roi_definition(tmp_file_path)
    def test_load_aif_roi_definition_wrong_types(self):
        tmp_file_path = os.path.join(self.test_dir, "wrong_types_roi.json"); wrong_type_props = self.sample_roi_props.copy(); wrong_type_props["slice_index"] = "not_an_integer" 
        with open(tmp_file_path, 'w') as f: json.dump(wrong_type_props, f)
        with self.assertRaisesRegex(ValueError, "slice_index must be an integer"): aif.load_aif_roi_definition(tmp_file_path)
        wrong_type_props_2 = self.sample_roi_props.copy(); wrong_type_props_2["pos_x"] = "not_a_float" 
        with open(tmp_file_path, 'w') as f: json.dump(wrong_type_props_2, f)
        with self.assertRaisesRegex(ValueError, "ROI position/size values must be numeric"): aif.load_aif_roi_definition(tmp_file_path)

if __name__ == '__main__':
    unittest.main()
