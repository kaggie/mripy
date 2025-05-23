import unittest
import numpy as np
import csv
import tempfile
import os
import shutil

# Add project root for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import reporting

class TestReportingFunctions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_calculate_roi_statistics_basic(self):
        """Test basic ROI statistics calculation."""
        data_slice = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        mask_slice = np.array([[True, True, False], [True, False, False], [False, False, False]], dtype=bool)
        # ROI values: [1, 2, 4]
        
        expected_stats = {
            "N": 3, "N_valid": 3, "Mean": np.mean([1,2,4]), "StdDev": np.std([1,2,4]),
            "Median": np.median([1,2,4]), "Min": 1.0, "Max": 4.0
        }
        
        calculated_stats = reporting.calculate_roi_statistics(data_slice, mask_slice)
        self.assertIsNotNone(calculated_stats)
        self.assertEqual(calculated_stats["N"], expected_stats["N"])
        self.assertEqual(calculated_stats["N_valid"], expected_stats["N_valid"])
        self.assertAlmostEqual(calculated_stats["Mean"], expected_stats["Mean"])
        self.assertAlmostEqual(calculated_stats["StdDev"], expected_stats["StdDev"])
        self.assertAlmostEqual(calculated_stats["Median"], expected_stats["Median"])
        self.assertAlmostEqual(calculated_stats["Min"], expected_stats["Min"])
        self.assertAlmostEqual(calculated_stats["Max"], expected_stats["Max"])

    def test_calculate_roi_statistics_empty_roi(self):
        """Test ROI statistics with an empty ROI mask."""
        data_slice = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        mask_slice = np.zeros_like(data_slice, dtype=bool) # All False
        
        expected_stats = {
            "N": 0, "Mean": np.nan, "StdDev": np.nan,
            "Median": np.nan, "Min": np.nan, "Max": np.nan
            # N_valid will be 0
        }
        
        calculated_stats = reporting.calculate_roi_statistics(data_slice, mask_slice)
        self.assertIsNotNone(calculated_stats)
        self.assertEqual(calculated_stats["N"], expected_stats["N"])
        self.assertTrue(np.isnan(calculated_stats["Mean"]))
        self.assertTrue(np.isnan(calculated_stats["StdDev"]))
        self.assertTrue(np.isnan(calculated_stats["Median"]))
        self.assertTrue(np.isnan(calculated_stats["Min"]))
        self.assertTrue(np.isnan(calculated_stats["Max"]))
        self.assertEqual(calculated_stats.get("N_valid", 0), 0)


    def test_calculate_roi_statistics_nan_data(self):
        """Test ROI statistics with NaN values within the ROI."""
        data_slice = np.array([[1, np.nan, 3], [4, 5, np.nan]], dtype=float)
        mask_slice = np.array([[True, True, False], [True, True, False]], dtype=bool)
        # ROI values: [1, np.nan, 4, 5]
        # Valid ROI values for nan-functions: [1, 4, 5]
        
        expected_stats = {
            "N": 4, "N_valid": 3, "Mean": np.nanmean([1, np.nan, 4, 5]), 
            "StdDev": np.nanstd([1, np.nan, 4, 5]),
            "Median": np.nanmedian([1, np.nan, 4, 5]), 
            "Min": np.nanmin([1, np.nan, 4, 5]), 
            "Max": np.nanmax([1, np.nan, 4, 5])
        }
        
        calculated_stats = reporting.calculate_roi_statistics(data_slice, mask_slice)
        self.assertIsNotNone(calculated_stats)
        self.assertEqual(calculated_stats["N"], expected_stats["N"])
        self.assertEqual(calculated_stats["N_valid"], expected_stats["N_valid"])
        self.assertAlmostEqual(calculated_stats["Mean"], expected_stats["Mean"])
        self.assertAlmostEqual(calculated_stats["StdDev"], expected_stats["StdDev"])
        self.assertAlmostEqual(calculated_stats["Median"], expected_stats["Median"])
        self.assertAlmostEqual(calculated_stats["Min"], expected_stats["Min"])
        self.assertAlmostEqual(calculated_stats["Max"], expected_stats["Max"])

    def test_calculate_roi_statistics_all_nan_in_roi(self):
        """Test ROI statistics when all values within the ROI are NaN."""
        data_slice = np.array([[np.nan, np.nan], [np.nan, 10]], dtype=float)
        mask_slice = np.array([[True, True], [True, False]], dtype=bool) # ROI covers only NaNs
        
        calculated_stats = reporting.calculate_roi_statistics(data_slice, mask_slice)
        self.assertIsNotNone(calculated_stats)
        self.assertEqual(calculated_stats["N"], 3) # 3 pixels in ROI
        self.assertEqual(calculated_stats["N_valid"], 0) # 0 valid pixels
        self.assertTrue(np.isnan(calculated_stats["Mean"]))
        self.assertTrue(np.isnan(calculated_stats["StdDev"]))
        # ... and so on for other stats

    def test_format_roi_statistics_to_string(self):
        """Test formatting ROI statistics to a string."""
        stats_dict = {"N": 3, "N_valid":3, "Mean": 2.33333, "StdDev": 1.247219}
        map_name = "TestMap"
        roi_name = "MyROI"
        
        formatted_string = reporting.format_roi_statistics_to_string(stats_dict, map_name, roi_name)
        self.assertIn(f"Statistics for {roi_name} on '{map_name}':", formatted_string)
        self.assertIn("Mean: 2.3333", formatted_string) # Check formatting precision
        self.assertIn("StdDev: 1.2472", formatted_string)

        # Test with N_valid = 0
        empty_stats = {"N": 5, "N_valid": 0, "Mean": np.nan}
        empty_string = reporting.format_roi_statistics_to_string(empty_stats, map_name, roi_name)
        self.assertEqual(empty_string, f"No valid data in {roi_name} for '{map_name}'.")
        
        # Test with None dict
        none_string = reporting.format_roi_statistics_to_string(None, map_name, roi_name)
        self.assertEqual(none_string, f"No valid data in {roi_name} for '{map_name}'.")


    def test_save_roi_statistics_csv(self):
        """Test saving ROI statistics to a CSV file."""
        stats_dict = {"N": 10, "N_valid": 8, "Mean": 5.5, "Median": 5.0, "StdDev": 1.5}
        map_name = "Ktrans"
        roi_name = "TumorROI_Slice5"
        
        tmp_file_path = os.path.join(self.test_dir, "stats.csv")
        
        reporting.save_roi_statistics_csv(stats_dict, tmp_file_path, map_name, roi_name)
        
        self.assertTrue(os.path.exists(tmp_file_path))
        
        # Read back and verify
        with open(tmp_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        self.assertEqual(len(rows), len(stats_dict)) # One row per statistic
        
        expected_data = []
        for stat_name, stat_value in stats_dict.items():
            expected_data.append({
                'MapName': map_name, 
                'ROIName': roi_name, 
                'Statistic': stat_name, 
                'Value': str(stat_value) # Values are read as strings from CSV
            })
        
        # Check if all expected rows are present (order might vary for dict items)
        for expected_row in expected_data:
            self.assertIn(expected_row, rows)

    def test_save_roi_statistics_csv_empty_stats(self):
        """Test ValueError when saving empty statistics."""
        tmp_file_path = os.path.join(self.test_dir, "empty_stats.csv")
        with self.assertRaisesRegex(ValueError, "No statistics data to save."):
            reporting.save_roi_statistics_csv({}, tmp_file_path, "TestMap")


if __name__ == '__main__':
    unittest.main()
