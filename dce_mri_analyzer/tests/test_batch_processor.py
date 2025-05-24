import unittest
import subprocess # To run the script
import os
import tempfile
import shutil
import sys

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Note: batch_processor.py itself also adds to sys.path, which is fine.

# We will test the argparsing by running the script with --help and by
# attempting to run it with minimal valid/invalid arguments.
# Full functional tests of the batch script are more like integration tests.

class TestBatchProcessorArgs(unittest.TestCase):
    def setUp(self):
        self.script_path = os.path.join(project_root, "batch_processor.py")
        self.test_dir = tempfile.mkdtemp()
        # Create dummy files that might be needed for args
        self.dce_dummy = os.path.join(self.test_dir, "dce.nii")
        self.t1_dummy = os.path.join(self.test_dir, "t1.nii")
        self.aif_dummy = os.path.join(self.test_dir, "aif.txt")
        for f_path in [self.dce_dummy, self.t1_dummy, self.aif_dummy]:
            with open(f_path, 'w') as f: f.write("dummy")


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_help_message(self):
        """Test that the script provides a help message."""
        try:
            result = subprocess.run([sys.executable, self.script_path, "--help"], 
                                    capture_output=True, text=True, check=False) # check=False to not raise on non-zero exit for --help
        except FileNotFoundError:
            self.fail(f"Python executable not found at {sys.executable} or script path {self.script_path} incorrect.")
        
        self.assertIn("usage: batch_processor.py", result.stdout.lower())
        self.assertIn("--dce", result.stdout)
        self.assertIn("--t1map", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--out_dir", result.stdout)
        # Check for AIF group
        self.assertIn("--aif_file", result.stdout)
        self.assertIn("--aif_pop_model", result.stdout)
        self.assertEqual(result.returncode, 0, "Running with --help should exit with 0.")


    def test_required_args_missing(self):
        """Test that the script exits if required arguments are missing."""
        # Missing --dce
        result = subprocess.run([sys.executable, self.script_path, 
                                 "--t1map", self.t1_dummy, 
                                 "--tr", "0.005", "--r1_relaxivity", "4.5",
                                 "--aif_file", self.aif_dummy,
                                 "--model", "Standard Tofts", "--out_dir", self.test_dir],
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0, "Script should fail without --dce.")
        self.assertIn("the following arguments are required: --dce", result.stderr.lower())
        
        # Missing AIF method (either --aif_file or --aif_pop_model)
        result = subprocess.run([sys.executable, self.script_path, 
                                 "--dce", self.dce_dummy, "--t1map", self.t1_dummy, 
                                 "--tr", "0.005", "--r1_relaxivity", "4.5",
                                 "--model", "Standard Tofts", "--out_dir", self.test_dir],
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0, "Script should fail without AIF specification.")
        # Argparse error for mutually exclusive group is a bit different
        self.assertIn("one of the arguments --aif_file --aif_pop_model is required", result.stderr.lower())


    def test_basic_arg_parsing_file_aif(self):
        """Test basic argument parsing with file-based AIF."""
        # This test doesn't run the processing, just checks if args would be parsed.
        # To do this without running, we'd need to import and call a parse_args function
        # from batch_processor.py if it were refactored that way.
        # For now, we infer correct parsing by providing all required args and checking for non-error exit (if it tried to run)
        # or by mocking the main processing logic.
        # As a simpler check, we can just ensure it doesn't complain about these specific args.
        
        # This will try to run and fail because dummy files are not valid NIfTI,
        # but it will pass arg parsing. We expect a non-zero exit code due to file errors,
        # but stderr should not contain "argument" errors for these.
        cmd = [sys.executable, self.script_path,
               "--dce", self.dce_dummy,
               "--t1map", self.t1_dummy,
               "--tr", "0.005",
               "--r1_relaxivity", "4.5",
               "--aif_file", self.aif_dummy,
               "--model", "Standard Tofts",
               "--out_dir", self.test_dir,
               "--num_processes", "1"] # Limit processes for test
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # We expect errors from file loading, not arg parsing
        self.assertNotIn("the following arguments are required", result.stderr.lower())
        self.assertNotIn("one of the arguments --aif_file --aif_pop_model is required", result.stderr.lower())
        # It will fail later, e.g. "Error loading data"
        self.assertTrue("error loading data" in result.stdout.lower() or "invalid nifti file" in result.stdout.lower() or "error loading data" in result.stderr.lower(), 
                        f"Script output did not indicate data loading error as expected. stdout: {result.stdout}, stderr: {result.stderr}")


    def test_population_aif_args(self):
        """Test argument parsing for population AIF and its parameters."""
        cmd = [sys.executable, self.script_path,
               "--dce", self.dce_dummy,
               "--t1map", self.t1_dummy,
               "--tr", "0.005",
               "--r1_relaxivity", "4.5",
               "--aif_pop_model", "parker",
               "--aif_param", "D_scaler", "0.9",
               "--aif_param", "A1", "0.75",
               "--model", "Patlak",
               "--out_dir", self.test_dir,
               "--num_processes", "1"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertNotIn("the following arguments are required", result.stderr.lower())
        self.assertNotIn("one of the arguments --aif_file --aif_pop_model is required", result.stderr.lower())
        # Check if the script's printout includes the parsed AIF params (from its own print statement)
        self.assertIn("population aif params: [['d_scaler', '0.9'], ['a1', '0.75']]", result.stdout.lower().replace(" ", "")) # Remove spaces for robust check
        # Expect data loading error
        self.assertTrue("error loading data" in result.stdout.lower() or "invalid nifti file" in result.stdout.lower() or "error loading data" in result.stderr.lower(),
                        f"Script output did not indicate data loading error as expected. stdout: {result.stdout}, stderr: {result.stderr}")


if __name__ == '__main__':
    unittest.main()
