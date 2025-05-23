import unittest
import numpy as np
import nibabel as nib
import os
import tempfile
import shutil
import sys

# Add the project root to the Python path to allow direct import of dce_mri_analyzer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import io

class TestIoFunctions(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for fake NIfTI files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory and its contents."""
        shutil.rmtree(self.test_dir)

    def _create_fake_nifti(self, shape, affine=np.eye(4), filename="test.nii.gz"):
        """
        Generates a nib.Nifti1Image with dummy data, saves it to a temporary file,
        and returns the full path to the file.
        """
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        filepath = os.path.join(self.test_dir, filename)
        nib.save(img, filepath)
        return filepath

    def test_load_nifti_file_success(self):
        """Test successful loading of a NIfTI file."""
        fake_nifti_path = self._create_fake_nifti(shape=(10, 10, 10))
        img = io.load_nifti_file(fake_nifti_path)
        self.assertIsInstance(img, nib.Nifti1Image)
        self.assertEqual(img.shape, (10, 10, 10))

    def test_load_nifti_file_not_found(self):
        """Test loading a non-existent NIfTI file."""
        non_existent_file = os.path.join(self.test_dir, "non_existent_file.nii")
        with self.assertRaises(FileNotFoundError):
            io.load_nifti_file(non_existent_file)

    def test_load_nifti_file_invalid(self):
        """Test loading an invalid NIfTI file (e.g., an empty text file)."""
        invalid_file_path = os.path.join(self.test_dir, "invalid.nii")
        with open(invalid_file_path, "w") as f:
            f.write("This is not a NIfTI file.")
        
        with self.assertRaises(ValueError): 
            io.load_nifti_file(invalid_file_path)

    def test_load_dce_series_success(self):
        """Test successful loading of a 4D DCE series."""
        fake_dce_path = self._create_fake_nifti(shape=(10, 10, 10, 20), filename="dce.nii.gz")
        dce_data = io.load_dce_series(fake_dce_path)
        self.assertIsInstance(dce_data, np.ndarray)
        self.assertEqual(dce_data.shape, (10, 10, 10, 20))
        # Basic data check (type)
        self.assertTrue(np.issubdtype(dce_data.dtype, np.floating) or np.issubdtype(dce_data.dtype, np.integer))


    def test_load_dce_series_wrong_dim(self):
        """Test loading a DCE series with incorrect (3D) dimensions."""
        fake_3d_path = self._create_fake_nifti(shape=(10, 10, 10), filename="3d.nii.gz")
        with self.assertRaisesRegex(ValueError, "DCE series must be a 4D NIfTI image."):
            io.load_dce_series(fake_3d_path)

    def test_load_t1_map_success(self):
        """Test successful loading of a 3D T1 map."""
        fake_t1_path = self._create_fake_nifti(shape=(10, 10, 10), filename="t1.nii.gz")
        t1_data = io.load_t1_map(fake_t1_path)
        self.assertIsInstance(t1_data, np.ndarray)
        self.assertEqual(t1_data.shape, (10, 10, 10))

    def test_load_t1_map_wrong_dim(self):
        """Test loading a T1 map with incorrect (4D) dimensions."""
        fake_4d_path = self._create_fake_nifti(shape=(10, 10, 10, 5), filename="4d_t1.nii.gz")
        with self.assertRaisesRegex(ValueError, "T1 map must be a 3D NIfTI image."):
            io.load_t1_map(fake_4d_path)

    def test_load_t1_map_dim_mismatch_with_dce(self):
        """Test T1 map loading with spatial dimensions mismatch against DCE shape."""
        dce_shape = (10, 10, 10, 5) # x, y, z, time
        fake_t1_mismatch_path = self._create_fake_nifti(shape=(5, 5, 5), filename="t1_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "T1 map dimensions do not match DCE series spatial dimensions."):
            io.load_t1_map(fake_t1_mismatch_path, dce_shape=dce_shape)

    def test_load_mask_success(self):
        """Test successful loading of a 3D mask file."""
        mask_data_orig = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data_orig, np.eye(4))
        fake_mask_path = os.path.join(self.test_dir, "mask.nii.gz")
        nib.save(mask_img, fake_mask_path)

        mask_data_loaded = io.load_mask(fake_mask_path)
        self.assertIsInstance(mask_data_loaded, np.ndarray)
        self.assertEqual(mask_data_loaded.dtype, bool)
        self.assertEqual(mask_data_loaded.shape, (10, 10, 10))
        np.testing.assert_array_equal(mask_data_loaded, mask_data_orig.astype(bool))

    def test_load_mask_wrong_dim(self):
        """Test loading a mask with incorrect (4D) dimensions."""
        fake_4d_mask_path = self._create_fake_nifti(shape=(10, 10, 10, 3), filename="4d_mask.nii.gz")
        with self.assertRaisesRegex(ValueError, "Mask must be a 3D NIfTI image."):
            io.load_mask(fake_4d_mask_path)

    def test_load_mask_dim_mismatch_with_ref(self):
        """Test mask loading with spatial dimensions mismatch against reference shape."""
        reference_shape = (10, 10, 10) # x, y, z
        fake_mask_mismatch_path = self._create_fake_nifti(shape=(5, 5, 5), filename="mask_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "Mask dimensions do not match the reference image dimensions."):
            io.load_mask(fake_mask_mismatch_path, reference_shape=reference_shape)

if __name__ == '__main__':
    unittest.main()
