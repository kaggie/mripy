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

    def _create_fake_nifti(self, shape, affine=np.eye(4), filename="test.nii.gz", data_type=np.float32):
        """
        Generates a nib.Nifti1Image with dummy data, saves it to a temporary file,
        and returns the full path to the file.
        """
        data = np.random.rand(*shape).astype(data_type)
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
        dce_shape = (10, 10, 10, 5) 
        fake_t1_mismatch_path = self._create_fake_nifti(shape=(5, 5, 5), filename="t1_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "T1 map dimensions do not match DCE series spatial dimensions."):
            io.load_t1_map(fake_t1_mismatch_path, dce_shape=dce_shape)

    def test_load_mask_success(self):
        """Test successful loading of a 3D mask file."""
        mask_data_orig = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.uint8)
        fake_mask_path = self._create_fake_nifti(shape=(10,10,10), data_type=np.uint8, filename="mask.nii.gz")
        # Overwrite with specific data if needed, or ensure _create_fake_nifti can pass it
        mask_img_for_ref = nib.Nifti1Image(mask_data_orig, np.eye(4))
        nib.save(mask_img_for_ref, fake_mask_path)


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
        reference_shape = (10, 10, 10) 
        fake_mask_mismatch_path = self._create_fake_nifti(shape=(5, 5, 5), filename="mask_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "Mask dimensions do not match the reference image dimensions."):
            io.load_mask(fake_mask_mismatch_path, reference_shape=reference_shape)

class TestSaveNiftiMap(unittest.TestCase): # New test class
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ref_affine = np.array([
            [-2.,  0.,  0.,  128.],
            [ 0.,  2.,  0., -128.],
            [ 0.,  0.,  2., -128.],
            [ 0.,  0.,  0.,    1.]
        ])
        self.ref_shape_3d = (5,5,5)
        self.ref_shape_4d = (5,5,5,10)

        # Create a 3D reference NIfTI
        self.ref_3d_filepath = os.path.join(self.test_dir, "ref_3d.nii.gz")
        ref_3d_data = np.zeros(self.ref_shape_3d, dtype=np.int16)
        ref_3d_img = nib.Nifti1Image(ref_3d_data, self.ref_affine)
        nib.save(ref_3d_img, self.ref_3d_filepath)
        
        # Create a 4D reference NIfTI
        self.ref_4d_filepath = os.path.join(self.test_dir, "ref_4d.nii.gz")
        ref_4d_data = np.zeros(self.ref_shape_4d, dtype=np.float32)
        ref_4d_img = nib.Nifti1Image(ref_4d_data, self.ref_affine)
        nib.save(ref_4d_img, self.ref_4d_filepath)


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_nifti_map_success_3d_ref(self):
        """Test saving a 3D map using a 3D reference NIfTI."""
        data_map = np.random.rand(*self.ref_shape_3d).astype(np.float32)
        output_filepath = os.path.join(self.test_dir, "output_map_3d_ref.nii.gz")
        
        io.save_nifti_map(data_map, self.ref_3d_filepath, output_filepath)
        
        self.assertTrue(os.path.exists(output_filepath))
        loaded_img = nib.load(output_filepath)
        
        self.assertEqual(loaded_img.shape, self.ref_shape_3d)
        np.testing.assert_array_almost_equal(loaded_img.get_fdata(), data_map, decimal=5)
        np.testing.assert_array_almost_equal(loaded_img.affine, self.ref_affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), np.float32)

    def test_save_nifti_map_success_4d_ref(self):
        """Test saving a 3D map using a 4D reference NIfTI."""
        data_map = np.random.rand(*self.ref_shape_3d).astype(np.float32) # Map is 3D
        output_filepath = os.path.join(self.test_dir, "output_map_4d_ref.nii.gz")
        
        io.save_nifti_map(data_map, self.ref_4d_filepath, output_filepath)
        
        self.assertTrue(os.path.exists(output_filepath))
        loaded_img = nib.load(output_filepath)
        
        self.assertEqual(loaded_img.shape, self.ref_shape_3d) # Output should be 3D
        np.testing.assert_array_almost_equal(loaded_img.get_fdata(), data_map, decimal=5)
        np.testing.assert_array_almost_equal(loaded_img.affine, self.ref_affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), np.float32)
        self.assertEqual(loaded_img.header['dim'][0], 3) # Check if header indicates 3D

    def test_save_nifti_map_ref_not_found(self):
        """Test FileNotFoundError if reference NIfTI does not exist."""
        data_map = np.random.rand(*self.ref_shape_3d)
        non_existent_ref = os.path.join(self.test_dir, "non_existent_ref.nii.gz")
        output_filepath = os.path.join(self.test_dir, "output_map.nii.gz")
        with self.assertRaises(FileNotFoundError):
            io.save_nifti_map(data_map, non_existent_ref, output_filepath)

    def test_save_nifti_map_data_not_3d(self):
        """Test ValueError if data_map is not 3D."""
        data_map_2d = np.random.rand(5,5)
        output_filepath = os.path.join(self.test_dir, "output_map.nii.gz")
        with self.assertRaisesRegex(ValueError, "data_map must be a 3D array"):
            io.save_nifti_map(data_map_2d, self.ref_3d_filepath, output_filepath)

    def test_save_nifti_map_shape_mismatch(self):
        """Test ValueError if data_map shape mismatches reference NIfTI spatial shape."""
        data_map_wrong_shape = np.random.rand(3,3,3) # Different shape
        output_filepath = os.path.join(self.test_dir, "output_map.nii.gz")
        with self.assertRaisesRegex(ValueError, "data_map shape .* does not match reference NIfTI spatial shape"):
            io.save_nifti_map(data_map_wrong_shape, self.ref_3d_filepath, output_filepath)


if __name__ == '__main__':
    unittest.main()
