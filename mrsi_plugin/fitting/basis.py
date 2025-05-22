import numpy as np
import csv
import os

def load_basis_spectrum_csv(filepath):
    """
    Loads a single metabolite's spectrum from a CSV file.

    The CSV file is expected to have two columns: "Frequency" (or "Chemical Shift")
    and "Intensity". It may or may not have a header row. This function
    primarily returns the intensity values as a 1D NumPy array.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        numpy.ndarray: A 1D NumPy array of intensity values.
                       Returns None if loading fails or the file is empty/malformed.
        numpy.ndarray: (Optional) A 1D NumPy array of frequency/chemical shift values.
                       Returns None if not extractable or loading fails. 
                       For now, this is a placeholder and might not be fully robust.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If the CSV data cannot be converted to numeric types or is malformed.
    """
    intensities = []
    frequencies = [] # Optional

    try:
        with open(filepath, 'r', newline='') as csvfile:
            # Sniff to detect header and dialect
            try:
                has_header = csv.Sniffer().has_header(csvfile.read(1024))
                csvfile.seek(0) # Rewind after sniffing
            except csv.Error: # Could not determine if header exists (e.g. empty or too few lines)
                has_header = False # Assume no header if sniffing fails
                csvfile.seek(0) 

            reader = csv.reader(csvfile)
            
            if has_header:
                try:
                    next(reader)  # Skip header row
                except StopIteration:
                    print(f"Warning: CSV file '{filepath}' seems to be empty after header.")
                    return None, None 

            for i, row in enumerate(reader):
                if len(row) >= 2: # Need at least two columns
                    try:
                        # Attempt to convert frequency/chemical shift (column 0)
                        frequencies.append(float(row[0]))
                        # Attempt to convert intensity (column 1)
                        intensities.append(float(row[1]))
                    except ValueError:
                        print(f"Warning: Could not convert row {i+1} (0-indexed) in '{filepath}' to float. "
                              f"Row content: {row}. Skipping row.")
                        # Decide if you want to skip or fail. Skipping for now.
                        # If skipping, ensure frequencies and intensities lists remain consistent
                        # For simplicity, if one fails, we might skip both for that row.
                        if len(frequencies) > len(intensities): frequencies.pop()
                        continue 
                elif len(row) == 1: # Only one column, assume it's intensity
                     try:
                        intensities.append(float(row[0]))
                        # Frequencies will remain empty or shorter
                     except ValueError:
                        print(f"Warning: Could not convert single-column row {i+1} in '{filepath}' to float. "
                              f"Row content: {row}. Skipping row.")
                        continue
                else:
                    print(f"Warning: Row {i+1} in '{filepath}' has less than 1 value. Skipping row.")
    
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"An error occurred while reading CSV file {filepath}: {e}")
        return None, None # Return None for both if a general error occurs

    if not intensities: # If no data was successfully read
        print(f"Warning: No intensity data extracted from '{filepath}'.")
        return None, None

    intensities_array = np.array(intensities, dtype=float)
    frequencies_array = np.array(frequencies, dtype=float) if frequencies and len(frequencies) == len(intensities) else None
    
    return intensities_array, frequencies_array


class BasisSet:
    """
    Represents a basis set of metabolite spectra.
    """
    def __init__(self, names, spectra):
        """
        Initializes the BasisSet.

        Args:
            names (list): A list of strings, metabolite names.
            spectra (list): A list of 1D NumPy arrays, where each array is the
                            spectrum for the corresponding metabolite. All spectra
                            must have the same length.

        Raises:
            ValueError: If the number of names does not match the number of spectra,
                        or if the spectra do not all have the same length.
        """
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise TypeError("Names must be a list of strings.")
        if not isinstance(spectra, list) or not all(isinstance(spec, np.ndarray) for spec in spectra):
            raise TypeError("Spectra must be a list of 1D NumPy arrays.")
        if len(names) != len(spectra):
            raise ValueError("Number of names must match the number of spectra.")
        if not spectra: # Empty basis set
            # Allow empty basis set initialization
            self._names = []
            self._spectra_list = []
            self.spectra_array = np.array([]) # Empty array with shape (0,) or (0,0)
            self._num_points = 0
            print("Initialized an empty BasisSet.")
            return

        self._num_points = spectra[0].shape[0]
        for i, spec in enumerate(spectra):
            if spec.ndim != 1:
                raise ValueError(f"Spectrum for '{names[i]}' is not 1D (shape: {spec.shape}).")
            if spec.shape[0] != self._num_points:
                raise ValueError(
                    f"All spectra in the basis set must have the same number of points. "
                    f"Spectrum for '{names[0]}' has {self._num_points} points, "
                    f"but spectrum for '{names[i]}' has {spec.shape[0]} points."
                )

        self._names = list(names) # Ensure it's a mutable list copy
        self._spectra_list = list(spectra) # Store original list of spectra
        self.spectra_array = np.array(self._spectra_list) # Shape: (num_metabolites, num_points)

    def add_spectrum(self, name, spectrum):
        """
        Adds a metabolite spectrum to the basis set.

        Args:
            name (str): Name of the metabolite.
            spectrum (numpy.ndarray): 1D NumPy array of the spectrum.

        Raises:
            ValueError: If the spectrum does not have the same number of points
                        as existing spectra in the basis set, or if name already exists.
            TypeError: If inputs are of incorrect type.
        """
        if not isinstance(name, str):
            raise TypeError("Metabolite name must be a string.")
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("Spectrum must be a NumPy array.")
        if spectrum.ndim != 1:
            raise ValueError("Spectrum must be 1D.")
        if name in self._names:
            raise ValueError(f"Metabolite '{name}' already exists in the basis set.")

        if not self._spectra_list: # If basis set is currently empty
            self._num_points = spectrum.shape[0]
        elif spectrum.shape[0] != self._num_points:
            raise ValueError(f"New spectrum has {spectrum.shape[0]} points, "
                             f"but existing spectra have {self._num_points} points.")

        self._names.append(name)
        self._spectra_list.append(spectrum)
        # Recreate spectra_array; could be optimized for large basis sets
        self.spectra_array = np.array(self._spectra_list)

    def get_spectrum_by_name(self, name):
        """
        Retrieves a spectrum by its metabolite name.

        Args:
            name (str): The name of the metabolite.

        Returns:
            numpy.ndarray: The 1D spectrum array.

        Raises:
            KeyError: If the name is not found in the basis set.
        """
        try:
            index = self._names.index(name)
            return self._spectra_list[index]
        except ValueError:
            raise KeyError(f"Metabolite '{name}' not found in the basis set.")

    def get_spectra_array(self):
        """
        Returns all basis spectra as a 2D NumPy array.

        Returns:
            numpy.ndarray: 2D array of shape (num_metabolites, num_points).
                           Returns an empty array if the basis set is empty.
        """
        return self.spectra_array

    def get_names(self):
        """
        Returns the list of metabolite names.

        Returns:
            list: List of metabolite names.
        """
        return list(self.names) # Return a copy

    @property # Make names accessible as a property
    def names(self):
        return self._names
    
    @property # Make spectra_list accessible as a property
    def spectra_list(self):
        return self._spectra_list

    def num_metabolites(self):
        """
        Returns the number of metabolites in the basis set.

        Returns:
            int: Number of metabolites.
        """
        return len(self.names)

    def num_points(self):
        """
        Returns the number of points in the spectra.

        Returns:
            int: Number of spectral points. Returns 0 if basis set is empty.
        """
        return self._num_points

    def __repr__(self):
        return (f"<BasisSet with {self.num_metabolites()} metabolites, "
                f"{self.num_points()} points each>")


def load_basis_set_from_directory(directory_path, file_extension=".csv"):
    """
    Loads a basis set from a directory containing individual spectrum files.

    Each file with the specified extension in the directory is assumed to be
    a basis spectrum. The filename (without extension) is used as the
    metabolite name.

    Args:
        directory_path (str): Path to the directory.
        file_extension (str, optional): Extension of the basis files.
                                        Defaults to ".csv".

    Returns:
        BasisSet: A BasisSet object populated with spectra from the directory.
                  Returns an empty BasisSet if no valid files are found or
                  if the directory does not exist.

    Raises:
        FileNotFoundError: If the directory_path does not exist.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    names = []
    spectra = []
    
    print(f"Scanning directory: {directory_path} for '*{file_extension}' files.")

    for filename in sorted(os.listdir(directory_path)): # sorted for consistent order
        if filename.endswith(file_extension):
            filepath = os.path.join(directory_path, filename)
            metabolite_name = os.path.splitext(filename)[0]
            
            print(f"Loading basis spectrum: {metabolite_name} from {filename}...")
            try:
                intensity_data, _ = load_basis_spectrum_csv(filepath) # Ignore frequency data for BasisSet
                if intensity_data is not None:
                    names.append(metabolite_name)
                    spectra.append(intensity_data)
                    print(f"Successfully loaded {metabolite_name} ({len(intensity_data)} points).")
                else:
                    print(f"Warning: Could not load data for {metabolite_name} from {filename}. Skipping.")
            except FileNotFoundError: # Should not happen if os.listdir provides valid files
                print(f"Error: File {filename} listed but not found at {filepath}. Skipping.")
            except Exception as e:
                print(f"Error loading spectrum {metabolite_name} from {filename}: {e}. Skipping.")
    
    if not names:
        print(f"No valid basis spectra found in directory {directory_path} with extension {file_extension}.")
        return BasisSet([], []) # Return an empty basis set

    try:
        return BasisSet(names, spectra)
    except ValueError as e:
        print(f"Error creating BasisSet from directory {directory_path}: {e}")
        print("This might be due to spectra having different lengths. Returning an empty BasisSet.")
        return BasisSet([], [])


if __name__ == '__main__':
    # --- Create Dummy CSV files for testing ---
    dummy_dir = "dummy_basis_set"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)

    # Metabolite 1: NAA.csv
    naa_freq = np.linspace(0, 5, 100)
    naa_intensity = np.exp(-((naa_freq - 2.0)**2) / (2 * 0.1**2)) # Gaussian peak at 2.0 ppm
    with open(os.path.join(dummy_dir, "NAA.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency", "Intensity"]) # Header
        for freq, intensity in zip(naa_freq, naa_intensity):
            writer.writerow([f"{freq:.4f}", f"{intensity:.4f}"])
    print("Created NAA.csv")

    # Metabolite 2: Cr.csv (same number of points)
    cr_freq = np.linspace(0, 5, 100)
    cr_intensity = np.exp(-((cr_freq - 3.0)**2) / (2 * 0.05**2)) # Gaussian peak at 3.0 ppm
    with open(os.path.join(dummy_dir, "Cr.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        # No header for this one to test sniffing
        for freq, intensity in zip(cr_freq, cr_intensity):
            writer.writerow([f"{freq:.4f}", f"{intensity:.4f}"])
    print("Created Cr.csv (no header)")
    
    # Metabolite 3: Cho.csv (different number of points for testing error handling)
    cho_freq_bad = np.linspace(0, 5, 90) # 90 points
    cho_intensity_bad = np.exp(-((cho_freq_bad - 3.9)**2) / (2 * 0.08**2))
    with open(os.path.join(dummy_dir, "Cho_bad.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency", "Intensity"])
        for freq, intensity in zip(cho_freq_bad, cho_intensity_bad):
            writer.writerow([f"{freq:.4f}", f"{intensity:.4f}"])
    print("Created Cho_bad.csv (90 points)")

    # Test 1: load_basis_spectrum_csv
    print("\\n--- Test 1: load_basis_spectrum_csv ---")
    naa_spec, naa_freq_axis = load_basis_spectrum_csv(os.path.join(dummy_dir, "NAA.csv"))
    if naa_spec is not None:
        print(f"NAA spectrum loaded successfully: {naa_spec.shape} points.")
        if naa_freq_axis is not None:
            print(f"NAA frequency axis loaded: {naa_freq_axis.shape} points.")
    else:
        print("Failed to load NAA.csv")

    cr_spec, cr_freq_axis = load_basis_spectrum_csv(os.path.join(dummy_dir, "Cr.csv"))
    if cr_spec is not None:
        print(f"Cr spectrum loaded successfully (no header): {cr_spec.shape} points.")
    else:
        print("Failed to load Cr.csv")
    print("------------------------------------\\n")

    # Test 2: BasisSet class initialization
    print("--- Test 2: BasisSet class ---")
    if naa_spec is not None and cr_spec is not None:
        try:
            basis_set = BasisSet(names=["NAA", "Cr"], spectra=[naa_spec, cr_spec])
            print(f"BasisSet created: {basis_set}")
            print(f"Metabolite names: {basis_set.get_names()}")
            print(f"Number of metabolites: {basis_set.num_metabolites()}")
            print(f"Number of points per spectrum: {basis_set.num_points()}")
            
            naa_from_basis = basis_set.get_spectrum_by_name("NAA")
            assert np.array_equal(naa_from_basis, naa_spec), "Mismatch in NAA spectrum from basis set"
            print("NAA spectrum retrieved successfully.")

            # Test adding a new spectrum
            cho_intensity_good = np.exp(-((np.linspace(0, 5, 100) - 3.9)**2) / (2 * 0.08**2))
            basis_set.add_spectrum("Cho", cho_intensity_good)
            print(f"After adding Cho: {basis_set}")
            assert "Cho" in basis_set.get_names()

        except ValueError as e:
            print(f"Error creating BasisSet: {e}")
    print("------------------------------------\\n")

    # Test 3: load_basis_set_from_directory
    print("--- Test 3: load_basis_set_from_directory ---")
    # This will attempt to load Cho_bad.csv and should print a warning or error
    # and potentially result in a BasisSet that excludes Cho_bad or is empty if all fail.
    full_basis_set = load_basis_set_from_directory(dummy_dir)
    print(f"BasisSet from directory: {full_basis_set}")
    print(f"Metabolites in directory-loaded basis set: {full_basis_set.get_names()}")
    # Expected: NAA, Cr. Cho_bad should be skipped due to different num_points.
    assert "NAA" in full_basis_set.get_names(), "NAA not in directory-loaded basis set"
    assert "Cr" in full_basis_set.get_names(), "Cr not in directory-loaded basis set"
    assert "Cho_bad" not in full_basis_set.get_names(), "Cho_bad (mismatched points) should not be in basis set"
    
    if full_basis_set.num_metabolites() > 0:
         print(f"Points in dir-loaded basis: {full_basis_set.num_points()}")
    print("------------------------------------\\n")

    # Test 4: Error handling for BasisSet
    print("--- Test 4: BasisSet error handling ---")
    try:
        # Mismatched lengths
        BasisSet(["Met1", "Met2"], [np.array([1,2,3]), np.array([4,5,6,7])])
    except ValueError as e:
        print(f"Caught expected error for mismatched lengths: {e}")
    
    try:
        # Mismatched names/spectra count
        BasisSet(["Met1"], [np.array([1,2,3]), np.array([4,5,6])])
    except ValueError as e:
        print(f"Caught expected error for mismatched counts: {e}")
    print("------------------------------------\\n")

    # Clean up dummy files and directory (optional)
    # import shutil
    # if os.path.exists(dummy_dir):
    #     shutil.rmtree(dummy_dir)
    #     print(f"Cleaned up {dummy_dir}")
    print("Test execution finished. Inspect dummy_basis_set directory and output for details.")
```
