# Medical Image Viewer and Processor

This application is a desktop tool for viewing and processing various medical image formats, including DICOM, NIfTI, and standard image types like PNG and JPEG. It provides basic image manipulation features and aims to support more advanced processing modules in the future.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python main.py
    ```

## MRSI Fitting Plugin

### Overview

The MRSI Fitting Plugin extends the Medical Image Viewer with specialized tools for Magnetic Resonance Spectroscopy Imaging (MRSI) data. It allows users to load MRSI spectral data, fit metabolite concentrations using a predefined basis set, and visualize individual spectra (raw, fitted, and residual) as well as spatial maps of fitted metabolite concentrations.

### Accessing the Plugin

The MRSI Fitting Plugin can be accessed from the main application menu:
**Tools -> MRSI Fitting**

This will open a dedicated dockable window with the MRSI fitting interface.

### Basic Usage Workflow

1.  **Load MRSI Data:**
    *   Click the "Load MRSI Data" button.
    *   Select the MRSI data file. This is typically a text-based file (e.g., `.csv`, `.txt`).
    *   **Format:** The data is expected to be structured such that spectral points are in rows and each column represents a different voxel. The `load_text_mrsi_data` function will then transpose this to a (voxels, points) orientation internally.
    *   **Metadata:** For correct spatial map display, metadata defining the grid dimensions is crucial. The plugin attempts to read this from an associated metadata file (e.g., `datafilename_meta.txt`) or directly if included. Important keys include:
        *   `OriginalShape`: e.g., "8x8x1024" (Rows x Columns x SpectralPoints) or "8x8x1x1024" (Rows x Columns x Slices x SpectralPoints). The last dimension is always spectral points.
        *   Alternatively: `GridRows`, `GridCols`, and optionally `GridSlices`.

2.  **Load Basis Set:**
    *   Click the "Load Basis Set" button.
    *   Select a directory containing the basis set files.
    *   **Format:** Each file in the directory should be a CSV (`.csv`) representing a single metabolite's spectrum. The CSV should ideally have two columns: the first for frequency/chemical shift (often ignored by the basic model but good practice) and the second for the intensity values. The plugin uses the filename (without extension) as the metabolite name. All basis spectra must have the same number of spectral points as the MRSI data.

3.  **Navigate Voxels (Optional):**
    *   Use the "Previous Voxel" and "Next Voxel" buttons to select the desired voxel for inspection or individual fitting. The spectrum display will update accordingly.

4.  **Preprocess Spectrum (Optional):**
    *   Click the "Preprocess Spectrum" button.
    *   Currently, this applies a basic polynomial baseline correction to the real part of the selected voxel's spectrum. Further preprocessing steps may be added in the future.

5.  **Fit Current Voxel:**
    *   Select a voxel using the navigation buttons.
    *   Click the "Fit Current Voxel" button.
    *   The plugin will fit the selected voxel's spectrum using the loaded basis set and the internal fitting model.
    *   Fitted parameters (concentrations, phase, baseline coefficients) will be displayed in the "Results" text area.
    *   The spectrum plot will update to show the original data (real part), the fitted spectrum (real part), and the residual.

6.  **Fit All Voxels:**
    *   Click the "Fit All Voxels" button.
    *   This will iterate through all voxels in the MRSI dataset, applying the fitting procedure to each.
    *   Progress will be updated in the "Results" text area.
    *   This process can take some time depending on the dataset size and fitting iterations.
    *   Once completed, the results for all voxels are stored internally, enabling the display of metabolite maps.

7.  **View Metabolite Maps:**
    *   After "Fit All Voxels" is complete, the "Metabolite Maps" section will become active.
    *   Select a metabolite from the dropdown menu.
    *   The corresponding concentration map will be displayed as a 2D or 3D image, depending on the MRSI data's spatial dimensions (requires correct metadata for reshaping).

### Data Format Notes

For the plugin to function correctly, your data should adhere to the following formats:

*   **MRSI Data File:**
    *   A plain text file (commonly CSV).
    *   Data should be arranged as **spectral points in rows, with each column representing a different voxel**. The loader transposes this to the internal (num_voxels, num_spectral_points) format.
    *   Example: If you have 64 voxels and 1024 points per spectrum, your CSV might have 1024 rows and 64 columns.
*   **MRSI Metadata:**
    *   Crucial for displaying metabolite maps correctly. The plugin looks for metadata to define the spatial grid of the MRSI data.
    *   This can be provided in a separate text file (e.g., `my_data_meta.txt` if your data is `my_data.csv`) or through specific metadata keys if your loader supports embedded metadata.
    *   **Key metadata fields for spatial dimensions:**
        *   `OriginalShape`: A string like "RowsxColumnsxSpectralPoints" (e.g., "8x8x1024") or "RowsxColumnsxSlicesxSpectralPoints" (e.g., "8x8x1x1024"). The plugin uses the non-spectral point dimensions to reconstruct the map.
        *   Alternatively: `GridRows: <number>`, `GridCols: <number>`, and (if 3D MRSI) `GridSlices: <number>`.
    *   Other metadata like `SpectralWidth (Hz)`, `EchoTime (ms)` are good to have for record-keeping and will be displayed if loaded.
*   **Basis Set Files:**
    *   A directory containing multiple `.csv` files.
    *   Each CSV file represents one metabolite. The filename (e.g., `NAA.csv`) is used as the metabolite name ("NAA").
    *   Each CSV file should contain at least two columns:
        1.  Frequency or Chemical Shift (often not directly used in the current simple model but good for consistency).
        2.  Intensity of the spectrum at that point.
    *   All basis spectra must have the same number of points, and this number must match the number of points in the loaded MRSI data spectra.

### (Optional) Sample Data

While specific sample datasets are not bundled, you can create dummy data adhering to the formats described above to test the plugin. For instance:

*   **MRSI Data (e.g., `sample_mrsi.csv` for an 2x2 grid, 512 points):**
    ```csv
    point1_vox1,point1_vox2,point1_vox3,point1_vox4
    point2_vox1,point2_vox2,point2_vox3,point2_vox4
    ... (512 rows)
    point512_vox1,point512_vox2,point512_vox3,point512_vox4
    ```
*   **Associated Metadata (e.g., `sample_mrsi_meta.txt`):**
    ```
    OriginalShape: 2x2x512
    SpectralWidth (Hz): 2000
    ```
*   **Basis Set (in a directory, e.g., `my_basis_set/`):**
    *   `MetaboliteA.csv`:
        ```csv
        Freq,Intensity
        0.1,100
        0.2,120
        ... (512 points)
        ```
    *   `MetaboliteB.csv`:
        ```csv
        Freq,Intensity
        0.1,80
        0.2,95
        ... (512 points)
        ```
