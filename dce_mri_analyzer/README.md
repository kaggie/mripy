# DCE-MRI Analysis Tool

## Overview

This tool is being developed to enable researchers and clinicians to load and manage DCE-MRI (Dynamic Contrast-Enhanced Magnetic Resonance Imaging) time-series data, convert raw signal intensity to contrast agent concentration, perform pharmacokinetic modeling, and visualize and report the results.

## Current Features

*   **Data Loading & Management:**
    *   Loading of 4D DCE NIfTI series (`.nii`, `.nii.gz`).
    *   Loading of 3D T1 maps (NIfTI).
    *   Loading of 3D Masks (NIfTI, optional).
    *   Validation of input file integrity (basic NIfTI format check) and dimensions (e.g., DCE is 4D, T1 map is 3D, spatial dimensions match).
*   **Signal-to-Concentration Conversion:**
    *   Conversion of raw signal intensity to contrast agent concentration using user-provided r1 relaxivity, TR (Repetition Time), and number of baseline time points.
*   **AIF Management:**
    *   Loading AIF from TXT/CSV files.
    *   Selection of population-based AIF models (e.g., Parker, Weinmann).
    *   Interactive AIF definition by drawing an ROI on the displayed image (mean signal from ROI converted to concentration).
    *   Input fields for AIF-specific parameters (T10_blood, r1_blood, AIF baseline points).
    *   Saving and loading of user-defined AIF ROI definitions (slice, position, size, reference image) to/from JSON files.
*   **Pharmacokinetic Model Fitting:**
    *   Implementation of Standard Tofts model (Ktrans, ve).
    *   Implementation of Extended Tofts model (Ktrans, ve, vp).
    *   Implementation of Patlak model (Ktrans_patlak, vp_patlak).
    *   Implementation of Two-Compartment Exchange Model (2CXM), fitting for Fp (Plasma Flow), PS (Permeability-Surface Area Product), vp (Plasma Volume), and ve (Extravascular Extracellular Space Volume). This model describes tracer exchange between plasma and EES compartments.
    *   Voxel-wise fitting of selected model to tissue concentration curves, optionally constrained by a mask.
    *   Parallelized voxel-wise pharmacokinetic model fitting using multiprocessing to leverage multiple CPU cores.
*   **Parameter Map Generation & Export:**
    *   Generation of 3D Ktrans, ve, vp, Ktrans_patlak, vp_patlak, Fp_2cxm, PS_2cxm, vp_2cxm, ve_2cxm parameter maps.
    *   Export of these maps as NIfTI files, using a reference NIfTI (e.g., T1 map or original DCE) for spatial alignment and header information.
*   **Visualization:**
    *   Display of loaded 3D/4D volumes (DCE, T1, Mask), generated concentration maps (mean over time), and pharmacokinetic parameter maps as 2D slices.
    *   Slice navigation using a slider.
    *   Interactive plotting of concentration-time curves for any selected voxel by double-clicking on the image viewer (plots tissue concentration, AIF, and the fitted model curve if available).
    *   Overlay of parameter maps: Display parameter maps semi-transparently on top of a selected anatomical base image (e.g., T1 map, Mean DCE), with controls for overlay map selection, alpha (transparency), and colormap.
*   **ROI Analysis & Reporting:**
    *   Tools to draw ROIs on displayed parameter maps or other images for statistical analysis.
    *   Calculation of basic statistics (mean, std, median, min, max, N, N_valid) for these ROIs.
    *   Display of ROI statistics in the GUI.
    *   Saving of ROI statistics to CSV files.
*   **User Interface:**
    *   Basic Graphical User Interface (GUI) for all functionalities.
    *   Logging of operations, loaded file details, and any errors encountered.

## Technical Stack

*   Python 3.x
*   NumPy: For numerical operations and array handling.
*   SciPy: For scientific computing, including optimization (curve fitting), integration (ODE solving, numerical integration), and interpolation.
*   NiBabel: For loading and interacting with NIfTI files.
*   PyQt5: For the graphical user interface.
*   PyQtGraph: For 2D image visualization and plotting.

## Setup and Running

1.  **Clone the repository:**
    ```bash
    # git clone <repository_url> # (Placeholder for when hosted)
    # cd dce-mri-analyzer 
    ```
    (Assuming the repository root will be named `dce-mri-analyzer`)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    (Navigate into the directory where `requirements.txt` is located, e.g., `dce-mri-analyzer`)
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    (From the directory where `main.py` and `requirements.txt` are located, e.g., `dce_mri_analyzer`)
    ```bash
    python main.py
    ```
    On Windows, if using multiprocessing, it's good practice to ensure the script is run in a way that `multiprocessing.freeze_support()` can be effective (this is included in `main.py`).

## Performance Note
Voxel-wise operations (like pharmacokinetic model fitting) can be time-consuming. The application now supports parallel processing for these operations to leverage multiple CPU cores, which can significantly reduce processing time. The number of cores can be selected in the UI. The 2CXM, due to its complexity (ODE solving per iteration), is notably slower than other models.

## To Do / Future Enhancements

*   **Advanced AIF Management:**
    *   Saving user-defined ROIs for AIF (currently saves definition, not the derived AIF curve itself).
    *   Integration of more population-based AIF models with UI for parameter adjustment.
*   **More Pharmacokinetic Models:**
    *   Implementation of other models (e.g., shutter-speed model).
*   **Improved Visualization:**
    *   ROI drawing tools for statistics (currently a single RectROI, could be more complex shapes or multiple ROIs).
    *   Direct display of NIfTI files without loading into NumPy arrays first for large datasets (memory efficiency).
*   **Batch Processing:**
    *   Ability to process multiple datasets via a script or batch interface.
*   **Output and Reporting:**
    *   More comprehensive export options (e.g., aggregated reports, saving plots).
    *   Saving and loading of analysis "sessions" or "projects".

This project aims to provide a user-friendly and modular tool for DCE-MRI analysis.
