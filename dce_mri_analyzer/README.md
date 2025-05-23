# DCE-MRI Analysis Tool

## Overview

This tool is being developed to enable researchers and clinicians to load and manage DCE-MRI (Dynamic Contrast-Enhanced Magnetic Resonance Imaging) time-series data, convert raw signal intensity to contrast agent concentration, perform pharmacokinetic modeling, and visualize the results.

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
    *   Selection of population-based AIF models (e.g., Parker).
    *   Interactive AIF definition by drawing an ROI on the displayed image (mean signal from ROI converted to concentration).
    *   Input fields for AIF-specific parameters (T10_blood, r1_blood, AIF baseline points).
*   **Pharmacokinetic Model Fitting:**
    *   Implementation of Standard Tofts model.
    *   Implementation of Extended Tofts model.
    *   Voxel-wise fitting of selected model to tissue concentration curves, optionally constrained by a mask.
*   **Parameter Map Generation & Export:**
    *   Generation of 3D Ktrans, ve, (and vp for Extended Tofts) parameter maps.
    *   Export of these maps as NIfTI files, using a reference NIfTI (e.g., T1 map or original DCE) for spatial alignment and header information.
*   **Visualization:**
    *   Display of loaded 3D/4D volumes (DCE, T1, Mask), generated concentration maps (mean over time), and pharmacokinetic parameter maps as 2D slices.
    *   Slice navigation using a slider.
    *   Interactive plotting of concentration-time curves for any selected voxel by double-clicking on the image viewer (plots tissue concentration, AIF, and the fitted model curve if available).
*   **User Interface:**
    *   Basic Graphical User Interface (GUI) for all functionalities.
    *   Logging of operations, loaded file details, and any errors encountered.

## Technical Stack

*   Python 3.x
*   NumPy: For numerical operations and array handling.
*   SciPy: For scientific computing, including optimization (curve fitting) and interpolation.
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

## Performance Note
Currently, voxel-wise operations (like pharmacokinetic model fitting) are performed single-threaded. For large datasets, these operations can be time-consuming. Future enhancements may include parallelization to improve performance.

## To Do / Future Enhancements

*   **Advanced AIF Management:**
    *   Saving user-defined ROIs for AIF.
    *   Integration of more population-based AIF models.
*   **More Pharmacokinetic Models:**
    *   Implementation of other models (e.g., two-compartment exchange model).
*   **Improved Visualization:**
    *   Overlaying parameter maps on anatomical images.
    *   ROI drawing tools for statistics.
*   **Batch Processing:**
    *   Ability to process multiple datasets via a script or batch interface.
*   **Parallelization:**
    *   Utilize multi-core processing for voxel-wise fitting to significantly speed up analysis.
*   **Output and Reporting:**
    *   Saving analysis reports (e.g., mean parameter values within ROIs).
    *   More comprehensive export options.

This project aims to provide a user-friendly and modular tool for DCE-MRI analysis.
