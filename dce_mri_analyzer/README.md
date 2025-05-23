# DCE-MRI Analysis Tool

## Overview

This tool is being developed to enable researchers and clinicians to load and manage DCE-MRI (Dynamic Contrast-Enhanced Magnetic Resonance Imaging) time-series data, convert raw signal intensity to contrast agent concentration, and eventually perform pharmacokinetic modeling. This is the initial phase focusing on data loading, validation, and signal-to-concentration conversion.

## Current Features (Phase 1)

*   Loading of 4D DCE NIfTI series (`.nii`, `.nii.gz`).
*   Loading of 3D T1 maps (NIfTI).
*   Loading of 3D Masks (NIfTI, optional).
*   Validation of input file integrity (basic NIfTI format check) and dimensions (e.g., DCE is 4D, T1 map is 3D, spatial dimensions match).
*   Signal-to-Concentration conversion using user-provided r1 relaxivity and TR (Repetition Time).
*   Basic Graphical User Interface (GUI) for:
    *   Selecting input NIfTI files (DCE series, T1 map, Mask).
    *   Inputting conversion parameters (r1 relaxivity, TR).
    *   Triggering the signal-to-concentration analysis.
*   Logging of operations, loaded file details, and any errors encountered during processing.

## Technical Stack

*   Python 3.x
*   NumPy: For numerical operations and array handling.
*   SciPy: (Currently planned for use in future modeling, included in requirements for consistency).
*   NiBabel: For loading and interacting with NIfTI files.
*   PyQt5: For the graphical user interface.

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
    (From the directory where `main.py` and `requirements.txt` are located, e.g., `dce-mri-analyzer`)
    ```bash
    python main.py
    ```

## To Do / Future Enhancements

*   **AIF Management:**
    *   Manual AIF selection from ROI on the image.
    *   Loading AIF data from external files (e.g., CSV, TXT).
    *   Integration of population-based AIF models.
*   **Pharmacokinetic Model Fitting:**
    *   Implementation of the Standard Tofts model.
    *   Implementation of the Extended Tofts model.
*   **Parameter Map Generation:**
    *   Generation and export of pharmacokinetic parameter maps (e.g., Ktrans, ve, vp) as NIfTI files.
*   **Advanced Visualization:**
    *   Display of parameter maps within the GUI.
    *   Plotting of concentration curves and model fits.
*   **Empirical Analyses:**
    *   Calculation of metrics like AUC (Area Under Curve), TTP (Time To Peak), Peak Enhancement.

This project aims to provide a user-friendly and modular tool for DCE-MRI analysis.
