"""
MRSIFittingWidget: A QWidget for MRSI data fitting controls and display.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QGroupBox, QFileDialog, QApplication, QComboBox) # Added QComboBox
import pyqtgraph as pg
import numpy as np
import torch # Added torch

# Attempt relative imports for plugin structure
try:
    from mrsi_plugin.data_io.loaders import load_text_mrsi_data
    from mrsi_plugin.fitting.basis import BasisSet, load_basis_set_from_directory
    from mrsi_plugin.fitting.fitter import fit_spectrum
    from mrsi_plugin.preprocessing.baseline import subtract_polynomial_baseline
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from mrsi_plugin.data_io.loaders import load_text_mrsi_data
    from mrsi_plugin.fitting.basis import BasisSet, load_basis_set_from_directory
    from mrsi_plugin.fitting.fitter import fit_spectrum
    from mrsi_plugin.preprocessing.baseline import subtract_polynomial_baseline


class MRSIFittingWidget(QWidget):
    """
    A QWidget that provides the GUI for the MRSI Fitting Plugin.
    It includes areas for spectrum display, control buttons, and results.
    """
    def __init__(self, parent=None):
        """
        Initializes the MRSIFittingWidget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("MRSI Fitting Interface")

        # Initialize data attributes
        self.mrsi_data = None
        self.mrsi_metadata = None
        self.current_voxel_index = 0
        self.current_data_filepath = None # Store the path of the loaded data file
        self.basis_set = None # For storing the loaded basis set
        self.fitted_model_current_voxel = None # Store the last fit for the current voxel
        self.all_voxels_results = None # For storing batch fit results (e.g., concentrations)
        self.all_fitted_models = None # Optionally store all models

        # --- Main Layout ---
        # Use a QHBoxLayout to have spectrum/controls on one side and map on the other
        main_h_layout = QHBoxLayout() # Main horizontal layout

        left_v_layout = QVBoxLayout() # Layout for spectrum, controls, results

        # --- Spectrum Viewer ---
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setTitle("Spectrum Viewer")
        self.spectrum_plot.setLabel('left', 'Intensity')
        self.spectrum_plot.setLabel('bottom', 'Frequency (ppm or Points)') # Placeholder
        left_v_layout.addWidget(self.spectrum_plot, stretch=2) # Give more stretch to plot

        # --- Control Panel ---
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        self.load_data_button = QPushButton("Load MRSI Data")
        self.load_data_button.clicked.connect(self.handle_load_data)
        controls_layout.addWidget(self.load_data_button)

        self.load_basis_button = QPushButton("Load Basis Set")
        self.load_basis_button.clicked.connect(self.handle_load_basis_set)
        controls_layout.addWidget(self.load_basis_button)
        
        # Voxel Navigation Buttons
        voxel_nav_layout = QHBoxLayout()
        self.prev_voxel_button = QPushButton("Previous Voxel")
        self.prev_voxel_button.clicked.connect(self.handle_prev_voxel)
        self.prev_voxel_button.setEnabled(False) # Initially disabled
        voxel_nav_layout.addWidget(self.prev_voxel_button)

        self.next_voxel_button = QPushButton("Next Voxel")
        self.next_voxel_button.clicked.connect(self.handle_next_voxel)
        self.next_voxel_button.setEnabled(False) # Initially disabled
        voxel_nav_layout.addWidget(self.next_voxel_button)
        controls_layout.addLayout(voxel_nav_layout)


        self.preprocess_button = QPushButton("Preprocess Spectrum")
        self.preprocess_button.clicked.connect(self.handle_preprocess_spectrum)
        controls_layout.addWidget(self.preprocess_button)

        self.fit_button = QPushButton("Fit Current Voxel")
        self.fit_button.clicked.connect(self.handle_fit_spectrum)
        controls_layout.addWidget(self.fit_button)

        self.fit_all_button = QPushButton("Fit All Voxels")
        self.fit_all_button.clicked.connect(self.handle_fit_all_voxels)
        controls_layout.addWidget(self.fit_all_button)
        
        controls_layout.addStretch(1) # Add stretch to push buttons to the top
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group, stretch=1)

        # --- Results Display ---
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Fitting results will be shown here...")
        results_layout.addWidget(self.results_display)
        
        results_group.setLayout(results_layout)
        left_v_layout.addWidget(results_group, stretch=1)

        main_h_layout.addLayout(left_v_layout, stretch=1) # Left panel takes 1/2 or 1/3 of space

        # --- Metabolite Map Viewer ---
        map_group = QGroupBox("Metabolite Maps")
        map_layout = QVBoxLayout()
        
        self.metabolite_selector_combo = QComboBox()
        self.metabolite_selector_combo.setEnabled(False) # Disable initially
        self.metabolite_selector_combo.currentIndexChanged.connect(self.handle_metabolite_map_selection_changed)
        map_layout.addWidget(self.metabolite_selector_combo)

        self.metabolite_map_view = pg.ImageView()
        self.metabolite_map_view.ui.histogram.hide() # Hide histogram for cleaner look initially
        self.metabolite_map_view.ui.roiBtn.hide() # Hide ROI button
        self.metabolite_map_view.ui.menuBtn.hide() # Hide Menu button
        map_layout.addWidget(self.metabolite_map_view)
        map_group.setLayout(map_layout)
        
        main_h_layout.addWidget(map_group, stretch=1) # Right panel for map view

        self.setLayout(main_h_layout) # Set the main horizontal layout for the widget

    def _get_spatial_shape_from_metadata(self):
        """
        Attempts to determine the spatial grid shape (e.g., (rows, cols) or (slices, rows, cols))
        from the loaded MRSI metadata.
        Returns a tuple or None if shape cannot be determined.
        """
        if not self.mrsi_metadata:
            return None

        # Try common keys first
        if 'GridRows' in self.mrsi_metadata and 'GridCols' in self.mrsi_metadata:
            try:
                rows = int(self.mrsi_metadata['GridRows'])
                cols = int(self.mrsi_metadata['GridCols'])
                if 'GridSlices' in self.mrsi_metadata: # For 3D data
                    slices = int(self.mrsi_metadata['GridSlices'])
                    return (slices, rows, cols) # ImageView expects (Time, Y, X) or (Z, Y, X)
                return (rows, cols) # For 2D data
            except ValueError:
                self.results_display.append("Warning: Could not parse GridRows/GridCols/GridSlices from metadata.")
        
        # Fallback to 'OriginalShape' if available (e.g., "RxCxSxP" or "RxCxP")
        original_shape_str = self.mrsi_metadata.get('OriginalShape')
        if original_shape_str:
            try:
                dims_str = original_shape_str.split('x')
                # Last dimension is always spectral points, ignore it for spatial shape
                spatial_dims_str = dims_str[:-1] 
                spatial_dims = [int(d) for d in spatial_dims_str]
                
                if len(spatial_dims) == 2: # (Rows, Cols)
                    return tuple(spatial_dims)
                elif len(spatial_dims) == 3: # (Slices, Rows, Cols) - assuming order
                    # For ImageView, if it's (S,R,C), it expects data as (S,R,C)
                    return tuple(spatial_dims) 
                elif len(spatial_dims) == 1: # E.g. a single row of voxels
                    return (1, spatial_dims[0]) # Treat as (1, Cols)
                else:
                    self.results_display.append(f"Warning: OriginalShape '{original_shape_str}' has unsupported number of spatial dimensions.")
                    return None
            except Exception as e:
                self.results_display.append(f"Warning: Could not parse 'OriginalShape' metadata ('{original_shape_str}'): {e}")
        
        self.results_display.append("Warning: Could not determine spatial shape from metadata. Map display might be incorrect or fail.")
        return None # Default if no suitable keys found


    def handle_load_data(self):
        """
        Handles the 'Load MRSI Data' button click.
        Opens a file dialog to select MRSI data and loads it.
        """
        # Define file filters for common text-based data formats
        # load_text_mrsi_data primarily expects CSV-like structures.
        file_filter = "CSV files (*.csv);;Text files (*.txt);;All files (*)"
        
        # Open file dialog
        # For simplicity, we'll assume metadata is found conventionally by load_text_mrsi_data if needed,
        # e.g., by looking for a similarly named file with a _meta.txt suffix, or it's part of the main file.
        data_filepath, _ = QFileDialog.getOpenFileName(
            self, "Open MRSI Data File", "", file_filter
        )

        if not data_filepath:
            self.results_display.append("Data loading cancelled by user.")
            return

        self.results_display.setText(f"Attempting to load MRSI data from: {data_filepath}...")
        QApplication.processEvents() # Update the UI

        try:
            # Assuming load_text_mrsi_data might also look for an associated metadata file
            # or that metadata is inferred/not strictly required by it for basic operation.
            loaded_data, loaded_metadata = load_text_mrsi_data(data_filepath=data_filepath)

            if loaded_data is None:
                self.results_display.append(f"Failed to load data from {data_filepath}. "
                                            "The loader returned no data. Check file format and console output.")
                self.mrsi_data = None
                self.mrsi_metadata = None
                self.current_data_filepath = None
                self.fitted_model_current_voxel = None
                self.all_voxels_results = None # Clear previous batch results
                self.all_fitted_models = None
            else:
                self.mrsi_data = loaded_data
                self.mrsi_metadata = loaded_metadata if loaded_metadata else {} # Ensure metadata is a dict
                self.current_voxel_index = 0
                self.current_data_filepath = data_filepath
                self.fitted_model_current_voxel = None # Reset fit on new data
                self.all_voxels_results = None # Clear previous batch results
                self.all_fitted_models = None
                
                self.results_display.append(f"Successfully loaded MRSI data ({self.mrsi_data.shape[0]} voxels, "
                                            f"{self.mrsi_data.shape[1]} points) from: {data_filepath}")
                self.update_spectrum_plot()
                self.update_metadata_display() # Call this after results_display is updated by success message
                self.update_voxel_navigation_buttons_state()

        except FileNotFoundError:
            self.results_display.append(f"Error: File not found at {data_filepath}.")
            self.mrsi_data = None
            self.mrsi_metadata = None
            self.current_data_filepath = None
            self.fitted_model_current_voxel = None
            self.all_voxels_results = None
            self.all_fitted_models = None
        except ValueError as ve:
            self.results_display.append(f"Error processing data from {data_filepath}: {ve}")
            self.mrsi_data = None
            self.mrsi_metadata = None
            self.current_data_filepath = None
            self.fitted_model_current_voxel = None
            self.all_voxels_results = None
            self.all_fitted_models = None
        except Exception as e:
            self.results_display.append(f"An unexpected error occurred while loading {data_filepath}: {e}")
            self.mrsi_data = None
            self.mrsi_metadata = None
            self.current_data_filepath = None
            self.fitted_model_current_voxel = None
            self.all_voxels_results = None
            self.all_fitted_models = None
        
        # Clear map view and disable selector on new data load or load failure
        self.metabolite_selector_combo.clear()
        self.metabolite_selector_combo.setEnabled(False)
        self.metabolite_map_view.setImage(clear=True) # Clear image view
        self.metabolite_map_view.getView().setTitle(None) # Clear title
        # self.map_group.setVisible(False) # Optionally hide the group

        self.update_spectrum_plot() # Also call here to clear plot if loading failed
        self.update_voxel_navigation_buttons_state()


    def handle_load_basis_set(self):
        """
        Handles 'Load Basis Set' button click.
        Opens a directory dialog to select a directory containing basis spectra.
        """
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Basis Set Directory", ""
        )

        if not selected_dir:
            self.results_display.append("Basis set loading cancelled by user.")
            return

        self.results_display.append(f"Attempting to load basis set from: {selected_dir}...")
        QApplication.processEvents()

        try:
            loaded_basis_set = load_basis_set_from_directory(selected_dir)
            if loaded_basis_set and loaded_basis_set.num_metabolites() > 0:
                self.basis_set = loaded_basis_set
                self.results_display.append(
                    f"Successfully loaded basis set with {self.basis_set.num_metabolites()} metabolites, "
                    f"{self.basis_set.num_points()} points each."
                )
            else:
                self.basis_set = None
                self.results_display.append(
                    f"Failed to load a valid basis set from {selected_dir}. "
                    "Ensure directory contains valid CSV files with consistent number of points."
                )
        except FileNotFoundError:
            self.results_display.append(f"Error: Directory not found at {selected_dir}.")
            self.basis_set = None
        except Exception as e:
            self.results_display.append(f"An unexpected error occurred while loading basis set: {e}")
            self.basis_set = None

    def handle_preprocess_spectrum(self):
        """ Basic placeholder for preprocessing. """
        if self.mrsi_data is None or self.mrsi_data.size == 0:
            self.results_display.append("Error: No MRSI data loaded to preprocess.")
            return

        current_spectrum_np = self.mrsi_data[self.current_voxel_index].copy() # Work on a copy

        # Example: Apply polynomial baseline correction to the real part
        # This is a simplified example. Proper preprocessing might involve more steps
        # and careful handling of complex data if the original data is complex.
        try:
            if np.iscomplexobj(current_spectrum_np):
                spectrum_for_baseline = current_spectrum_np.real
            else:
                spectrum_for_baseline = current_spectrum_np
            
            # Assuming polynomial_order=3 for this example
            corrected_real_part, _ = subtract_polynomial_baseline(spectrum_for_baseline, polynomial_order=3)
            
            if np.iscomplexobj(current_spectrum_np):
                # If original was complex, only the real part was corrected by this simple baseline.
                # This might not be ideal for all complex data.
                # For simplicity, we'll update the real part and keep imag part.
                # A more robust approach might phase data to be real first, or use complex-aware baseline methods.
                self.mrsi_data[self.current_voxel_index] = corrected_real_part + 1j * current_spectrum_np.imag
                self.results_display.append("Applied baseline correction to the real part of the spectrum.")
            else:
                self.mrsi_data[self.current_voxel_index] = corrected_real_part
                self.results_display.append("Applied baseline correction to the spectrum.")

            self.fitted_model_current_voxel = None # Invalidate previous fit after preprocessing
            self.update_spectrum_plot()
        except Exception as e:
            self.results_display.append(f"Error during preprocessing: {e}")


    def handle_fit_spectrum(self):
        """ Handles 'Fit Current Voxel' button click. """
        if self.mrsi_data is None:
            self.results_display.append("Error: No MRSI data loaded. Cannot fit.")
            return
        if self.basis_set is None:
            self.results_display.append("Error: No basis set loaded. Cannot fit.")
            return

        current_spectrum_np = self.mrsi_data[self.current_voxel_index]
        
        # Ensure data is complex for the fitter, as fit_spectrum expects complex input
        if not np.iscomplexobj(current_spectrum_np):
            # If data is real, convert it to complex (e.g. by adding zero imaginary part)
            # This is important if fit_spectrum's loss or model expects complex.
            current_spectrum_np = current_spectrum_np.astype(np.complex64) 
            
        measured_spectrum_tensor = torch.from_numpy(current_spectrum_np).cfloat()

        # Check for consistent number of points
        if measured_spectrum_tensor.shape[0] != self.basis_set.num_points():
            self.results_display.append(
                f"Error: MRSI data points ({measured_spectrum_tensor.shape[0]}) "
                f"do not match basis set points ({self.basis_set.num_points()}). Cannot fit."
            )
            return

        self.results_display.append(f"Fitting Voxel {self.current_voxel_index + 1}...")
        QApplication.processEvents()

        try:
            num_baseline_coeffs = 4 # Example: cubic baseline
            # Consider making iterations and lr configurable via UI elements later
            self.fitted_model_current_voxel, final_loss = fit_spectrum(
                measured_spectrum_tensor, 
                self.basis_set, 
                num_baseline_coeffs,
                num_iterations=1500, # Increased iterations
                learning_rate=0.005, # Potentially smaller LR for stability
                print_loss_every=300 
            )

            results_str = f"\n--- Fit Results (Voxel {self.current_voxel_index + 1}) ---\n"
            results_str += f"  Final Loss: {final_loss:.6e}\n"
            
            if self.fitted_model_current_voxel:
                concentrations = self.fitted_model_current_voxel.concentrations.detach().cpu().numpy()
                phase_rad = self.fitted_model_current_voxel.phase_rad.detach().cpu().numpy()
                results_str += f"  Fitted Phase (rad): {phase_rad:.4f} (deg: {np.rad2deg(phase_rad):.2f})\n"
                
                if self.fitted_model_current_voxel.baseline_coeffs is not None:
                    baseline_coeffs = self.fitted_model_current_voxel.baseline_coeffs.detach().cpu().numpy()
                    results_str += f"  Baseline Coefficients: {np.array2string(baseline_coeffs, precision=3)}\n"

                for i, name in enumerate(self.basis_set.get_names()):
                    results_str += f"  Conc. {name}: {concentrations[i]:.4f}\n"
            
            self.results_display.append(results_str)
            self.update_fit_plots(self.fitted_model_current_voxel, measured_spectrum_tensor)

        except Exception as e:
            self.results_display.append(f"Error during fitting: {e}")
            self.fitted_model_current_voxel = None # Clear fit if error
            self.update_spectrum_plot() # Revert to original data plot

    def update_spectrum_plot(self):
        """
        Updates the spectrum plot with the current voxel's data.
        If a fit exists for the current voxel, it calls update_fit_plots instead.
        """
        if self.fitted_model_current_voxel and self.mrsi_data is not None:
            current_spectrum_np = self.mrsi_data[self.current_voxel_index]
            if not np.iscomplexobj(current_spectrum_np):
                 current_spectrum_np = current_spectrum_np.astype(np.complex64)
            measured_spectrum_tensor = torch.from_numpy(current_spectrum_np).cfloat()
            self.update_fit_plots(self.fitted_model_current_voxel, measured_spectrum_tensor)
        else:
            self.spectrum_plot.clear()
            if self.spectrum_plot.legend: # Remove legend if it exists
                self.spectrum_plot.legend.scene().removeItem(self.spectrum_plot.legend)
                self.spectrum_plot.legend = None

            if self.mrsi_data is not None and self.mrsi_data.size > 0:
                if 0 <= self.current_voxel_index < self.mrsi_data.shape[0]:
                    spectrum_to_plot = self.mrsi_data[self.current_voxel_index]
                    
                    if np.iscomplexobj(spectrum_to_plot):
                        plot_data = spectrum_to_plot.real
                        self.spectrum_plot.setLabel('left', 'Intensity (Real Part)')
                    else:
                        plot_data = spectrum_to_plot
                        self.spectrum_plot.setLabel('left', 'Intensity')
                    
                    self.spectrum_plot.plot(plot_data, pen='b', name="Original Data") # Changed pen to blue
                    self.spectrum_plot.setTitle(f"Spectrum - Voxel {self.current_voxel_index + 1}/{self.mrsi_data.shape[0]}")
                else:
                    self.spectrum_plot.setTitle("Spectrum Viewer - Invalid Voxel Index")
                    self.results_display.append(f"Warning: Current voxel index {self.current_voxel_index} is out of bounds.")
            else:
                self.spectrum_plot.setTitle("Spectrum Viewer - No Data Loaded")
            
            if self.mrsi_data is None or self.mrsi_data.size == 0 :
                self.spectrum_plot.setLabel('left', 'Intensity')
                self.spectrum_plot.setLabel('bottom', 'Frequency (ppm or Points)')


    def update_fit_plots(self, fitted_model, original_spectrum_tensor):
        """
        Updates the plot with original data, fitted spectrum, and residual.
        """
        self.spectrum_plot.clear()
        if self.spectrum_plot.legend: # Remove old legend
             self.spectrum_plot.legend.scene().removeItem(self.spectrum_plot.legend)
        self.spectrum_plot.addLegend()

        # Plot original data (real part or magnitude)
        original_data_np = original_spectrum_tensor.detach().cpu().numpy()
        if np.iscomplexobj(original_data_np):
            self.spectrum_plot.plot(original_data_np.real, name="Original (Real)", pen='b')
        else: # Should be complex due to earlier conversion in handle_fit_spectrum
            self.spectrum_plot.plot(original_data_np, name="Original", pen='b')


        if fitted_model and self.basis_set:
            try:
                basis_spectra_np = self.basis_set.get_spectra_array()
                basis_spectra_tensor = torch.from_numpy(basis_spectra_np).float().to(original_spectrum_tensor.device)
                
                fitted_model.eval() # Ensure model is in eval mode for reconstruction
                with torch.no_grad(): # No need for gradients here
                    reconstructed_spectrum_tensor = fitted_model(basis_spectra_tensor).detach().cpu()

                # Plot fitted spectrum (real part)
                self.spectrum_plot.plot(reconstructed_spectrum_tensor.real.numpy(), name="Fitted (Real)", pen='r')

                # Plot residual (real part)
                residual_real = original_spectrum_tensor.real.cpu().numpy() - reconstructed_spectrum_tensor.real.numpy()
                self.spectrum_plot.plot(residual_real, name="Residual (Real)", pen='g')
                
                self.spectrum_plot.setTitle(f"Fit - Voxel {self.current_voxel_index + 1}/{self.mrsi_data.shape[0]}")

            except Exception as e:
                self.results_display.append(f"Error updating fit plots: {e}")
                # Fallback to just plotting original data if reconstruction fails
                if np.iscomplexobj(original_data_np):
                     self.spectrum_plot.plot(original_data_np.real, name="Original (Real) - Error in Fit Plot", pen='b')
                else:
                     self.spectrum_plot.plot(original_data_np, name="Original - Error in Fit Plot", pen='b')
        else:
             if np.iscomplexobj(original_data_np):
                self.spectrum_plot.plot(original_data_np.real, name="Original (Real) - No Fit Model", pen='b')
             else:
                self.spectrum_plot.plot(original_data_np, name="Original - No Fit Model", pen='b')


    def update_metadata_display(self):
        """
        Updates the results display with loaded metadata.
        This method appends metadata to the existing text in results_display.
        """
        if self.mrsi_metadata and isinstance(self.mrsi_metadata, dict):
            metadata_str = "\n--- Loaded Metadata ---\n"
            if not self.mrsi_metadata:
                metadata_str += "No metadata loaded or metadata is empty.\n"
            else:
                for key, value in self.mrsi_metadata.items():
                    metadata_str += f"- {key}: {value}\n"
            
            self.results_display.append(metadata_str) # Append to keep previous messages
        elif self.mrsi_data is not None: # Data loaded but no metadata dictionary
             self.results_display.append("\n--- Loaded Metadata ---\nNo separate metadata dictionary loaded.\n")

    def handle_prev_voxel(self):
        if self.mrsi_data is not None and self.current_voxel_index > 0:
            self.current_voxel_index -= 1
            self.fitted_model_current_voxel = None # Clear fit when changing voxel
            self.update_spectrum_plot()
            self.update_voxel_navigation_buttons_state()
            self.results_display.append(f"Navigated to Voxel {self.current_voxel_index + 1}.")
        else:
            self.results_display.append("Already at the first voxel or no data loaded.")
            
    def handle_next_voxel(self):
        if self.mrsi_data is not None and self.current_voxel_index < self.mrsi_data.shape[0] - 1:
            self.current_voxel_index += 1
            self.fitted_model_current_voxel = None # Clear fit when changing voxel
            self.update_spectrum_plot()
            self.update_voxel_navigation_buttons_state()
            self.results_display.append(f"Navigated to Voxel {self.current_voxel_index + 1}.")
        else:
            self.results_display.append("Already at the last voxel or no data loaded.")

    def update_voxel_navigation_buttons_state(self):
        if self.mrsi_data is not None and self.mrsi_data.shape[0] > 1:
            self.prev_voxel_button.setEnabled(self.current_voxel_index > 0)
            self.next_voxel_button.setEnabled(self.current_voxel_index < self.mrsi_data.shape[0] - 1)
        else:
            self.prev_voxel_button.setEnabled(False)
            self.next_voxel_button.setEnabled(False)

    def handle_fit_all_voxels(self):
        """
        Handles 'Fit All Voxels' button click.
        Iterates through all voxels, fits each one, and stores the results.
        """
        if self.mrsi_data is None:
            self.results_display.append("Error: No MRSI data loaded. Cannot fit all voxels.")
            return
        if self.basis_set is None:
            self.results_display.append("Error: No basis set loaded. Cannot fit all voxels.")
            return

        num_voxels = self.mrsi_data.shape[0]
        num_metabolites = self.basis_set.num_metabolites()

        # Initialize storage for results
        self.all_voxels_results = np.zeros((num_voxels, num_metabolites))
        self.all_fitted_models = [None] * num_voxels # Store individual models if needed

        self.results_display.append("\n--- Starting Batch Fitting for All Voxels ---")
        QApplication.processEvents()

        # Parameters for batch fitting (can be made configurable later)
        num_baseline_coeffs_batch = 4 
        num_iterations_batch = 500 # Reduced iterations for batch mode
        learning_rate_batch = 0.01

        for i in range(num_voxels):
            self.current_voxel_index = i # Update current voxel index
            
            status_message = f"Fitting voxel {i + 1}/{num_voxels}..."
            self.results_display.append(status_message)
            QApplication.processEvents() # Keep GUI responsive

            current_spectrum_np = self.mrsi_data[i]
            if not np.iscomplexobj(current_spectrum_np):
                current_spectrum_np = current_spectrum_np.astype(np.complex64)
            measured_spectrum_tensor = torch.from_numpy(current_spectrum_np).cfloat()

            if measured_spectrum_tensor.shape[0] != self.basis_set.num_points():
                error_msg = (f"Skipping voxel {i+1}: MRSI data points ({measured_spectrum_tensor.shape[0]}) "
                             f"do not match basis set points ({self.basis_set.num_points()}).")
                self.results_display.append(error_msg)
                continue # Skip to next voxel

            try:
                fitted_model, final_loss = fit_spectrum(
                    measured_spectrum_tensor,
                    self.basis_set,
                    num_baseline_coeffs_batch,
                    num_iterations=num_iterations_batch,
                    learning_rate=learning_rate_batch,
                    print_loss_every=0 # Suppress per-iteration loss printing for batch
                )
                
                if fitted_model:
                    self.all_voxels_results[i, :] = fitted_model.concentrations.detach().cpu().numpy()
                    self.all_fitted_models[i] = fitted_model # Store the model
                    
                    # Update plot for the currently processed voxel (optional, can be slow)
                    # self.fitted_model_current_voxel = fitted_model # Set for display
                    # self.update_fit_plots(fitted_model, measured_spectrum_tensor)
                    # QApplication.processEvents() # Update plot
                    
                    self.results_display.append(f"  Voxel {i+1} fitted. Final Loss: {final_loss:.4e}")
                else:
                    self.results_display.append(f"  Voxel {i+1} fitting failed (no model returned).")

            except Exception as e:
                self.results_display.append(f"  Error fitting voxel {i+1}: {e}")
                # Ensure results for this voxel are marked as invalid if needed, e.g., with NaNs
                self.all_voxels_results[i, :] = np.nan 
            
            # After processing each voxel, ensure the UI is still somewhat responsive
            if i % 5 == 0: # Process events every 5 voxels, for example
                 QApplication.processEvents()


        self.results_display.append("\n--- Batch Fitting Complete for All Voxels ---")
        self.results_display.append("Metabolite concentration maps can now be generated from stored results.")
        
        # Optionally, after batch fitting, set the current display to the first voxel's fit
        if num_voxels > 0:
            self.current_voxel_index = 0
            self.fitted_model_current_voxel = self.all_fitted_models[0] if self.all_fitted_models else None
            self.update_spectrum_plot() # This will call update_fit_plots if model exists
            self.update_voxel_navigation_buttons_state()
        
        # Populate metabolite selector combo box if fit was successful
        if self.all_voxels_results is not None and self.basis_set is not None:
            self.metabolite_selector_combo.clear()
            self.metabolite_selector_combo.addItems(self.basis_set.get_names())
            self.metabolite_selector_combo.setEnabled(True)
            # self.map_group.setVisible(True) # Ensure visible
            if self.metabolite_selector_combo.count() > 0:
                 self.handle_metabolite_map_selection_changed() # Display first map

    def handle_metabolite_map_selection_changed(self):
        """
        Handles changes in the metabolite selector ComboBox for map display.
        """
        if self.all_voxels_results is None or self.basis_set is None or self.metabolite_selector_combo.count() == 0:
            self.metabolite_map_view.setImage(clear=True)
            self.metabolite_map_view.getView().setTitle(None)
            return

        selected_metabolite_name = self.metabolite_selector_combo.currentText()
        if not selected_metabolite_name:
            return

        try:
            metabolite_idx = self.basis_set.get_names().index(selected_metabolite_name)
            concentrations_1d = self.all_voxels_results[:, metabolite_idx]
        except (ValueError, IndexError) as e:
            self.results_display.append(f"Error accessing concentration data for '{selected_metabolite_name}': {e}")
            self.metabolite_map_view.setImage(clear=True)
            self.metabolite_map_view.getView().setTitle(None)
            return

        map_shape_attempt = self._get_spatial_shape_from_metadata()
        
        if map_shape_attempt:
            expected_num_voxels = np.prod(map_shape_attempt)
            if concentrations_1d.size == expected_num_voxels:
                try:
                    # Reshape. For ImageView, if data is (Z, Y, X), it handles 3D.
                    # If (Y,X), it's 2D.
                    reshaped_map = concentrations_1d.reshape(map_shape_attempt)
                    
                    # ImageView typically expects (time, y, x) or (y,x)
                    # If map_shape_attempt is (rows, cols), reshape to (rows, cols)
                    # If map_shape_attempt is (slices, rows, cols), reshape to (slices, rows, cols)
                    # Transposition might be needed if data order (e.g. Fortran vs C) is different from ImageView's expectation.
                    # For now, assume direct reshape is okay. .T is often for 2D.
                    if reshaped_map.ndim == 2:
                         self.metabolite_map_view.setImage(reshaped_map.T, autoRange=True, autoLevels=True)
                    elif reshaped_map.ndim == 3: # (slices, rows, cols)
                         self.metabolite_map_view.setImage(reshaped_map, autoRange=True, autoLevels=True) # ImageView handles 3D
                    else:
                         self.results_display.append(f"Error: Reshaped map for '{selected_metabolite_name}' has unsupported dimensions: {reshaped_map.ndim}.")
                         self.metabolite_map_view.setImage(clear=True)
                         return
                    
                    self.metabolite_map_view.getView().setTitle(f"{selected_metabolite_name} Map")
                    return # Success
                except ValueError as e:
                    self.results_display.append(f"Error reshaping concentration map for '{selected_metabolite_name}': {e}. "
                                                f"Expected shape {map_shape_attempt}, got data for {concentrations_1d.size} voxels.")
            else:
                self.results_display.append(
                    f"Mismatch between number of fitted voxels ({concentrations_1d.size}) and "
                    f"expected spatial dimensions product ({expected_num_voxels}) from metadata shape {map_shape_attempt}."
                )
        else: # map_shape_attempt is None
             self.results_display.append(
                f"Could not determine spatial shape for map display of '{selected_metabolite_name}'. "
                "Consider providing 'GridRows'/'GridCols' or 'OriginalShape' in metadata."
            )

        # If any error or inability to reshape, clear the map
        self.metabolite_map_view.setImage(clear=True)
        self.metabolite_map_view.getView().setTitle(f"{selected_metabolite_name} Map - Error")


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    
    main_widget = MRSIFittingWidget()
    main_widget.setGeometry(100, 100, 700, 900) # x, y, width, height
    main_widget.show()
    
    # --- Create dummy data for direct testing ---
    dummy_data_dir = "dummy_mrsi_fitting_widget_test_data"
    dummy_mrsi_data_file = os.path.join(dummy_data_dir, "test_mrsi.csv")
    dummy_basis_dir = os.path.join(dummy_data_dir, "basis_set")

    if not os.path.exists(dummy_data_dir): os.makedirs(dummy_data_dir)
    if not os.path.exists(dummy_basis_dir): os.makedirs(dummy_basis_dir)

    # Create dummy MRSI data (2 voxels, 100 points)
    voxel1_data = np.array([10*np.exp(-((i-30)/10)**2) + 5*np.exp(-((i-70)/5)**2) for i in range(100)])
    voxel2_data = np.array([8*np.exp(-((i-40)/8)**2) + 6*np.exp(-((i-60)/10)**2) for i in range(100)])
    mrsi_test_data = np.vstack([voxel1_data, voxel2_data]) # Shape (2, 100)
    np.savetxt(dummy_mrsi_data_file, mrsi_test_data, delimiter=',')
    print(f"Created dummy MRSI data: {dummy_mrsi_data_file}")

    # Create dummy basis files
    basis_names_test = ["Met1", "Met2"]
    met1_data = np.array([np.exp(-((i-30)/10)**2) for i in range(100)])
    met2_data = np.array([np.exp(-((i-70)/5)**2) for i in range(100)])
    np.savetxt(os.path.join(dummy_basis_dir, "Met1.csv"), np.vstack((np.arange(100), met1_data)).T, delimiter=',', header="Freq,Intensity", comments='')
    np.savetxt(os.path.join(dummy_basis_dir, "Met2.csv"), np.vstack((np.arange(100), met2_data)).T, delimiter=',', header="Freq,Intensity", comments='')
    print(f"Created dummy basis set in: {dummy_basis_dir}")
    
    main_widget.results_display.setText("Test environment set up. Please use UI to load data and basis set.\n"
                                        f"MRSI Data: {dummy_mrsi_data_file}\n"
                                        f"Basis Set Dir: {dummy_basis_dir}")
    # You would then manually test by clicking buttons in the UI.
    # Automated UI clicking is beyond this scope.
    
    # Example of programmatic loading for quick check (does not test UI interaction)
    # main_widget.current_data_filepath = dummy_mrsi_data_file
    # main_widget.handle_load_data() # This would pop a dialog, so not ideal for full auto test here.
    # main_widget.basis_set = load_basis_set_from_directory(dummy_basis_dir)
    # if main_widget.basis_set:
    #     main_widget.results_display.append(f"Programmatically loaded basis: {main_widget.basis_set.num_metabolites()} mets")

    sys.exit(app.exec_())
```
