from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QTextEdit, QGroupBox, QLineEdit, QApplication) # Added QApplication
import pyqtgraph as pg # Assuming pyqtgraph is available
import numpy as np

# Attempt relative imports for plugin structure
try:
    from ..data_io.load_dmri import load_nifti_dmri_data
    from ..fitting.dti_fitter import fit_dti_volume
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from dmri_plugin.data_io.load_dmri import load_nifti_dmri_data
    from dmri_plugin.fitting.dti_fitter import fit_dti_volume


class DMRIFittingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("dMRI Fitting Controls")

        # Initialize data and results attributes
        self.image_data = None
        self.b_values = None
        self.b_vectors = None
        self.parameter_maps = None

        main_layout = QVBoxLayout(self)

        # Data Loading Section
        data_group = QGroupBox("Load dMRI Data")
        data_layout = QVBoxLayout()
        # NIfTI file
        nifti_layout = QHBoxLayout()
        nifti_layout.addWidget(QLabel("NIfTI Image:"))
        self.nifti_path_edit = QLineEdit()
        self.nifti_path_edit.setPlaceholderText("Path to .nii or .nii.gz")
        nifti_layout.addWidget(self.nifti_path_edit)
        self.browse_nifti_button = QPushButton("Browse...")
        nifti_layout.addWidget(self.browse_nifti_button)
        data_layout.addLayout(nifti_layout)
        # bval file
        bval_layout = QHBoxLayout()
        bval_layout.addWidget(QLabel("bval File:"))
        self.bval_path_edit = QLineEdit()
        self.bval_path_edit.setPlaceholderText("Path to .bval")
        bval_layout.addWidget(self.bval_path_edit)
        self.browse_bval_button = QPushButton("Browse...")
        bval_layout.addWidget(self.browse_bval_button)
        data_layout.addLayout(bval_layout)
        # bvec file
        bvec_layout = QHBoxLayout()
        bvec_layout.addWidget(QLabel("bvec File:"))
        self.bvec_path_edit = QLineEdit()
        self.bvec_path_edit.setPlaceholderText("Path to .bvec")
        bvec_layout.addWidget(self.bvec_path_edit)
        self.browse_bvec_button = QPushButton("Browse...")
        bvec_layout.addWidget(self.browse_bvec_button)
        data_layout.addLayout(bvec_layout)
        
        self.load_dmri_button = QPushButton("Load dMRI Data") # This button will trigger the actual loading logic
        self.load_dmri_button = QPushButton("Load dMRI Data") 
        self.load_dmri_button.clicked.connect(self.handle_load_dmri_data) # Connect signal
        data_layout.addWidget(self.load_dmri_button)
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # Model & Fitting Section
        fit_group = QGroupBox("Fitting")
        fit_layout = QVBoxLayout()
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Diffusion Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["DTI"]) # Initially DTI
        model_select_layout.addWidget(self.model_combo)
        fit_layout.addLayout(model_select_layout)
        self.fit_button = QPushButton("Fit Volume")
        self.fit_button.clicked.connect(self.handle_fit_volume) # Connect signal
        fit_layout.addWidget(self.fit_button)
        fit_group.setLayout(fit_layout)
        main_layout.addWidget(fit_group)

        # Results Visualization Section
        results_group = QGroupBox("Results Viewer")
        results_layout = QHBoxLayout() 
        
        map_controls_layout = QVBoxLayout()
        map_select_layout = QHBoxLayout()
        map_select_layout.addWidget(QLabel("Display Map:"))
        self.map_combo = QComboBox() 
        self.map_combo.currentIndexChanged.connect(self.handle_map_selection_changed) # Connect signal
        map_select_layout.addWidget(self.map_combo)
        map_controls_layout.addLayout(map_select_layout)
        map_controls_layout.addStretch() 
        results_layout.addLayout(map_controls_layout)

        # Image View for maps
        self.map_image_view = pg.ImageView()
        # Hide ImageView buttons for cleaner look initially
        self.map_image_view.ui.roiBtn.hide()
        self.map_image_view.ui.menuBtn.hide()
        # self.map_image_view.ui.histogram.hide() # Optional: hide histogram
        results_layout.addWidget(self.map_image_view, stretch=1) # Give more space to image view
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # Status Area
        status_group = QGroupBox("Status & Logs")
        status_layout = QVBoxLayout()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Status messages and logs will appear here...")
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Connect browse buttons
        self.browse_nifti_button.clicked.connect(lambda: self._browse_file(self.nifti_path_edit, "NIfTI files (*.nii *.nii.gz)"))
        self.browse_bval_button.clicked.connect(lambda: self._browse_file(self.bval_path_edit, "bval files (*.bval)"))
        self.browse_bvec_button.clicked.connect(lambda: self._browse_file(self.bvec_path_edit, "bvec files (*.bvec)"))

    def _browse_file(self, path_edit_widget, file_filter):
        # Helper for file browsing
        # QFileDialog needs QApplication instance. Assuming it's run from main app.
        filepath, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if filepath:
            path_edit_widget.setText(filepath)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication # Required for QFileDialog and running the widget standalone
    import sys

    # Ensure QApplication instance exists for standalone testing
    if QApplication.instance() is None:
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    main_widget = DMRIFittingWidget()
    main_widget.setGeometry(100, 100, 800, 700) 
    main_widget.show()
    
    # For standalone test, QFileDialog might need QApplication.exec_() if not run from main app.
    # However, this test just shows the widget.
    if not hasattr(app, '_already_running_for_test'): # Basic check to avoid recursive exec_
        app._already_running_for_test = True
        # sys.exit(app.exec_()) # Only if this is the main entry point and not imported

    # New methods to be added below this line in the class

    def handle_load_dmri_data(self):
        nifti_fp = self.nifti_path_edit.text()
        bval_fp = self.bval_path_edit.text()
        bvec_fp = self.bvec_path_edit.text()

        if not all([nifti_fp, bval_fp, bvec_fp]):
            self.status_text.append("Error: NIfTI, bval, and bvec file paths must all be specified.")
            return

        self.status_text.setText(f"Loading dMRI data...\n  NIfTI: {nifti_fp}\n  bval: {bval_fp}\n  bvec: {bvec_fp}")
        QApplication.processEvents()

        try:
            image_data, b_vals, b_vecs = load_nifti_dmri_data(nifti_fp, bval_fp, bvec_fp)
            if image_data is not None:
                self.image_data = image_data
                self.b_values = b_vals
                self.b_vectors = b_vecs
                self.parameter_maps = None # Clear previous results
                self.map_combo.clear()
                self.map_image_view.clear()
                self.status_text.append("dMRI data loaded successfully.")
                self.status_text.append(f"  Image shape: {self.image_data.shape}")
                self.status_text.append(f"  b-values count: {len(self.b_values)}")
                self.status_text.append(f"  b-vectors shape: {self.b_vectors.shape}")
            else:
                self.image_data = None
                self.b_values = None
                self.b_vectors = None
                self.parameter_maps = None
                self.map_combo.clear()
                self.map_image_view.clear()
                self.status_text.append("Error loading dMRI data. Check paths and file formats. See console for details.")
        except Exception as e:
            self.image_data = None
            self.b_values = None
            self.b_vectors = None
            self.parameter_maps = None
            self.map_combo.clear()
            self.map_image_view.clear()
            self.status_text.append(f"An unexpected error occurred during data loading: {e}")


    def handle_fit_volume(self):
        if self.image_data is None or self.b_values is None or self.b_vectors is None:
            self.status_text.append("Error: dMRI data not loaded. Please load data first.")
            return

        current_model = self.model_combo.currentText()
        if current_model == "DTI":
            self.status_text.append("Starting DTI model fitting for the entire volume...")
            QApplication.processEvents()  # Keep GUI responsive

            try:
                # Assuming fit_dti_volume takes b0_threshold and min_S0_intensity_threshold as optional params
                # Or uses sensible defaults. For now, pass with defaults.
                self.parameter_maps = fit_dti_volume(self.image_data, self.b_values, self.b_vectors)
                QApplication.processEvents()

                if self.parameter_maps is not None:
                    self.status_text.append("DTI fitting complete.")
                    self.populate_map_selector()
                    if self.map_combo.count() > 0: # If maps were populated
                        self.map_combo.setCurrentIndex(0) # Trigger display of first map
                        self.handle_map_selection_changed() 
                else:
                    self.status_text.append("DTI fitting failed. Result was None.")
            except Exception as e:
                self.status_text.append(f"An error occurred during DTI fitting: {e}")
                self.parameter_maps = None # Clear results on error
                self.map_combo.clear()
                self.map_image_view.clear()
        else:
            self.status_text.append(f"Model '{current_model}' not yet implemented.")


    def populate_map_selector(self):
        self.map_combo.clear()
        if self.parameter_maps is not None:
            for map_name in self.parameter_maps.keys():
                # Check if the map data is a NumPy array and suitable for display (e.g., 3D)
                map_data = self.parameter_maps[map_name]
                if isinstance(map_data, np.ndarray) and map_data.ndim == 3:
                    self.map_combo.addItem(map_name)
                elif map_name == "D_tensor_map" and isinstance(map_data, np.ndarray) and map_data.ndim == 5:
                    # Optionally add individual tensor components like Dxx, Dyy, etc.
                    # For now, just skip the full 5D tensor map from direct combo selection.
                    pass 
            if self.map_combo.count() == 0:
                 self.status_text.append("No 3D scalar maps found in fitting results to display.")


    def handle_map_selection_changed(self):
        if self.parameter_maps is None or self.map_combo.count() == 0:
            self.map_image_view.clear()
            return

        selected_map_name = self.map_combo.currentText()
        if not selected_map_name: # Should not happen if populated, but good check
            self.map_image_view.clear()
            return

        if selected_map_name in self.parameter_maps:
            map_data_3d = self.parameter_maps[selected_map_name]
            
            if map_data_3d is not None and isinstance(map_data_3d, np.ndarray) and map_data_3d.ndim == 3:
                # Transpose for ImageView: (x,y,z) -> (z,x,y) which ImageView often prefers for slices
                # Or (z,y,x) if original was (x,y,z) and .T was (z,y,x)
                # Let's assume input (x,y,z) and we want z to be the slicing dimension (first dim for ImageView)
                try:
                    transposed_map = np.transpose(map_data_3d, (2, 0, 1)) # (z, x, y)
                    self.map_image_view.setImage(transposed_map, autoRange=True, autoLevels=True)
                    self.status_text.append(f"Displaying map: {selected_map_name}")
                except Exception as e:
                    self.map_image_view.clear()
                    self.status_text.append(f"Error displaying map '{selected_map_name}': {e}")
            else:
                self.map_image_view.clear()
                self.status_text.append(f"Cannot display map '{selected_map_name}'. Data is not a 3D array or is None.")
        else:
            self.map_image_view.clear()
            self.status_text.append(f"Selected map '{selected_map_name}' not found in results.")
```
