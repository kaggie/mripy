import sys
import traceback
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QFormLayout,
)
from PyQt5.QtCore import Qt

# Assuming main.py is in dce_mri_analyzer and runs this module,
# relative imports should work.
from ..core import io
from ..core import conversion
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DCE-MRI Analysis Tool")
        self.setGeometry(100, 100, 900, 700)  # x, y, width, height

        # Initialize data storage attributes
        self.dce_data = None
        self.t10_data = None
        self.mask_data = None
        self.Ct_data = None # To store concentration data
        self.dce_shape_for_validation = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left panel for controls
        self.left_panel_layout = QVBoxLayout()
        self._create_file_io_section() # Creates group box and elements
        self._create_conversion_settings_section()
        self._create_aif_section()
        self._create_model_fitting_section()
        self._create_processing_section()

        self.left_panel_layout.addWidget(self.file_io_group)
        self.left_panel_layout.addWidget(self.conversion_settings_group)
        self.left_panel_layout.addWidget(self.aif_group)
        self.left_panel_layout.addWidget(self.model_fitting_group)
        self.left_panel_layout.addWidget(self.processing_group)
        self.left_panel_layout.addStretch(1)

        # Right panel for display and log
        self.right_panel_layout = QVBoxLayout()
        self._create_display_area()
        self._create_log_console()

        self.right_panel_layout.addWidget(self.display_area_group, stretch=2)
        self.right_panel_layout.addWidget(self.log_console_group, stretch=1)

        self.main_layout.addLayout(self.left_panel_layout, stretch=1)
        self.main_layout.addLayout(self.right_panel_layout, stretch=2)

        # Connect buttons after they are created
        self.load_dce_button.clicked.connect(self.load_dce_file)
        self.load_t1_button.clicked.connect(self.load_t1_file)
        self.load_mask_button.clicked.connect(self.load_mask_file)
        self.run_button.clicked.connect(self.run_analysis)


    def _create_file_io_section(self):
        self.file_io_group = QGroupBox("File I/O")
        layout = QVBoxLayout()

        self.load_dce_button = QPushButton("Load DCE Series")
        self.dce_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_dce_button)
        layout.addWidget(self.dce_path_label)

        self.load_t1_button = QPushButton("Load T1 Map")
        self.t1_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_t1_button)
        layout.addWidget(self.t1_path_label)

        self.load_mask_button = QPushButton("Load Mask")
        self.mask_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_mask_button)
        layout.addWidget(self.mask_path_label)

        self.file_io_group.setLayout(layout)
        # No return needed as group is stored in self

    def _create_conversion_settings_section(self):
        self.conversion_settings_group = QGroupBox("Conversion Settings")
        layout = QFormLayout()

        self.r1_input = QLineEdit()
        self.r1_input.setPlaceholderText("4.5")
        layout.addRow(QLabel("r1 Relaxivity (s⁻¹mM⁻¹):"), self.r1_input)

        self.tr_input = QLineEdit()
        self.tr_input.setPlaceholderText("0.005") # 5 ms
        layout.addRow(QLabel("TR (s):"), self.tr_input)
        
        self.te_input = QLineEdit() # TE input
        self.te_input.setPlaceholderText("0.002") # 2 ms
        layout.addRow(QLabel("TE (s):"), self.te_input)

        self.conversion_settings_group.setLayout(layout)
        # No return needed

    def _create_aif_section(self):
        self.aif_group = QGroupBox("Arterial Input Function (AIF)")
        layout = QVBoxLayout()
        self.aif_label = QLabel("AIF options placeholder") # Stored as attribute
        layout.addWidget(self.aif_label)
        self.aif_group.setLayout(layout)
        # No return needed

    def _create_model_fitting_section(self):
        self.model_fitting_group = QGroupBox("Model Fitting")
        layout = QVBoxLayout()
        self.model_fitting_label = QLabel("Model fitting options placeholder") # Stored
        layout.addWidget(self.model_fitting_label)
        self.model_fitting_group.setLayout(layout)
        # No return needed

    def _create_processing_section(self):
        self.processing_group = QGroupBox("Processing")
        layout = QVBoxLayout()
        self.run_button = QPushButton("Run Analysis") # Stored
        layout.addWidget(self.run_button)
        self.processing_group.setLayout(layout)
        # No return needed

    def _create_display_area(self):
        self.display_area_group = QGroupBox("Display Area")
        layout = QVBoxLayout()
        self.display_label = QLabel("Parameter maps and plots will appear here.") # Stored
        self.display_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.display_label)
        self.display_area_group.setLayout(layout)
        # No return needed

    def _create_log_console(self):
        self.log_console_group = QGroupBox("Log Console")
        layout = QVBoxLayout()
        self.log_console = QTextEdit() # Stored
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console)
        self.log_console_group.setLayout(layout)
        # No return needed

    def load_dce_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load DCE NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load DCE series: {filepath}")
                self.dce_data = io.load_dce_series(filepath)
                self.dce_shape_for_validation = self.dce_data.shape
                self.dce_path_label.setText(filepath)
                self.log_console.append("DCE series loaded successfully.")
                self.log_console.append(f"DCE data shape: {self.dce_data.shape}")
            except (FileNotFoundError, ValueError, Exception) as e:
                self.dce_data = None
                self.dce_shape_for_validation = None
                self.dce_path_label.setText("Error loading file")
                self.log_console.append(f"Error loading DCE: {e}\n{traceback.format_exc()}")

    def load_t1_file(self):
        if self.dce_data is None or self.dce_shape_for_validation is None:
            self.log_console.append("Please load DCE series first before loading T1 map.")
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "Load T1 Map NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load T1 map: {filepath}")
                self.t10_data = io.load_t1_map(filepath, dce_shape=self.dce_shape_for_validation)
                self.t1_path_label.setText(filepath)
                self.log_console.append("T1 map loaded successfully.")
                self.log_console.append(f"T10 data shape: {self.t10_data.shape}")
            except (FileNotFoundError, ValueError, Exception) as e:
                self.t10_data = None
                self.t1_path_label.setText("Error loading file")
                self.log_console.append(f"Error loading T1 map: {e}\n{traceback.format_exc()}")

    def load_mask_file(self):
        if self.dce_data is None or self.dce_shape_for_validation is None:
            self.log_console.append("Please load DCE series first before loading mask.")
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "Load Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load mask: {filepath}")
                self.mask_data = io.load_mask(filepath, reference_shape=self.dce_shape_for_validation[:3])
                self.mask_path_label.setText(filepath)
                self.log_console.append("Mask loaded successfully.")
                self.log_console.append(f"Mask data shape: {self.mask_data.shape}, type: {self.mask_data.dtype}")
            except (FileNotFoundError, ValueError, Exception) as e:
                self.mask_data = None
                self.mask_path_label.setText("Error loading file")
                self.log_console.append(f"Error loading mask: {e}\n{traceback.format_exc()}")

    def run_analysis(self):
        self.log_console.append("Run Analysis button clicked.")
        # a. Check data
        if self.dce_data is None or self.t10_data is None:
            self.log_console.append("Error: DCE data and T1 map must be loaded before running analysis.")
            return

        # b. Get parameters
        r1_relaxivity_str = self.r1_input.text()
        tr_str = self.tr_input.text()
        te_str = self.te_input.text() # Get TE value

        try:
            r1_relaxivity_float = float(r1_relaxivity_str)
            tr_float = float(tr_str)
            # TE is not used in current conversion, but we can parse it for future use
            if te_str: # if TE is provided, try to convert
                te_float = float(te_str)
            else: # else, maybe use a default or indicate it's not used.
                te_float = None # Or some default if your functions expect it
            
            if r1_relaxivity_float <=0 or tr_float <=0 :
                 self.log_console.append("Error: r1 relaxivity and TR must be positive values.")
                 return
            if te_float is not None and te_float <=0:
                 self.log_console.append("Error: TE must be a positive value if specified.")
                 return


        except ValueError:
            self.log_console.append("Error: Invalid r1, TR, or TE values. Please enter numeric values.")
            return

        # c. Log start
        self.log_console.append(f"Starting signal-to-concentration conversion with r1={r1_relaxivity_float}, TR={tr_float}" + (f", TE={te_float}" if te_float is not None else ""))

        # d. Call conversion function
        try:
            # Assuming signal_to_concentration takes baseline_time_points as an optional arg
            # Add a QLineEdit for baseline_time_points if it needs to be user-configurable
            self.Ct_data = conversion.signal_to_concentration(
                dce_series_data=self.dce_data,
                t10_map_data=self.t10_data,
                r1_relaxivity=r1_relaxivity_float,
                TR=tr_float,
                # baseline_time_points=5 # Default in function, or get from UI
            )
            # e. Handle results
            self.log_console.append("Signal-to-concentration conversion completed successfully.")
            self.log_console.append(f"Ct data shape: {self.Ct_data.shape}")
            # Optionally, display some result or enable further processing steps
            self.display_label.setText(f"Conversion successful.\nCt data shape: {self.Ct_data.shape}")

        # f. Catch errors
        except Exception as e:
            self.log_console.append(f"Error during conversion: {e}\n{traceback.format_exc()}")
            self.display_label.setText("Conversion failed. See log for details.")


if __name__ == '__main__':
    # This allows running the main_window.py directly for testing purposes
    # It assumes that the dce_mri_analyzer package is in the PYTHONPATH
    # or that this script is run from the directory above dce_mri_analyzer
    # For example, if dce_mri_analyzer is in /home/user/projects/dce_mri_analyzer
    # and you run `python dce_mri_analyzer/ui/main_window.py` from /home/user/projects/
    # then the relative imports should work.

    # To make it robust for direct execution, one might adjust sys.path here:
    # import os
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_dir) # This is dce_mri_analyzer
    # sys.path.insert(0, os.path.dirname(project_root)) # This is the dir containing dce_mri_analyzer

    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
