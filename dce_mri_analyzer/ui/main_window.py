import sys
import traceback
import numpy as np
import nibabel as nib 
from scipy.interpolate import interp1d 
import os 

import pyqtgraph as pg 
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
    QComboBox,
    QRadioButton,
    QSizePolicy, 
    QSlider, 
)
from PyQt5.QtCore import Qt 

from ..core import io
from ..core import conversion
from ..core import aif
from ..core import modeling 


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DCE-MRI Analysis Tool")
        self.setGeometry(100, 100, 1200, 800) 

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.dce_data = None; self.t10_data = None; self.mask_data = None
        self.Ct_data = None; self.dce_shape_for_validation = None
        self.dce_time_vector = None 
        self.dce_filepath = None; self.t1_filepath = None
        self.aif_time = None; self.aif_concentration = None
        self.Cp_interp_func = None 
        self.population_aif_time_vector = np.linspace(0, 600, 300) 
        self.selected_model_name = None
        self.parameter_maps = {} 
        self.displayable_volumes = {}  
        self.current_display_key = None 
        self.current_slice_index = 0
        self.aif_roi_object = None # For pg.RectROI

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel_layout = QVBoxLayout()
        self._create_file_io_section()
        self._create_conversion_settings_section()
        self._create_aif_section() # Will be updated
        self._create_model_fitting_section() 
        self._create_processing_section()

        self.left_panel_layout.addWidget(self.file_io_group)
        self.left_panel_layout.addWidget(self.conversion_settings_group)
        self.left_panel_layout.addWidget(self.aif_group)
        self.left_panel_layout.addWidget(self.model_fitting_group)
        self.left_panel_layout.addWidget(self.processing_section)
        self.left_panel_layout.addStretch(1)

        self.right_panel_layout = QHBoxLayout() 
        self._create_display_area() 
        self._create_log_console_and_plot_area() 

        self.main_layout.addLayout(self.left_panel_layout, stretch=1)
        self.main_layout.addLayout(self.right_panel_layout, stretch=3) 

        # Connect buttons and signals
        self.load_dce_button.clicked.connect(self.load_dce_file)
        self.load_t1_button.clicked.connect(self.load_t1_file)
        self.load_mask_button.clicked.connect(self.load_mask_file)
        self.run_button.clicked.connect(self.run_analysis)
        
        self.load_aif_button.clicked.connect(self.handle_load_aif_file)
        self.select_population_aif_button.clicked.connect(self.handle_apply_population_aif)
        self.aif_load_file_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_population_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_roi_radio.toggled.connect(self.update_aif_ui_state)
        self.draw_aif_roi_button.clicked.connect(self.handle_draw_aif_roi_button) # New connection

        self.model_standard_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_extended_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_none_radio.toggled.connect(self.handle_model_selection)
        
        self.export_ktrans_button.clicked.connect(lambda: self.export_map("Ktrans"))
        self.export_ve_button.clicked.connect(lambda: self.export_map("ve"))
        self.export_vp_button.clicked.connect(lambda: self.export_map("vp"))

        self.map_selector_combo.currentIndexChanged.connect(self.handle_map_selection_changed)
        self.slice_slider.valueChanged.connect(self.handle_slice_changed)
        self.image_view.getView().scene().sigMouseClicked.connect(self.handle_voxel_clicked)

        self.update_aif_ui_state() 
        self.handle_model_selection() 

    def _create_file_io_section(self):
        self.file_io_group = QGroupBox("File I/O")
        layout = QVBoxLayout()
        self.load_dce_button = QPushButton("Load DCE Series"); self.dce_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_dce_button); layout.addWidget(self.dce_path_label)
        self.load_t1_button = QPushButton("Load T1 Map"); self.t1_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_t1_button); layout.addWidget(self.t1_path_label)
        self.load_mask_button = QPushButton("Load Mask"); self.mask_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_mask_button); layout.addWidget(self.mask_path_label)
        self.file_io_group.setLayout(layout)

    def _create_conversion_settings_section(self):
        self.conversion_settings_group = QGroupBox("Conversion Settings (Tissue)")
        layout = QFormLayout()
        self.r1_input = QLineEdit("4.5"); layout.addRow(QLabel("r1 Relaxivity (s⁻¹mM⁻¹):"), self.r1_input)
        self.tr_input = QLineEdit("0.005"); layout.addRow(QLabel("TR (s):"), self.tr_input)
        self.te_input = QLineEdit("0.002"); layout.addRow(QLabel("TE (s):"), self.te_input)
        self.baseline_points_input = QLineEdit("5"); layout.addRow(QLabel("Baseline Pts (Tissue):"), self.baseline_points_input)
        self.conversion_settings_group.setLayout(layout)

    def _create_aif_section(self):
        self.aif_group = QGroupBox("Arterial Input Function (AIF)")
        v_layout = QVBoxLayout()

        # Radio buttons for AIF source
        self.aif_load_file_radio = QRadioButton("Load AIF from File"); self.aif_load_file_radio.setChecked(True)
        self.aif_population_radio = QRadioButton("Select Population AIF")
        self.aif_roi_radio = QRadioButton("Define AIF from ROI"); self.aif_roi_radio.setEnabled(True) # Enable ROI radio
        
        v_layout.addWidget(self.aif_load_file_radio)
        file_load_h_layout = QHBoxLayout()
        self.load_aif_button = QPushButton("Load AIF File...")
        file_load_h_layout.addWidget(self.load_aif_button)
        self.aif_file_label = QLabel("No AIF file loaded."); self.aif_file_label.setWordWrap(True)
        file_load_h_layout.addWidget(self.aif_file_label, 1) 
        v_layout.addLayout(file_load_h_layout)

        v_layout.addWidget(self.aif_population_radio)
        pop_aif_h_layout = QHBoxLayout()
        self.population_aif_combo = QComboBox()
        if aif.POPULATION_AIFS: self.population_aif_combo.addItems(aif.POPULATION_AIFS.keys())
        pop_aif_h_layout.addWidget(self.population_aif_combo)
        self.select_population_aif_button = QPushButton("Apply") 
        pop_aif_h_layout.addWidget(self.select_population_aif_button)
        v_layout.addLayout(pop_aif_h_layout)

        v_layout.addWidget(self.aif_roi_radio) 
        # ROI AIF specific inputs
        roi_aif_form_layout = QFormLayout()
        self.draw_aif_roi_button = QPushButton("Define/Redraw AIF ROI")
        roi_aif_form_layout.addRow(self.draw_aif_roi_button)
        self.aif_t10_blood_input = QLineEdit("1.4") # Default T10 blood
        roi_aif_form_layout.addRow(QLabel("AIF T10 Blood (s):"), self.aif_t10_blood_input)
        self.aif_r1_blood_input = QLineEdit("4.5") # Default r1 blood
        roi_aif_form_layout.addRow(QLabel("AIF r1 Blood (s⁻¹mM⁻¹):"), self.aif_r1_blood_input)
        self.aif_baseline_points_input = QLineEdit("5") # Default baseline points for AIF
        roi_aif_form_layout.addRow(QLabel("AIF Baseline Points:"), self.aif_baseline_points_input)
        v_layout.addLayout(roi_aif_form_layout)
        
        self.aif_status_label = QLabel("AIF: Not defined")
        v_layout.addWidget(self.aif_status_label)
        self.aif_group.setLayout(v_layout)

    def _create_model_fitting_section(self): # Unchanged from previous
        self.model_fitting_group = QGroupBox("Model Fitting")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Model:"))
        self.model_standard_tofts_radio = QRadioButton("Standard Tofts")
        self.model_extended_tofts_radio = QRadioButton("Extended Tofts")
        self.model_none_radio = QRadioButton("No Model Fitting"); self.model_none_radio.setChecked(True)
        layout.addWidget(self.model_standard_tofts_radio); layout.addWidget(self.model_extended_tofts_radio); layout.addWidget(self.model_none_radio)
        export_layout = QHBoxLayout()
        self.export_ktrans_button = QPushButton("Export Ktrans"); self.export_ktrans_button.setEnabled(False)
        self.export_ve_button = QPushButton("Export ve"); self.export_ve_button.setEnabled(False)
        self.export_vp_button = QPushButton("Export vp"); self.export_vp_button.setEnabled(False)
        export_layout.addWidget(self.export_ktrans_button); export_layout.addWidget(self.export_ve_button); export_layout.addWidget(self.export_vp_button)
        layout.addLayout(export_layout)
        self.model_fitting_group.setLayout(layout)

    def handle_model_selection(self): # Unchanged from previous
        self.export_ktrans_button.setEnabled(False); self.export_ve_button.setEnabled(False); self.export_vp_button.setEnabled(False)
        if self.model_standard_tofts_radio.isChecked(): self.selected_model_name = "Standard Tofts"
        elif self.model_extended_tofts_radio.isChecked(): self.selected_model_name = "Extended Tofts"
        else: self.selected_model_name = None
        self.log_console.append(f"Model selected: {self.selected_model_name}")

    def _create_processing_section(self): # Unchanged
        self.processing_section = QGroupBox("Processing") # Changed from self.processing_group
        layout = QVBoxLayout()
        self.run_button = QPushButton("Run Full Analysis (S->C & Voxel-wise Fitting)")
        layout.addWidget(self.run_button)
        self.processing_section.setLayout(layout)

    def _create_display_area(self): # Unchanged
        self.display_area_group = QGroupBox("Image Display") 
        display_layout = QVBoxLayout(self.display_area_group)
        self.map_selector_combo = QComboBox()
        display_layout.addWidget(self.map_selector_combo)
        slider_layout = QHBoxLayout()
        self.slice_slider_label = QLabel("Slice: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal); self.slice_slider.setEnabled(False) 
        slider_layout.addWidget(QLabel("Slice:")); slider_layout.addWidget(self.slice_slider, 1); slider_layout.addWidget(self.slice_slider_label)
        display_layout.addLayout(slider_layout)
        self.image_view = pg.ImageView()
        display_layout.addWidget(self.image_view)
        self.right_panel_layout.addWidget(self.display_area_group, stretch=2)

    def _create_log_console_and_plot_area(self): # Unchanged
        plot_log_group = QGroupBox("Plots and Log")
        plot_log_layout = QVBoxLayout(plot_log_group)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setLabel('bottom', 'Time'); self.plot_widget.setLabel('left', 'Concentration'); self.plot_widget.addLegend(offset=(-10,10)) 
        plot_log_layout.addWidget(self.plot_widget, stretch=1)
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        plot_log_layout.addWidget(self.log_console, stretch=1)
        self.right_panel_layout.addWidget(plot_log_group, stretch=1)

    def update_displayable_volume(self, name: str, data: np.ndarray): # Unchanged
        if data is None: return
        self.displayable_volumes[name] = data; current_selection_text = self.map_selector_combo.currentText()
        self.map_selector_combo.blockSignals(True); self.map_selector_combo.clear(); self.map_selector_combo.addItems(self.displayable_volumes.keys())
        idx = self.map_selector_combo.findText(name) 
        if idx != -1: self.map_selector_combo.setCurrentIndex(idx)
        elif current_selection_text and self.map_selector_combo.findText(current_selection_text) != -1: self.map_selector_combo.setCurrentText(current_selection_text)
        self.map_selector_combo.blockSignals(False)
        if self.map_selector_combo.currentText() == name: self.handle_map_selection_changed()

    def handle_map_selection_changed(self): # Unchanged
        selected_key = self.map_selector_combo.currentText()
        if not selected_key: self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); return
        self.current_display_key = selected_key; volume_data = self.displayable_volumes.get(self.current_display_key)
        if volume_data is None: self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); return
        display_data = None
        if volume_data.ndim == 3: display_data = volume_data.transpose(2, 1, 0)
        elif volume_data.ndim == 4: mean_over_time = np.mean(volume_data, axis=3); display_data = mean_over_time.transpose(2, 1, 0)
        else: self.log_console.append(f"Volume '{selected_key}' unsupported dim: {volume_data.ndim}."); self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); return
        self.image_view.setImage(display_data, autoRange=True, autoLevels=True, autoHistogramRange=True)
        num_slices = display_data.shape[0]
        self.slice_slider.setEnabled(True); self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(num_slices - 1)
        current_idx = min(self.current_slice_index, num_slices - 1)
        if current_idx < 0: current_idx = 0
        self.image_view.setCurrentIndex(current_idx); self.slice_slider.setValue(current_idx)
        self.slice_slider_label.setText(f"Slice: {current_idx + 1}/{num_slices}")

    def handle_slice_changed(self, value): # Unchanged
        self.current_slice_index = value
        if self.image_view.image is not None:
            num_slices = self.image_view.image.shape[0]; safe_value = np.clip(value, 0, num_slices - 1)
            self.image_view.setCurrentIndex(safe_value); self.slice_slider_label.setText(f"Slice: {safe_value + 1}/{num_slices}")
            if value != safe_value and self.slice_slider.value() != safe_value : self.slice_slider.setValue(safe_value)

    def handle_voxel_clicked(self, mouse_click_event): # Unchanged
        if not mouse_click_event.double(): return 
        image_item = self.image_view.getImageItem()
        if image_item is None or image_item.image is None: return
        scene_pos = mouse_click_event.scenePos(); img_coords_float = image_item.mapFromScene(scene_pos)
        y_in_slice = int(round(img_coords_float.y())); x_in_slice = int(round(img_coords_float.x())); current_z_index_in_display = self.image_view.currentIndex 
        current_slice_shape = self.image_view.image[current_z_index_in_display].shape
        if not (0 <= y_in_slice < current_slice_shape[0] and 0 <= x_in_slice < current_slice_shape[1]):
            self.log_console.append(f"Clicked outside current slice boundaries."); return
        z_orig = current_z_index_in_display; y_orig = y_in_slice; x_orig = x_in_slice
        self.log_console.append(f"Image double-clicked. Mapped to original (X:{x_orig}, Y:{y_orig}, Z:{z_orig})")
        if self.Ct_data is None or not (0 <= x_orig < self.Ct_data.shape[0] and 0 <= y_orig < self.Ct_data.shape[1] and 0 <= z_orig < self.Ct_data.shape[2]):
            self.log_console.append(f"Clicked coords ({x_orig},{y_orig},{z_orig}) outside Ct_data bounds."); return
        self.plot_selected_voxel_curves(x_orig, y_orig, z_orig)

    def plot_selected_voxel_curves(self, x_idx, y_idx, z_idx): # Unchanged
        self.plot_widget.clear(); self.plot_widget.setTitle(f"Curves for Voxel (X:{x_idx}, Y:{y_idx}, Z:{z_idx})")
        if self.Ct_data is None: self.log_console.append("Ct data not available for plotting."); return
        Ct_voxel = self.Ct_data[x_idx, y_idx, z_idx, :]; t_values = self.dce_time_vector
        if t_values is None: 
            try: tr_val = float(self.tr_input.text()); num_time_points = self.Ct_data.shape[3]; t_values = np.arange(num_time_points) * tr_val
            except ValueError: self.log_console.append("TR value invalid for plotting time axis."); return
        self.plot_widget.plot(t_values, Ct_voxel, pen=pg.mkPen('b', width=2), name='Tissue Conc.')
        if self.aif_time is not None and self.aif_concentration is not None: self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF')
        if self.selected_model_name and self.parameter_maps and self.Cp_interp_func:
            model_params = {}; valid_params = True
            if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps:
                Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]; ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]
                if np.isnan(Ktrans_val) or np.isnan(ve_val): valid_params = False
                model_params['Ktrans'] = Ktrans_val; model_params['ve'] = ve_val
            else: valid_params = False
            if self.selected_model_name == "Extended Tofts":
                if "vp" in self.parameter_maps: vp_val = self.parameter_maps["vp"][x_idx, y_idx, z_idx]; model_params['vp'] = vp_val;
                else: valid_params = False
            if valid_params:
                fitted_curve = None
                if self.selected_model_name == "Standard Tofts": fitted_curve = modeling.standard_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], self.Cp_interp_func)
                elif self.selected_model_name == "Extended Tofts": fitted_curve = modeling.extended_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], model_params['vp'], self.Cp_interp_func)
                if fitted_curve is not None: self.plot_widget.plot(t_values, fitted_curve, pen='g', name=f'{self.selected_model_name} Fit')
                self.log_console.append(f"Plotted fit for ({x_idx},{y_idx},{z_idx}).")
            else: self.log_console.append(f"No valid pre-fitted parameters for voxel ({x_idx},{y_idx},{z_idx}).")
        self.plot_widget.autoRange()

    def update_aif_ui_state(self):
        is_file_mode = self.aif_load_file_radio.isChecked()
        is_pop_mode = self.aif_population_radio.isChecked()
        is_roi_mode = self.aif_roi_radio.isChecked()

        self.load_aif_button.setEnabled(is_file_mode)
        self.aif_file_label.setEnabled(is_file_mode)
        
        self.population_aif_combo.setEnabled(is_pop_mode)
        self.select_population_aif_button.setEnabled(is_pop_mode)
        
        self.draw_aif_roi_button.setEnabled(is_roi_mode)
        self.aif_t10_blood_input.setEnabled(is_roi_mode)
        self.aif_r1_blood_input.setEnabled(is_roi_mode)
        self.aif_baseline_points_input.setEnabled(is_roi_mode)

        if not is_roi_mode and self.aif_roi_object:
            self.image_view.removeItem(self.aif_roi_object)
            self.aif_roi_object = None
            self.log_console.append("AIF ROI removed from image.")


    def handle_load_aif_file(self): # Unchanged
        filepath, _ = QFileDialog.getOpenFileName(self, "Load AIF File", "", "AIF Files (*.txt *.csv)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load AIF from: {filepath}")
                self.aif_time, self.aif_concentration = aif.load_aif_from_file(filepath)
                self.Cp_interp_func = interp1d(self.aif_time, self.aif_concentration, kind='linear', bounds_error=False, fill_value=0)
                self.aif_file_label.setText(os.path.basename(filepath)); self.aif_status_label.setText(f"AIF: Loaded from file. Points: {len(self.aif_time)}")
                self.log_console.append("AIF loaded successfully from file.")
            except Exception as e:
                self.aif_time, self.aif_concentration, self.Cp_interp_func = None, None, None
                self.aif_file_label.setText("Error loading AIF."); self.aif_status_label.setText("AIF: Error loading.")
                self.log_console.append(f"Error loading AIF from file: {e}\n{traceback.format_exc()}")

    def handle_apply_population_aif(self): # Unchanged
        model_name = self.population_aif_combo.currentText()
        if not model_name: self.log_console.append("No population AIF model selected."); return
        current_time_vector = self.dce_time_vector if self.dce_time_vector is not None else self.population_aif_time_vector
        self.log_console.append(f"Applying population AIF: {model_name} with {len(current_time_vector)} points.")
        try:
            aif_c = aif.generate_population_aif(model_name, current_time_vector)
            if aif_c is not None:
                self.aif_time, self.aif_concentration = current_time_vector, aif_c
                self.Cp_interp_func = interp1d(self.aif_time, self.aif_concentration, kind='linear', bounds_error=False, fill_value=0)
                self.aif_status_label.setText(f"AIF: Applied '{model_name}'. Points: {len(self.aif_time)}")
                self.log_console.append(f"Population AIF '{model_name}' applied.")
            else:
                self.aif_time, self.aif_concentration, self.Cp_interp_func = None, None, None
                self.aif_status_label.setText(f"AIF: Error applying '{model_name}'."); self.log_console.append(f"Failed to generate population AIF: {model_name}.")
        except Exception as e:
            self.aif_time, self.aif_concentration, self.Cp_interp_func = None, None, None
            self.aif_status_label.setText(f"AIF: Error applying '{model_name}'."); self.log_console.append(f"Error applying population AIF: {e}\n{traceback.format_exc()}")

    def handle_draw_aif_roi_button(self):
        if self.image_view.getImageItem().image is None: # Check if image is displayed
            self.log_console.append("No image displayed to draw ROI on. Please load and select a volume (e.g., Mean DCE).")
            return
            
        if self.aif_roi_object:
            self.image_view.removeItem(self.aif_roi_object)
            self.aif_roi_object = None # Ensure it's None before creating new

        # Get current displayed image data - this is already transposed (Z, Y, X)
        current_display_data = self.image_view.getImageItem().image 
        
        # ROI position and size relative to the currently displayed slice's Y and X axes
        # For example, center of the current view of the slice
        # Slice shape for current Z is (Y_orig_dim, X_orig_dim)
        slice_shape_yx = current_display_data[self.image_view.currentIndex].shape 
        
        roi_x_disp = slice_shape_yx[1] // 4  # X-axis of display (original X)
        roi_y_disp = slice_shape_yx[0] // 4  # Y-axis of display (original Y)
        roi_w_disp = slice_shape_yx[1] // 2
        roi_h_disp = slice_shape_yx[0] // 2

        self.aif_roi_object = pg.RectROI(
            pos=(roi_x_disp, roi_y_disp), 
            size=(roi_w_disp, roi_h_disp), 
            pen='r', movable=True, resizable=True, rotatable=False, hoverPen='m'
        )
        self.image_view.addItem(self.aif_roi_object)
        self.aif_roi_object.sigRegionChangeFinished.connect(self.handle_aif_roi_processing) # Connect signal
        self.log_console.append("AIF ROI created/reset. Adjust it on the image. ROI processing triggered on release.")
        self.handle_aif_roi_processing() # Initial processing

    def handle_aif_roi_processing(self):
        if self.aif_roi_object is None or self.dce_data is None:
            self.log_console.append("AIF ROI or DCE data not available for processing.")
            return

        roi_state = self.aif_roi_object.getState()
        # roi_state['pos'] and roi_state['size'] are in (X_disp, Y_disp) coords of the image item
        # Image item data is (Z_orig, Y_orig, X_orig)
        x_roi_disp = int(round(roi_state['pos'].x())) # Corresponds to original X axis
        y_roi_disp = int(round(roi_state['pos'].y())) # Corresponds to original Y axis
        w_roi_disp = int(round(roi_state['size'].x()))
        h_roi_disp = int(round(roi_state['size'].y()))
        
        z_orig_slice = self.image_view.currentIndex # This is the original Z index

        # Validate ROI bounds against the original dimensions of the slice in dce_data
        # dce_data shape is (X, Y, Z, T)
        if not (0 <= x_roi_disp < self.dce_data.shape[0] and \
                0 <= y_roi_disp < self.dce_data.shape[1] and \
                x_roi_disp + w_roi_disp <= self.dce_data.shape[0] and \
                y_roi_disp + h_roi_disp <= self.dce_data.shape[1]):
            self.log_console.append(f"AIF ROI (X:{x_roi_disp}-{x_roi_disp+w_roi_disp}, Y:{y_roi_disp}-{y_roi_disp+h_roi_disp}) "
                                    f"is outside original data boundaries for slice Z={z_orig_slice}. Adjust ROI.")
            return
        if w_roi_disp <=0 or h_roi_disp <=0:
            self.log_console.append("AIF ROI width or height is zero/negative. Adjust ROI.")
            return

        # roi_2d_coords_orig is (x_start, y_start, width, height) in original X,Y index space
        roi_2d_coords_orig = (x_roi_disp, y_roi_disp, w_roi_disp, h_roi_disp)
        
        try:
            t10_b_str = self.aif_t10_blood_input.text()
            r1_b_str = self.aif_r1_blood_input.text()
            tr_val_str = self.tr_input.text() # TR from main conversion settings
            aif_baseline_pts_str = self.aif_baseline_points_input.text()

            if not all([t10_b_str, r1_b_str, tr_val_str, aif_baseline_pts_str]):
                self.log_console.append("One or more AIF ROI parameters (T10, r1, TR, baseline) are empty.")
                return

            t10_b = float(t10_b_str)
            r1_b = float(r1_b_str)
            tr_val = float(tr_val_str) # Use TR from main conversion settings
            aif_baseline_pts = int(aif_baseline_pts_str)

            if t10_b <=0 or r1_b <=0 or tr_val <=0 or aif_baseline_pts <=0:
                self.log_console.append("AIF ROI parameters (T10, r1, TR, baseline) must be positive.")
                return

            self.log_console.append(f"Processing AIF ROI: Slice Z={z_orig_slice}, Coords (orig X,Y)=({x_roi_disp},{y_roi_disp}), Size=({w_roi_disp},{h_roi_disp})")
            self.aif_time, self.aif_concentration = aif.extract_aif_from_roi(
                self.dce_data, roi_2d_coords_orig, z_orig_slice, t10_b, r1_b, tr_val, aif_baseline_pts
            )
            self.Cp_interp_func = interp1d(self.aif_time, self.aif_concentration, kind='linear', bounds_error=False, fill_value=0)
            self.aif_status_label.setText(f"AIF from ROI (Z={z_orig_slice}) processed. Points: {len(self.aif_time)}")
            self.log_console.append(f"AIF extracted from ROI. Points: {len(self.aif_time)}. Plotting AIF.")
            
            # Plot the newly extracted AIF
            self.plot_widget.clear()
            self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF (ROI)')
            self.plot_widget.autoRange()

        except ValueError as ve:
            self.log_console.append(f"Invalid AIF ROI parameters (T10, r1, TR, baseline points): {ve}")
        except Exception as e:
            self.log_console.append(f"Error processing AIF ROI: {e}\n{traceback.format_exc()}")


    def load_dce_file(self): # Unchanged logic, just ensure it's present
        filepath, _ = QFileDialog.getOpenFileName(self, "Load DCE NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            self.dce_filepath = filepath 
            try:
                self.log_console.append(f"Loading DCE series: {filepath}")
                self.dce_data = io.load_dce_series(filepath) 
                self.dce_shape_for_validation = self.dce_data.shape
                self.dce_path_label.setText(os.path.basename(filepath))
                self.log_console.append(f"DCE series loaded. Shape: {self.dce_data.shape}")
                self.update_displayable_volume("Original DCE (Mean)", np.mean(self.dce_data, axis=3))
                try:
                    tr_val_str = self.tr_input.text()
                    if not tr_val_str: self.log_console.append("TR not set. DCE time vector not defined."); self.dce_time_vector = None; return
                    tr_val = float(tr_val_str)
                    if tr_val > 0: self.dce_time_vector = np.arange(self.dce_data.shape[3]) * tr_val; self.log_console.append(f"DCE time vector defined: {len(self.dce_time_vector)} points, TR={tr_val}s.")
                    else: self.log_console.append("TR not positive. DCE time vector not defined."); self.dce_time_vector = None
                except ValueError: self.log_console.append("TR invalid. DCE time vector not defined."); self.dce_time_vector = None
            except Exception as e:
                self.dce_data, self.dce_shape_for_validation, self.dce_time_vector, self.dce_filepath = None, None, None, None
                self.dce_path_label.setText("Error loading file"); self.log_console.append(f"Error loading DCE: {e}\n{traceback.format_exc()}")

    def load_t1_file(self): # Unchanged
        if self.dce_data is None: self.log_console.append("Load DCE series first."); return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load T1 Map NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            self.t1_filepath = filepath 
            try:
                self.log_console.append(f"Loading T1 map: {filepath}")
                self.t10_data = io.load_t1_map(filepath, dce_shape=self.dce_shape_for_validation)
                self.t1_path_label.setText(os.path.basename(filepath)); self.log_console.append(f"T1 map loaded. Shape: {self.t10_data.shape}")
                self.update_displayable_volume("T1 Map", self.t10_data)
            except Exception as e:
                self.t10_data, self.t1_filepath = None, None; self.t1_path_label.setText("Error loading file"); self.log_console.append(f"Error loading T1 map: {e}\n{traceback.format_exc()}")

    def load_mask_file(self): # Unchanged
        if self.dce_data is None: self.log_console.append("Load DCE series first."); return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Loading mask: {filepath}")
                self.mask_data = io.load_mask(filepath, reference_shape=self.dce_shape_for_validation[:3])
                self.mask_path_label.setText(os.path.basename(filepath)); self.log_console.append(f"Mask loaded. Shape: {self.mask_data.shape}, Type: {self.mask_data.dtype}")
                self.update_displayable_volume("Mask", self.mask_data.astype(np.uint8))
            except Exception as e:
                self.mask_data = None; self.mask_path_label.setText("Error loading file"); self.log_console.append(f"Error loading mask: {e}\n{traceback.format_exc()}")

    def run_analysis(self): # Unchanged
        self.log_console.append("Run Analysis button clicked."); self.display_label.setText("Processing... See log for details."); QApplication.processEvents() 
        if self.dce_data is None or self.t10_data is None: self.log_console.append("Error: DCE data and T1 map must be loaded."); self.display_label.setText("Analysis failed: DCE or T1 data missing."); return
        if self.aif_time is None or self.aif_concentration is None or self.Cp_interp_func is None : self.log_console.append("Error: AIF not defined/loaded or interpolator not created."); self.display_label.setText("Analysis failed: AIF not defined."); return
        try:
            r1_val = float(self.r1_input.text()); tr_val = float(self.tr_input.text()); baseline_pts = int(self.baseline_points_input.text())
            if r1_val <= 0 or tr_val <= 0 or baseline_pts <= 0: raise ValueError("Params must be positive.")
            if baseline_pts >= self.dce_data.shape[3]: raise ValueError("Baseline points exceed total time points.")
        except ValueError as e: self.log_console.append(f"Error: Invalid conversion parameters: {e}"); self.display_label.setText(f"Analysis failed: Invalid params ({e})."); return
        self.log_console.append(f"Starting S-to-C conversion: r1={r1_val}, TR={tr_val}, baseline={baseline_pts}"); QApplication.processEvents()
        try:
            self.Ct_data = conversion.signal_to_concentration(self.dce_data, self.t10_data, r1_val, tr_val, baseline_pts)
            self.log_console.append(f"S-to-C conversion successful. Ct_data shape: {self.Ct_data.shape}"); self.display_label.setText(f"Conversion successful. Ct shape: {self.Ct_data.shape}")
            self.update_displayable_volume("Ct (Concentration Mean)", np.mean(self.Ct_data, axis=3))
        except Exception as e: self.log_console.append(f"Error during S-to-C: {e}\n{traceback.format_exc()}"); self.display_label.setText("Conversion failed."); self.Ct_data = None; return
        QApplication.processEvents()
        if not self.selected_model_name: self.log_console.append("No model selected. Skipping voxel-wise fitting."); self.display_label.setText("Conversion done. No model selected for fitting."); return
        if self.Ct_data is None: self.log_console.append("Ct_data not available. Skipping model fitting."); self.display_label.setText("Ct data not available. Fitting skipped."); return
        self.log_console.append(f"Starting voxel-wise {self.selected_model_name} model fitting. This may take time..."); self.display_label.setText(f"Fitting {self.selected_model_name} voxel-wise... This may take a while."); QApplication.processEvents()
        t_tissue = self.dce_time_vector 
        if t_tissue is None: 
             if tr_val > 0: t_tissue = np.arange(self.Ct_data.shape[3]) * tr_val
             else: self.log_console.append("Error: Cannot determine t_tissue for fitting."); self.display_label.setText("Fitting failed: t_tissue unknown."); return
        mask_to_use = self.mask_data if self.mask_data is not None else None; self.parameter_maps = {} 
        try:
            if self.selected_model_name == "Standard Tofts": self.parameter_maps = modeling.fit_standard_tofts_voxelwise(self.Ct_data, t_tissue, self.Cp_interp_func, mask=mask_to_use)
            elif self.selected_model_name == "Extended Tofts": self.parameter_maps = modeling.fit_extended_tofts_voxelwise(self.Ct_data, t_tissue, self.Cp_interp_func, mask=mask_to_use)
            self.log_console.append(f"Voxel-wise {self.selected_model_name} fitting completed."); self.display_label.setText(f"{self.selected_model_name} fitting done. Maps generated: {', '.join(self.parameter_maps.keys())}")
            for map_name, map_data in self.parameter_maps.items(): self.update_displayable_volume(map_name, map_data)
            self.export_ktrans_button.setEnabled("Ktrans" in self.parameter_maps); self.export_ve_button.setEnabled("ve" in self.parameter_maps); self.export_vp_button.setEnabled("vp" in self.parameter_maps and self.selected_model_name == "Extended Tofts")
        except Exception as e: self.log_console.append(f"Error during voxel-wise fitting: {e}\n{traceback.format_exc()}"); self.display_label.setText(f"Voxel-wise fitting failed. See log.")
        QApplication.processEvents()

    def export_map(self, map_name: str): # Unchanged
        self.log_console.append(f"Export map button clicked for: {map_name}"); param_map_data = self.parameter_maps.get(map_name)
        if param_map_data is None: self.log_console.append(f"Error: {map_name} map not available for export."); return
        reference_nifti_path = self.t1_filepath if self.t1_filepath else self.dce_filepath
        if not reference_nifti_path: self.log_console.append("Error: No reference NIfTI (T1 or DCE) loaded for header info."); return
        default_filename = f"{map_name}_map.nii.gz"; output_filepath, _ = QFileDialog.getSaveFileName(self, f"Save {map_name} Map", default_filename, "NIfTI files (*.nii *.nii.gz)")
        if output_filepath:
            try:
                self.log_console.append(f"Saving {map_name} map to: {output_filepath} using ref: {reference_nifti_path}")
                io.save_nifti_map(param_map_data, reference_nifti_path, output_filepath)
                self.log_console.append(f"{map_name} map saved successfully."); self.display_label.setText(f"{map_name} map saved to {os.path.basename(output_filepath)}")
            except Exception as e: self.log_console.append(f"Error saving {map_name} map: {e}\n{traceback.format_exc()}"); self.display_label.setText(f"Error saving {map_name} map. See log.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
