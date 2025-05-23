import sys
import traceback
import numpy as np
import nibabel as nib 
from scipy.interpolate import interp1d 
from scipy.integrate import cumtrapz 
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
    QSpinBox, 
)
from PyQt5.QtCore import Qt, QPointF 

from ..core import io
from ..core import conversion
from ..core import aif
from ..core import modeling 
from ..core import reporting # Added

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DCE-MRI Analysis Tool")
        self.setGeometry(100, 100, 1200, 850) # Increased height slightly for stats display

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Data attributes
        self.dce_data = None; self.t10_data = None; self.mask_data = None
        self.Ct_data = None; self.dce_shape_for_validation = None
        self.dce_time_vector = None 
        self.dce_filepath = None; self.t1_filepath = None
        # AIF attributes
        self.aif_time = None; self.aif_concentration = None
        self.Cp_interp_func = None 
        self.integral_Cp_dt_interp_func = None 
        self.population_aif_time_vector = np.linspace(0, 600, 300) 
        # Model attributes
        self.selected_model_name = None
        self.parameter_maps = {} 
        # Display attributes
        self.displayable_volumes = {}  
        self.current_display_key = None 
        self.current_slice_index = 0
        self.aif_roi_object = None 
        # Stats ROI attributes
        self.stats_roi_object = None
        self.current_roi_stats = None


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel_layout = QVBoxLayout()
        self._create_file_io_section()
        self._create_conversion_settings_section()
        self._create_aif_section() 
        self._create_model_fitting_section() 
        self._create_processing_section() 
        self._create_roi_stats_section() # New section for stats ROI

        self.left_panel_layout.addWidget(self.file_io_group)
        self.left_panel_layout.addWidget(self.conversion_settings_group)
        self.left_panel_layout.addWidget(self.aif_group)
        self.left_panel_layout.addWidget(self.model_fitting_group)
        self.left_panel_layout.addWidget(self.processing_section) 
        self.left_panel_layout.addWidget(self.roi_stats_group) # Add new group
        self.left_panel_layout.addStretch(1)

        self.right_panel_layout = QHBoxLayout() 
        self._create_display_area() 
        self._create_log_console_and_plot_area() 

        self.main_layout.addLayout(self.left_panel_layout, stretch=1)
        self.main_layout.addLayout(self.right_panel_layout, stretch=3) 

        # Connect signals
        self._connect_signals()
        
        self.update_aif_ui_state() 
        self.handle_model_selection() 

    def _connect_signals(self):
        self.load_dce_button.clicked.connect(self.load_dce_file)
        self.load_t1_button.clicked.connect(self.load_t1_file)
        self.load_mask_button.clicked.connect(self.load_mask_file)
        self.run_button.clicked.connect(self.run_analysis)
        
        self.load_aif_button.clicked.connect(self.handle_load_aif_file)
        self.select_population_aif_button.clicked.connect(self.handle_apply_population_aif)
        self.aif_load_file_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_population_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_roi_radio.toggled.connect(self.update_aif_ui_state)
        self.draw_aif_roi_button.clicked.connect(self.handle_draw_aif_roi_button) 
        self.save_aif_roi_button.clicked.connect(self.handle_save_aif_roi_def) 
        self.load_aif_roi_button.clicked.connect(self.handle_load_aif_roi_def) 

        self.model_standard_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_extended_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_patlak_radio.toggled.connect(self.handle_model_selection) 
        self.model_none_radio.toggled.connect(self.handle_model_selection)
        
        self.export_ktrans_button.clicked.connect(lambda: self.export_map("Ktrans"))
        self.export_ve_button.clicked.connect(lambda: self.export_map("ve"))
        self.export_vp_button.clicked.connect(lambda: self.export_map("vp"))
        self.export_ktrans_patlak_button.clicked.connect(lambda: self.export_map("Ktrans_patlak")) 
        self.export_vp_patlak_button.clicked.connect(lambda: self.export_map("vp_patlak"))       

        self.map_selector_combo.currentIndexChanged.connect(self.handle_map_selection_changed)
        self.slice_slider.valueChanged.connect(self.handle_slice_changed)
        self.image_view.getView().scene().sigMouseClicked.connect(self.handle_voxel_clicked)

        # Stats ROI signals
        self.draw_stats_roi_button.clicked.connect(self.handle_draw_stats_roi)
        self.save_stats_button.clicked.connect(self.handle_save_roi_stats)


    def _create_file_io_section(self): # Unchanged
        self.file_io_group = QGroupBox("File I/O")
        layout = QVBoxLayout()
        self.load_dce_button = QPushButton("Load DCE Series"); self.dce_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_dce_button); layout.addWidget(self.dce_path_label)
        self.load_t1_button = QPushButton("Load T1 Map"); self.t1_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_t1_button); layout.addWidget(self.t1_path_label)
        self.load_mask_button = QPushButton("Load Mask"); self.mask_path_label = QLabel("Not loaded")
        layout.addWidget(self.load_mask_button); layout.addWidget(self.mask_path_label)
        self.file_io_group.setLayout(layout)

    def _create_conversion_settings_section(self): # Unchanged
        self.conversion_settings_group = QGroupBox("Conversion Settings (Tissue)")
        layout = QFormLayout()
        self.r1_input = QLineEdit("4.5"); layout.addRow(QLabel("r1 Relaxivity (s⁻¹mM⁻¹):"), self.r1_input)
        self.tr_input = QLineEdit("0.005"); layout.addRow(QLabel("TR (s):"), self.tr_input)
        self.te_input = QLineEdit("0.002"); layout.addRow(QLabel("TE (s):"), self.te_input)
        self.baseline_points_input = QLineEdit("5"); layout.addRow(QLabel("Baseline Pts (Tissue):"), self.baseline_points_input)
        self.conversion_settings_group.setLayout(layout)

    def _create_aif_section(self): # Unchanged
        self.aif_group = QGroupBox("Arterial Input Function (AIF)")
        v_layout = QVBoxLayout()
        self.aif_load_file_radio = QRadioButton("Load AIF from File"); self.aif_load_file_radio.setChecked(True)
        self.aif_population_radio = QRadioButton("Select Population AIF")
        self.aif_roi_radio = QRadioButton("Define AIF from ROI"); self.aif_roi_radio.setEnabled(True)
        v_layout.addWidget(self.aif_load_file_radio)
        file_load_h_layout = QHBoxLayout()
        self.load_aif_button = QPushButton("Load AIF File..."); file_load_h_layout.addWidget(self.load_aif_button)
        self.aif_file_label = QLabel("No AIF file loaded."); self.aif_file_label.setWordWrap(True); file_load_h_layout.addWidget(self.aif_file_label, 1) 
        v_layout.addLayout(file_load_h_layout)
        v_layout.addWidget(self.aif_population_radio)
        pop_aif_h_layout = QHBoxLayout()
        self.population_aif_combo = QComboBox(); pop_aif_h_layout.addWidget(self.population_aif_combo)
        if aif.POPULATION_AIFS: self.population_aif_combo.addItems(aif.POPULATION_AIFS.keys())
        self.select_population_aif_button = QPushButton("Apply"); pop_aif_h_layout.addWidget(self.select_population_aif_button)
        v_layout.addLayout(pop_aif_h_layout)
        v_layout.addWidget(self.aif_roi_radio) 
        roi_aif_controls_layout = QVBoxLayout() 
        roi_aif_buttons_layout = QHBoxLayout()
        self.draw_aif_roi_button = QPushButton("Define/Redraw AIF ROI"); roi_aif_buttons_layout.addWidget(self.draw_aif_roi_button)
        self.save_aif_roi_button = QPushButton("Save AIF ROI Def."); self.save_aif_roi_button.setEnabled(False); roi_aif_buttons_layout.addWidget(self.save_aif_roi_button)
        self.load_aif_roi_button = QPushButton("Load AIF ROI Def."); roi_aif_buttons_layout.addWidget(self.load_aif_roi_button)
        roi_aif_controls_layout.addLayout(roi_aif_buttons_layout)
        roi_aif_form_layout = QFormLayout()
        self.aif_t10_blood_input = QLineEdit("1.4"); roi_aif_form_layout.addRow(QLabel("AIF T10 Blood (s):"), self.aif_t10_blood_input)
        self.aif_r1_blood_input = QLineEdit("4.5"); roi_aif_form_layout.addRow(QLabel("AIF r1 Blood (s⁻¹mM⁻¹):"), self.aif_r1_blood_input)
        self.aif_baseline_points_input = QLineEdit("5"); roi_aif_form_layout.addRow(QLabel("AIF Baseline Points:"), self.aif_baseline_points_input)
        roi_aif_controls_layout.addLayout(roi_aif_form_layout); v_layout.addLayout(roi_aif_controls_layout)
        self.aif_status_label = QLabel("AIF: Not defined"); v_layout.addWidget(self.aif_status_label)
        self.aif_group.setLayout(v_layout)

    def _create_model_fitting_section(self): # Unchanged
        self.model_fitting_group = QGroupBox("Model Fitting")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Model:"))
        self.model_standard_tofts_radio = QRadioButton("Standard Tofts")
        self.model_extended_tofts_radio = QRadioButton("Extended Tofts")
        self.model_patlak_radio = QRadioButton("Patlak Model") 
        self.model_none_radio = QRadioButton("No Model Fitting"); self.model_none_radio.setChecked(True)
        layout.addWidget(self.model_standard_tofts_radio); layout.addWidget(self.model_extended_tofts_radio); layout.addWidget(self.model_patlak_radio); layout.addWidget(self.model_none_radio)
        export_layout = QHBoxLayout()
        self.export_ktrans_button = QPushButton("Export Ktrans"); self.export_ktrans_button.setEnabled(False)
        self.export_ve_button = QPushButton("Export ve"); self.export_ve_button.setEnabled(False)
        self.export_vp_button = QPushButton("Export vp"); self.export_vp_button.setEnabled(False)
        export_layout.addWidget(self.export_ktrans_button); export_layout.addWidget(self.export_ve_button); export_layout.addWidget(self.export_vp_button)
        self.export_ktrans_patlak_button = QPushButton("Export Ktrans (Patlak)"); self.export_ktrans_patlak_button.setEnabled(False) 
        self.export_vp_patlak_button = QPushButton("Export vp (Patlak)"); self.export_vp_patlak_button.setEnabled(False)       
        export_layout.addWidget(self.export_ktrans_patlak_button); export_layout.addWidget(self.export_vp_patlak_button)
        layout.addLayout(export_layout)
        self.model_fitting_group.setLayout(layout)

    def _create_roi_stats_section(self): # New
        self.roi_stats_group = QGroupBox("ROI Statistics")
        layout = QVBoxLayout()
        
        self.draw_stats_roi_button = QPushButton("Draw/Reset Stats ROI")
        layout.addWidget(self.draw_stats_roi_button)
        
        self.stats_results_display = QTextEdit()
        self.stats_results_display.setReadOnly(True)
        self.stats_results_display.setPlaceholderText("ROI statistics will appear here.")
        self.stats_results_display.setFixedHeight(150) # Set a fixed height
        layout.addWidget(self.stats_results_display)
        
        self.save_stats_button = QPushButton("Save ROI Stats")
        self.save_stats_button.setEnabled(False)
        layout.addWidget(self.save_stats_button)
        
        self.roi_stats_group.setLayout(layout)


    def handle_model_selection(self): # Unchanged
        self.export_ktrans_button.setEnabled(False); self.export_ve_button.setEnabled(False); self.export_vp_button.setEnabled(False)
        self.export_ktrans_patlak_button.setEnabled(False); self.export_vp_patlak_button.setEnabled(False)
        if self.model_standard_tofts_radio.isChecked(): self.selected_model_name = "Standard Tofts"
        elif self.model_extended_tofts_radio.isChecked(): self.selected_model_name = "Extended Tofts"
        elif self.model_patlak_radio.isChecked(): self.selected_model_name = "Patlak" 
        else: self.selected_model_name = None
        self.log_console.append(f"Model selected: {self.selected_model_name}")
        self.update_export_buttons_state() 

    def _create_processing_section(self): # Unchanged
        self.processing_section = QGroupBox("Processing") 
        layout = QFormLayout() 
        self.num_processes_input = QSpinBox(); self.num_processes_input.setMinimum(1)
        num_cpus = os.cpu_count(); self.num_processes_input.setMaximum(num_cpus if num_cpus else 1); self.num_processes_input.setValue(num_cpus if num_cpus else 1)
        layout.addRow(QLabel("Number of Cores for Fitting:"), self.num_processes_input)
        self.run_button = QPushButton("Run Full Analysis (S->C & Voxel-wise Fitting)"); layout.addRow(self.run_button)
        self.processing_section.setLayout(layout)

    def _create_display_area(self): # Unchanged
        self.display_area_group = QGroupBox("Image Display") 
        display_layout = QVBoxLayout(self.display_area_group)
        self.map_selector_combo = QComboBox(); display_layout.addWidget(self.map_selector_combo)
        slider_layout = QHBoxLayout()
        self.slice_slider_label = QLabel("Slice: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal); self.slice_slider.setEnabled(False) 
        slider_layout.addWidget(QLabel("Slice:")); slider_layout.addWidget(self.slice_slider, 1); slider_layout.addWidget(self.slice_slider_label)
        display_layout.addLayout(slider_layout)
        self.image_view = pg.ImageView(); display_layout.addWidget(self.image_view)
        self.right_panel_layout.addWidget(self.display_area_group, stretch=2)

    def _create_log_console_and_plot_area(self): # Unchanged
        plot_log_group = QGroupBox("Plots and Log")
        plot_log_layout = QVBoxLayout(plot_log_group)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setLabel('bottom', 'Time'); self.plot_widget.setLabel('left', 'Concentration'); self.plot_widget.addLegend(offset=(-10,10)) 
        plot_log_layout.addWidget(self.plot_widget, stretch=1)
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True); plot_log_layout.addWidget(self.log_console, stretch=1)
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

    def handle_map_selection_changed(self): # Modified to clear stats ROI if map changes
        if self.stats_roi_object: # Remove stats ROI if map changes
            self.image_view.removeItem(self.stats_roi_object)
            self.stats_roi_object = None
            self.stats_results_display.clear()
            self.save_stats_button.setEnabled(False)
            self.current_roi_stats = None
            self.log_console.append("Stats ROI cleared due to map change.")

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

    def handle_slice_changed(self, value): # Modified to update stats ROI if it exists
        self.current_slice_index = value
        if self.image_view.image is not None:
            num_slices = self.image_view.image.shape[0]; safe_value = np.clip(value, 0, num_slices - 1)
            self.image_view.setCurrentIndex(safe_value); self.slice_slider_label.setText(f"Slice: {safe_value + 1}/{num_slices}")
            if value != safe_value and self.slice_slider.value() != safe_value : self.slice_slider.setValue(safe_value)
            if self.stats_roi_object: # If stats ROI exists, re-calculate stats for the new slice
                self.handle_stats_roi_updated()


    def handle_voxel_clicked(self, mouse_click_event): # Unchanged
        if not mouse_click_event.double(): return 
        image_item = self.image_view.getImageItem()
        if image_item is None or image_item.image is None: return
        scene_pos = mouse_click_event.scenePos(); img_coords_float = image_item.mapFromScene(scene_pos)
        y_in_slice = int(round(img_coords_float.y())); x_in_slice = int(round(img_coords_float.x())); current_z_index_in_display = self.image_view.currentIndex 
        current_slice_shape = self.image_view.image[current_z_index_in_display].shape
        if not (0 <= y_in_slice < current_slice_shape[0] and 0 <= x_in_slice < current_slice_shape[1]): self.log_console.append(f"Clicked outside current slice boundaries."); return
        z_orig = current_z_index_in_display; y_orig = y_in_slice; x_orig = x_in_slice
        self.log_console.append(f"Image double-clicked. Mapped to original (X:{x_orig}, Y:{y_orig}, Z:{z_orig})")
        if self.Ct_data is None or not (0 <= x_orig < self.Ct_data.shape[0] and 0 <= y_orig < self.Ct_data.shape[1] and 0 <= z_orig < self.Ct_data.shape[2]): self.log_console.append(f"Clicked coords ({x_orig},{y_orig},{z_orig}) outside Ct_data bounds."); return
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
            model_params = {}; valid_params = True; fitted_curve = None
            if self.selected_model_name == "Standard Tofts":
                if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps: Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]; ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]; 
                if np.isnan(Ktrans_val) or np.isnan(ve_val): valid_params = False; else: model_params['Ktrans'] = Ktrans_val; model_params['ve'] = ve_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.standard_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], self.Cp_interp_func)
            elif self.selected_model_name == "Extended Tofts":
                if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps and "vp" in self.parameter_maps: Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]; ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]; vp_val = self.parameter_maps["vp"][x_idx, y_idx, z_idx];
                if np.isnan(Ktrans_val) or np.isnan(ve_val) or np.isnan(vp_val): valid_params = False; else: model_params['Ktrans'] = Ktrans_val; model_params['ve'] = ve_val; model_params['vp'] = vp_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.extended_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], model_params['vp'], self.Cp_interp_func)
            elif self.selected_model_name == "Patlak": 
                if "Ktrans_patlak" in self.parameter_maps and "vp_patlak" in self.parameter_maps and self.integral_Cp_dt_interp_func: Ktrans_val = self.parameter_maps["Ktrans_patlak"][x_idx, y_idx, z_idx]; vp_val = self.parameter_maps["vp_patlak"][x_idx, y_idx, z_idx];
                if np.isnan(Ktrans_val) or np.isnan(vp_val): valid_params = False; else: model_params['Ktrans_patlak'] = Ktrans_val; model_params['vp_patlak'] = vp_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.patlak_model(t_values, model_params['Ktrans_patlak'], model_params['vp_patlak'], self.Cp_interp_func, self.integral_Cp_dt_interp_func)
            if valid_params and fitted_curve is not None: self.plot_widget.plot(t_values, fitted_curve, pen='g', name=f'{self.selected_model_name} Fit'); param_str = ", ".join([f"{k}={v:.3f}" for k,v in model_params.items()]); self.log_console.append(f"Plotted fit for ({x_idx},{y_idx},{z_idx}). Params: {param_str}")
            elif valid_params and fitted_curve is None: self.log_console.append(f"Fit parameters valid but curve generation failed for {self.selected_model_name} at ({x_idx},{y_idx},{z_idx}).")
            else: self.log_console.append(f"No valid pre-fitted parameters for voxel ({x_idx},{y_idx},{z_idx}) for {self.selected_model_name}.")
        self.plot_widget.autoRange()

    def update_aif_ui_state(self): # Unchanged
        is_file_mode = self.aif_load_file_radio.isChecked(); is_pop_mode = self.aif_population_radio.isChecked(); is_roi_mode = self.aif_roi_radio.isChecked()
        self.load_aif_button.setEnabled(is_file_mode); self.aif_file_label.setEnabled(is_file_mode)
        self.population_aif_combo.setEnabled(is_pop_mode); self.select_population_aif_button.setEnabled(is_pop_mode)
        self.draw_aif_roi_button.setEnabled(is_roi_mode); self.save_aif_roi_button.setEnabled(is_roi_mode and self.aif_roi_object is not None); self.load_aif_roi_button.setEnabled(is_roi_mode)
        self.aif_t10_blood_input.setEnabled(is_roi_mode); self.aif_r1_blood_input.setEnabled(is_roi_mode); self.aif_baseline_points_input.setEnabled(is_roi_mode)
        if not is_roi_mode and self.aif_roi_object: self.image_view.removeItem(self.aif_roi_object); self.aif_roi_object = None; self.log_console.append("AIF ROI removed from image."); self.save_aif_roi_button.setEnabled(False)

    def _create_aif_interpolators(self): # Unchanged
        if self.aif_time is not None and self.aif_concentration is not None and len(self.aif_time) > 1:
            try:
                self.Cp_interp_func = interp1d(self.aif_time, self.aif_concentration, kind='linear', bounds_error=False, fill_value=0.0)
                integral_Cp_dt_aif = cumtrapz(self.aif_concentration, self.aif_time, initial=0)
                self.integral_Cp_dt_interp_func = interp1d(self.aif_time, integral_Cp_dt_aif, kind='linear', bounds_error=False, fill_value=0.0)
                return True
            except Exception as e: self.log_console.append(f"Failed to create AIF interpolators: {e}"); self.Cp_interp_func = None; self.integral_Cp_dt_interp_func = None; return False
        else: self.Cp_interp_func = None; self.integral_Cp_dt_interp_func = None; 
        if self.aif_time is not None: self.log_console.append("AIF time or concentration has insufficient points for interpolation."); return False
        return False

    def handle_load_aif_file(self): # Unchanged
        filepath, _ = QFileDialog.getOpenFileName(self, "Load AIF File", "", "AIF Files (*.txt *.csv)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load AIF from: {filepath}"); self.aif_time, self.aif_concentration = aif.load_aif_from_file(filepath)
                if self._create_aif_interpolators(): self.aif_file_label.setText(os.path.basename(filepath)); self.aif_status_label.setText(f"AIF: Loaded from file. Points: {len(self.aif_time)}"); self.log_console.append("AIF loaded successfully from file.")
                else: self.aif_time, self.aif_concentration = None, None; self.aif_file_label.setText("Error creating interpolator."); self.aif_status_label.setText("AIF: Error processing.")
            except Exception as e: self.aif_time, self.aif_concentration, self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None, None, None; self.aif_file_label.setText("Error loading AIF."); self.aif_status_label.setText("AIF: Error loading."); self.log_console.append(f"Error loading AIF from file: {e}\n{traceback.format_exc()}")

    def handle_apply_population_aif(self): # Unchanged
        model_name = self.population_aif_combo.currentText()
        if not model_name: self.log_console.append("No population AIF model selected."); return
        current_time_vector = self.dce_time_vector if self.dce_time_vector is not None else self.population_aif_time_vector
        self.log_console.append(f"Applying population AIF: {model_name} with {len(current_time_vector)} points.")
        try:
            aif_c = aif.generate_population_aif(model_name, current_time_vector)
            if aif_c is not None:
                self.aif_time, self.aif_concentration = current_time_vector, aif_c
                if self._create_aif_interpolators(): self.aif_status_label.setText(f"AIF: Applied '{model_name}'. Points: {len(self.aif_time)}"); self.log_console.append(f"Population AIF '{model_name}' applied.")
                else: self.aif_time, self.aif_concentration = None, None; self.aif_status_label.setText(f"AIF: Error processing '{model_name}'.")
            else: self.aif_time, self.aif_concentration, self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None, None, None; self.aif_status_label.setText(f"AIF: Error applying '{model_name}'."); self.log_console.append(f"Failed to generate population AIF: {model_name}.")
        except Exception as e: self.aif_time, self.aif_concentration, self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None, None, None; self.aif_status_label.setText(f"AIF: Error applying '{model_name}'."); self.log_console.append(f"Error applying population AIF: {e}\n{traceback.format_exc()}")

    def handle_draw_aif_roi_button(self): # Unchanged
        if self.image_view.getImageItem().image is None: self.log_console.append("No image displayed to draw ROI on."); return
        if self.aif_roi_object: self.image_view.removeItem(self.aif_roi_object); self.aif_roi_object = None 
        current_display_data = self.image_view.getImageItem().image; slice_shape_yx = current_display_data[self.image_view.currentIndex].shape 
        roi_x_disp = slice_shape_yx[1] // 4; roi_y_disp = slice_shape_yx[0] // 4; roi_w_disp = slice_shape_yx[1] // 2; roi_h_disp = slice_shape_yx[0] // 2
        self.aif_roi_object = pg.RectROI(pos=(roi_x_disp, roi_y_disp), size=(roi_w_disp, roi_h_disp), pen='r', movable=True, resizable=True, rotatable=False, hoverPen='m')
        self.image_view.addItem(self.aif_roi_object); self.aif_roi_object.sigRegionChangeFinished.connect(self.handle_aif_roi_processing) 
        self.log_console.append("AIF ROI created/reset. Adjust it on the image."); self.handle_aif_roi_processing()
        self.save_aif_roi_button.setEnabled(True) 

    def handle_aif_roi_processing(self): # Unchanged
        if self.aif_roi_object is None or self.dce_data is None: self.log_console.append("AIF ROI or DCE data not available for processing."); return
        roi_state = self.aif_roi_object.getState(); x_roi_disp = int(round(roi_state['pos'].x())); y_roi_disp = int(round(roi_state['pos'].y())); w_roi_disp = int(round(roi_state['size'].x())); h_roi_disp = int(round(roi_state['size'].y()))
        z_orig_slice = self.image_view.currentIndex
        if not (0 <= x_roi_disp < self.dce_data.shape[0] and 0 <= y_roi_disp < self.dce_data.shape[1] and x_roi_disp + w_roi_disp <= self.dce_data.shape[0] and y_roi_disp + h_roi_disp <= self.dce_data.shape[1]): self.log_console.append(f"AIF ROI is outside original data boundaries. Adjust ROI."); return
        if w_roi_disp <=0 or h_roi_disp <=0: self.log_console.append("AIF ROI width or height is zero/negative. Adjust ROI."); return
        roi_2d_coords_orig = (x_roi_disp, y_roi_disp, w_roi_disp, h_roi_disp)
        try:
            t10_b_str = self.aif_t10_blood_input.text(); r1_b_str = self.aif_r1_blood_input.text(); tr_val_str = self.tr_input.text(); aif_baseline_pts_str = self.aif_baseline_points_input.text()
            if not all([t10_b_str, r1_b_str, tr_val_str, aif_baseline_pts_str]): self.log_console.append("One or more AIF ROI parameters are empty."); return
            t10_b = float(t10_b_str); r1_b = float(r1_b_str); tr_val = float(tr_val_str); aif_baseline_pts = int(aif_baseline_pts_str)
            if t10_b <=0 or r1_b <=0 or tr_val <=0 or aif_baseline_pts <=0: self.log_console.append("AIF ROI parameters must be positive."); return
            self.log_console.append(f"Processing AIF ROI: Slice Z={z_orig_slice}, Coords (orig X,Y)=({x_roi_disp},{y_roi_disp}), Size=({w_roi_disp},{h_roi_disp})")
            self.aif_time, self.aif_concentration = aif.extract_aif_from_roi(self.dce_data, roi_2d_coords_orig, z_orig_slice, t10_b, r1_b, tr_val, aif_baseline_pts)
            if self._create_aif_interpolators(): self.aif_status_label.setText(f"AIF from ROI (Z={z_orig_slice}) processed. Points: {len(self.aif_time)}"); self.log_console.append(f"AIF extracted from ROI. Points: {len(self.aif_time)}. Plotting AIF."); self.plot_widget.clear(); self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF (ROI)'); self.plot_widget.autoRange()
            else: self.aif_time, self.aif_concentration = None, None; self.aif_status_label.setText("AIF: Error processing ROI AIF.")
        except ValueError as ve: self.log_console.append(f"Invalid AIF ROI parameters: {ve}")
        except Exception as e: self.log_console.append(f"Error processing AIF ROI: {e}\n{traceback.format_exc()}")

    def handle_save_aif_roi_def(self): # Unchanged
        if self.aif_roi_object is None or not self.aif_roi_radio.isChecked(): self.log_console.append("No active AIF ROI to save or ROI mode not selected."); return
        roi_state = self.aif_roi_object.getState(); current_image_key = self.map_selector_combo.currentText()
        if not current_image_key or current_image_key not in self.displayable_volumes: self.log_console.append("Cannot determine reference image for ROI."); return
        slice_idx_in_display = self.image_view.currentIndex
        roi_properties = {"slice_index": slice_idx_in_display, "pos_x": roi_state['pos'].x(), "pos_y": roi_state['pos'].y(), "size_w": roi_state['size'].x(), "size_h": roi_state['size'].y(), "image_ref_name": current_image_key}
        filepath, _ = QFileDialog.getSaveFileName(self, "Save AIF ROI Definition", "", "JSON files (*.json)")
        if filepath:
            try: aif.save_aif_roi_definition(roi_properties, filepath); self.log_console.append(f"AIF ROI definition saved to {filepath}")
            except Exception as e: self.log_console.append(f"Error saving AIF ROI: {e}\n{traceback.format_exc()}")

    def handle_load_aif_roi_def(self): # Unchanged
        if not self.aif_roi_radio.isChecked(): self.aif_roi_radio.setChecked(True); QApplication.processEvents()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load AIF ROI Definition", "", "JSON files (*.json)")
        if filepath:
            try:
                roi_props = aif.load_aif_roi_definition(filepath)
                if roi_props is None: self.log_console.append("Failed to load AIF ROI definition."); return
                ref_img_name = roi_props.get("image_ref_name")
                if ref_img_name not in self.displayable_volumes: self.log_console.append(f"Required image '{ref_img_name}' for ROI not loaded."); return
                if self.map_selector_combo.currentText() != ref_img_name:
                    idx = self.map_selector_combo.findText(ref_img_name)
                    if idx != -1: self.map_selector_combo.setCurrentIndex(idx); QApplication.processEvents()
                    else: self.log_console.append(f"Could not select reference image '{ref_img_name}'."); return
                if self.image_view.getImageItem().image is None: self.log_console.append(f"Reference image '{ref_img_name}' not displayed."); return
                slice_to_set = roi_props.get("slice_index", 0)
                if not (0 <= slice_to_set < self.image_view.image.shape[0]): self.log_console.append(f"Warning: Saved slice index {slice_to_set} out of bounds. Using current slice."); slice_to_set = self.image_view.currentIndex
                self.image_view.setCurrentIndex(slice_to_set); self.slice_slider.setValue(slice_to_set)
                if self.aif_roi_object: self.image_view.removeItem(self.aif_roi_object)
                pos_roi = (roi_props["pos_x"], roi_props["pos_y"]); size_roi = (roi_props["size_w"], roi_props["size_h"])
                self.aif_roi_object = pg.RectROI(pos=pos_roi, size=size_roi, pen='r', movable=True, resizable=True, rotatable=False, hoverPen='m')
                self.image_view.addItem(self.aif_roi_object); self.aif_roi_object.sigRegionChangeFinished.connect(self.handle_aif_roi_processing); self.handle_aif_roi_processing() 
                self.log_console.append(f"AIF ROI definition loaded from {filepath} and applied."); self.save_aif_roi_button.setEnabled(True) 
            except Exception as e: self.log_console.append(f"Error loading AIF ROI: {e}\n{traceback.format_exc()}")

    def load_dce_file(self): # Unchanged
        filepath, _ = QFileDialog.getOpenFileName(self, "Load DCE NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            self.dce_filepath = filepath 
            try:
                self.log_console.append(f"Loading DCE series: {filepath}"); self.dce_data = io.load_dce_series(filepath) 
                self.dce_shape_for_validation = self.dce_data.shape; self.dce_path_label.setText(os.path.basename(filepath))
                self.log_console.append(f"DCE series loaded. Shape: {self.dce_data.shape}"); self.update_displayable_volume("Original DCE (Mean)", np.mean(self.dce_data, axis=3))
                try:
                    tr_val_str = self.tr_input.text()
                    if not tr_val_str: self.log_console.append("TR not set. DCE time vector not defined."); self.dce_time_vector = None; return
                    tr_val = float(tr_val_str)
                    if tr_val > 0: self.dce_time_vector = np.arange(self.dce_data.shape[3]) * tr_val; self.log_console.append(f"DCE time vector defined: {len(self.dce_time_vector)} points, TR={tr_val}s.")
                    else: self.log_console.append("TR not positive. DCE time vector not defined."); self.dce_time_vector = None
                except ValueError: self.log_console.append("TR invalid. DCE time vector not defined."); self.dce_time_vector = None
            except Exception as e: self.dce_data, self.dce_shape_for_validation, self.dce_time_vector, self.dce_filepath = None, None, None, None; self.dce_path_label.setText("Error loading file"); self.log_console.append(f"Error loading DCE: {e}\n{traceback.format_exc()}")

    def load_t1_file(self): # Unchanged
        if self.dce_data is None: self.log_console.append("Load DCE series first."); return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load T1 Map NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            self.t1_filepath = filepath 
            try:
                self.log_console.append(f"Loading T1 map: {filepath}"); self.t10_data = io.load_t1_map(filepath, dce_shape=self.dce_shape_for_validation)
                self.t1_path_label.setText(os.path.basename(filepath)); self.log_console.append(f"T1 map loaded. Shape: {self.t10_data.shape}"); self.update_displayable_volume("T1 Map", self.t10_data)
            except Exception as e: self.t10_data, self.t1_filepath = None, None; self.t1_path_label.setText("Error loading file"); self.log_console.append(f"Error loading T1 map: {e}\n{traceback.format_exc()}")

    def load_mask_file(self): # Unchanged
        if self.dce_data is None: self.log_console.append("Load DCE series first."); return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Loading mask: {filepath}"); self.mask_data = io.load_mask(filepath, reference_shape=self.dce_shape_for_validation[:3])
                self.mask_path_label.setText(os.path.basename(filepath)); self.log_console.append(f"Mask loaded. Shape: {self.mask_data.shape}, Type: {self.mask_data.dtype}"); self.update_displayable_volume("Mask", self.mask_data.astype(np.uint8))
            except Exception as e: self.mask_data = None; self.mask_path_label.setText("Error loading file"); self.log_console.append(f"Error loading mask: {e}\n{traceback.format_exc()}")

    def run_analysis(self): # Unchanged
        self.log_console.append("Run Analysis button clicked."); self.display_label.setText("Processing... See log for details."); QApplication.processEvents() 
        if self.dce_data is None or self.t10_data is None: self.log_console.append("Error: DCE data and T1 map must be loaded."); self.display_label.setText("Analysis failed: DCE or T1 data missing."); return
        if self.aif_time is None or self.aif_concentration is None: self.log_console.append("Error: AIF not defined/loaded."); self.display_label.setText("Analysis failed: AIF not defined."); return
        if not self._create_aif_interpolators(): self.log_console.append("Error: Failed to create AIF interpolators. Cannot run analysis."); self.display_label.setText("Analysis failed: AIF interpolation error."); return
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
        num_cores_to_use = self.num_processes_input.value()
        self.log_console.append(f"Starting parallel voxel-wise {self.selected_model_name} model fitting using up to {num_cores_to_use} cores...")
        self.display_label.setText(f"Fitting {self.selected_model_name} voxel-wise (up to {num_cores_to_use} cores)... This may take a while."); QApplication.processEvents()
        t_tissue = self.dce_time_vector 
        if t_tissue is None: 
             if tr_val > 0: t_tissue = np.arange(self.Ct_data.shape[3]) * tr_val
             else: self.log_console.append("Error: Cannot determine t_tissue for fitting."); self.display_label.setText("Fitting failed: t_tissue unknown."); return
        mask_to_use = self.mask_data if self.mask_data is not None else None; self.parameter_maps = {} 
        try:
            if self.selected_model_name == "Standard Tofts": self.parameter_maps = modeling.fit_standard_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Extended Tofts": self.parameter_maps = modeling.fit_extended_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Patlak": self.parameter_maps = modeling.fit_patlak_model_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            self.log_console.append(f"Parallel voxel-wise {self.selected_model_name} fitting completed."); self.display_label.setText(f"{self.selected_model_name} fitting done. Maps generated: {', '.join(self.parameter_maps.keys())}")
            for map_name, map_data in self.parameter_maps.items(): self.update_displayable_volume(map_name, map_data)
            self.update_export_buttons_state() 
        except Exception as e: self.log_console.append(f"Error during voxel-wise fitting: {e}\n{traceback.format_exc()}"); self.display_label.setText(f"Voxel-wise fitting failed. See log.")
        QApplication.processEvents()

    def update_export_buttons_state(self): # Unchanged
        self.export_ktrans_button.setEnabled("Ktrans" in self.parameter_maps and self.selected_model_name in ["Standard Tofts", "Extended Tofts"])
        self.export_ve_button.setEnabled("ve" in self.parameter_maps and self.selected_model_name in ["Standard Tofts", "Extended Tofts"])
        self.export_vp_button.setEnabled("vp" in self.parameter_maps and self.selected_model_name == "Extended Tofts")
        self.export_ktrans_patlak_button.setEnabled("Ktrans_patlak" in self.parameter_maps and self.selected_model_name == "Patlak")
        self.export_vp_patlak_button.setEnabled("vp_patlak" in self.parameter_maps and self.selected_model_name == "Patlak")

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

    # --- New Stats ROI Methods ---
    def handle_draw_stats_roi(self):
        if self.stats_roi_object: 
            self.image_view.removeItem(self.stats_roi_object)
            self.stats_roi_object = None
        
        current_img_item = self.image_view.getImageItem()
        if current_img_item is None or current_img_item.image is None: 
            self.log_console.append("No image displayed to draw Stats ROI on. Please load and select a volume.")
            return
            
        view_data_shape = current_img_item.image[self.image_view.currentIndex].shape # Y_orig, X_orig of current slice
        roi_y_disp = view_data_shape[0] // 4 
        roi_x_disp = view_data_shape[1] // 4 
        roi_h_disp = view_data_shape[0] // 2
        roi_w_disp = view_data_shape[1] // 2

        self.stats_roi_object = pg.RectROI(
            pos=(roi_x_disp, roi_y_disp), 
            size=(roi_w_disp, roi_h_disp), 
            pen=pg.mkPen('g', width=2), # Green pen for Stats ROI
            movable=True, resizable=True, hoverPen=pg.mkPen('m', width=2), rotatable=False
        )
        self.image_view.addItem(self.stats_roi_object)
        self.stats_roi_object.sigRegionChangeFinished.connect(self.handle_stats_roi_updated)
        self.handle_stats_roi_updated() # Initial calculation
        self.save_stats_button.setEnabled(False) # Enable only after stats are valid and non-zero N

    def handle_stats_roi_updated(self):
        if self.stats_roi_object is None: return
        
        current_map_name = self.map_selector_combo.currentText()
        if not current_map_name or current_map_name not in self.displayable_volumes:
            self.log_console.append("Select a map first to calculate ROI statistics.")
            self.stats_results_display.clear()
            self.save_stats_button.setEnabled(False)
            self.current_roi_stats = None
            return
            
        img_item = self.image_view.getImageItem()
        if img_item is None or img_item.image is None: 
            self.stats_results_display.clear()
            self.save_stats_button.setEnabled(False)
            self.current_roi_stats = None
            return

        z_idx_display = self.image_view.currentIndex # This is the original Z index
        
        # Get the original 3D data for the currently selected map
        original_volume_data = self.displayable_volumes.get(current_map_name)
        if original_volume_data is None or original_volume_data.ndim !=3 : # Stats only on 3D maps
             self.log_console.append(f"Statistics ROI can only be applied to 3D parameter maps. '{current_map_name}' is not suitable.")
             self.stats_results_display.setText(f"Stats ROI not applicable to '{current_map_name}'.")
             self.save_stats_button.setEnabled(False)
             self.current_roi_stats = None
             return

        current_slice_data_original_orientation = original_volume_data[:, :, z_idx_display] # X, Y for original Z slice

        roi_state = self.stats_roi_object.getState()
        # ROI coords from pg.RectROI are (x_disp, y_disp) for pos and size
        # Displayed image is (Y_orig, X_orig) for the current slice Z_orig
        # So, roi_state['pos'].x() is index along X_orig axis
        # roi_state['pos'].y() is index along Y_orig axis
        x_start_orig = int(round(roi_state['pos'].x()))
        y_start_orig = int(round(roi_state['pos'].y()))
        w_orig = int(round(roi_state['size'].x()))
        h_orig = int(round(roi_state['size'].y()))

        slice_cols_orig, slice_rows_orig = current_slice_data_original_orientation.shape # Shape is (X_orig, Y_orig)

        roi_mask_on_slice = np.zeros_like(current_slice_data_original_orientation, dtype=bool)

        x_start_clipped = max(0, x_start_orig)
        y_start_clipped = max(0, y_start_orig)
        x_end_clipped = min(x_start_orig + w_orig, slice_cols_orig)
        y_end_clipped = min(y_start_orig + h_orig, slice_rows_orig)

        if y_start_clipped < y_end_clipped and x_start_clipped < x_end_clipped:
            roi_mask_on_slice[x_start_clipped:x_end_clipped, y_start_clipped:y_end_clipped] = True
        
        self.current_roi_stats = reporting.calculate_roi_statistics(current_slice_data_original_orientation, roi_mask_on_slice)
        roi_name_str = f"StatsROI_slice{z_idx_display}"
        formatted_stats_str = reporting.format_roi_statistics_to_string(self.current_roi_stats, current_map_name, roi_name_str)
        self.stats_results_display.setText(formatted_stats_str)
        self.log_console.append(f"Stats calculated for ROI on '{current_map_name}', slice {z_idx_display}.")
        self.save_stats_button.setEnabled(self.current_roi_stats is not None and self.current_roi_stats.get("N_valid",0) > 0)


    def handle_save_roi_stats(self):
        if self.current_roi_stats is None or self.current_roi_stats.get("N_valid", 0) == 0:
            self.log_console.append("No valid ROI statistics to save.")
            return
        
        current_map_name = self.map_selector_combo.currentText()
        # z_idx_display = self.image_view.currentIndex # Original Z index
        # Get current slice from the stats_roi_object if possible, or use image_view's current index
        # For simplicity, assume image_view.currentIndex is correct for the slice where stats were computed
        z_idx_display = self.current_slice_index # or self.image_view.currentIndex

        roi_name_str = f"StatsROI_slice{z_idx_display}_on_{current_map_name.replace(' ','_').replace('(','').replace(')','')}"
        default_filename = f"{roi_name_str}_stats.csv"
        filepath, _ = QFileDialog.getSaveFileName(self, "Save ROI Statistics", default_filename, "CSV files (*.csv)")
        
        if filepath:
            try: 
                reporting.save_roi_statistics_csv(self.current_roi_stats, filepath, current_map_name, roi_name_str)
                self.log_console.append(f"ROI statistics saved to {filepath}")
            except Exception as e: 
                self.log_console.append(f"Error saving ROI stats: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    if sys.platform.startswith('win'): 
        import multiprocessing 
        multiprocessing.freeze_support() 
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
