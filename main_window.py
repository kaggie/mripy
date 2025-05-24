from PyQt5.QtWidgets import QMainWindow, QDockWidget, QTextEdit, QAction, QFileDialog
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import os
import pydicom
import nibabel
from PIL import Image
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Editor")
        self.setGeometry(100, 100, 1200, 800)  # Increased size for docks

        # Create an ImageView widget (main image display)
        self.image_view_widget = pg.ImageView() # Renamed from self.imageView

        # Image Viewer Dock
        self.image_viewer_dock = QDockWidget("Image Viewer", self)
        self.image_viewer_dock.setWidget(self.image_view_widget) # Use renamed widget
        self.addDockWidget(Qt.LeftDockWidgetArea, self.image_viewer_dock)

        # Metadata Panel Dock
        self.metadata_panel_dock = QDockWidget("Metadata Panel", self)
        metadata_placeholder = QTextEdit("Metadata will be displayed here.")
        metadata_placeholder.setReadOnly(True)
        self.metadata_panel_dock.setWidget(metadata_placeholder)
        self.addDockWidget(Qt.RightDockWidgetArea, self.metadata_panel_dock)

        # Toolbox Dock
        self.toolbox_dock = QDockWidget("Toolbox", self)
        toolbox_placeholder = QTextEdit("Tools will be available here.")
        toolbox_placeholder.setReadOnly(True)
        self.toolbox_dock.setWidget(toolbox_placeholder)
        self.addDockWidget(Qt.RightDockWidgetArea, self.toolbox_dock)
        
        # File Explorer Dock
        self.file_explorer_dock = QDockWidget("File Explorer", self)
        file_explorer_placeholder = QTextEdit("File system will be browsable here.")
        file_explorer_placeholder.setReadOnly(True)
        self.file_explorer_dock.setWidget(file_explorer_placeholder)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_explorer_dock)

        # Remove the central widget explicitly, as docks will form the layout
        self.setCentralWidget(None)

        # --- Menu Bar ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_file_action = QAction("Open File...", self)
        open_file_action.triggered.connect(self.open_file) 
        file_menu.addAction(open_file_action)

        open_dir_action = QAction("Open Directory...", self)
        open_dir_action.triggered.connect(self.open_directory) 
        file_menu.addAction(open_dir_action)

        save_action = QAction("Save", self)
        # save_action.triggered.connect(self.save_file) # Connect later
        file_menu.addAction(save_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        
        view_menu.addAction(self.image_viewer_dock.toggleViewAction())
        view_menu.addAction(self.metadata_panel_dock.toggleViewAction())
        view_menu.addAction(self.toolbox_dock.toggleViewAction())
        view_menu.addAction(self.file_explorer_dock.toggleViewAction())
        
        view_menu.addSeparator()
        reset_layout_action = QAction("Reset Layout", self)
        # reset_layout_action.triggered.connect(self.reset_layout) # Connect later
        view_menu.addAction(reset_layout_action)

        # Tools Menu
        self.tools_menu = menu_bar.addMenu("&Tools") # Changed to self.tools_menu
        # Populate later

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("About", self)
        # about_action.triggered.connect(self.show_about_dialog) # Connect later
        help_menu.addAction(about_action)

        docs_action = QAction("Documentation...", self)
        # docs_action.triggered.connect(self.open_documentation) # Connect later
        help_menu.addAction(docs_action)

    def load_image(self, filepath):
        try:
            filename = os.path.basename(filepath)
            extension = os.path.splitext(filename)[1].lower()
            image_data = None

            print(f"Attempting to load: {filepath}")

            if extension in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(filepath)
                image_data = dicom_data.pixel_array
                print(f"Successfully loaded DICOM: {filename}, shape: {image_data.shape}")
                self.display_dicom_metadata(filepath) # Call to display metadata
            elif extension in ['.nii', '.nii.gz']:
                nifti_data = nibabel.load(filepath)
                image_data = nifti_data.get_fdata()
                print(f"Successfully loaded NIfTI: {filename}, shape: {image_data.shape}")
            elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                pil_image = Image.open(filepath)
                image_data = np.array(pil_image)
                print(f"Successfully loaded image: {filename}, shape: {image_data.shape}")
            else:
                print(f"Unsupported file type: {filename}")
                return None
            
            return image_data

        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

    def open_file(self):
        supported_formats = [
            "DICOM files (*.dcm *.dicom)",
            "NIfTI files (*.nii *.nii.gz)",
            "PNG files (*.png)",
            "JPEG files (*.jpg *.jpeg)",
            "BMP files (*.bmp)",
            "TIFF files (*.tif *.tiff)",
            "All files (*)"
        ]
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "", # Start directory
            ";;".join(supported_formats)
        )
        if filepath:
            print(f"Selected file: {filepath}")
            image_data = self.load_image(filepath)
            if image_data is not None:
                self.image_view_widget.setImage(image_data)
                # print(f"Image data loaded successfully. Shape: {image_data.shape}") # Removed
            else:
                print("Failed to load image data.")

    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Open Image Directory",
            "" # Start directory
        )
        if dir_path:
            print(f"Selected directory: {dir_path}")
            supported_extensions = ['.dcm', '.dicom', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if os.path.isfile(filepath):
                    extension = os.path.splitext(filename)[1].lower()
                    if extension in supported_extensions:
                        print(f"Found supported file: {filepath}")
                        image_data = self.load_image(filepath)
                        if image_data is not None:
                            self.image_view_widget.setImage(image_data) # Display last loaded image
                            # print(f"Image data from directory loaded. Shape: {image_data.shape}") # Removed
                        else:
                            print(f"Failed to load image data for {filename} from directory.")
                    else:
                        print(f"Skipping unsupported file in directory: {filename}")

    def keyPressEvent(self, event):
        if self.image_view_widget.image is not None:
            img_shape = self.image_view_widget.image.shape
            current_index = self.image_view_widget.currentIndex

            # Check if it's a 3D or 4D image (slicing applies to the first axis)
            if len(img_shape) >= 3:
                num_slices = img_shape[0]
                
                if event.key() == Qt.Key_Up:
                    current_index -= 1
                    if current_index < 0:
                        current_index = 0
                    self.image_view_widget.setCurrentIndex(current_index)
                    event.accept()
                    return
                elif event.key() == Qt.Key_Down:
                    current_index += 1
                    if current_index >= num_slices:
                        current_index = num_slices - 1
                    self.image_view_widget.setCurrentIndex(current_index)
                    event.accept()
                    return
                # For Left/Right, we'll also control the first axis for now.
                # Alternatively, one could choose a different axis if data is 4D,
                # but ImageView's setCurrentIndex primarily affects the first axis.
                elif event.key() == Qt.Key_Left:
                    current_index -= 1
                    if current_index < 0:
                        current_index = 0
                    self.image_view_widget.setCurrentIndex(current_index)
                    event.accept()
                    return
                elif event.key() == Qt.Key_Right:
                    current_index += 1
                    if current_index >= num_slices:
                        current_index = num_slices - 1
                    self.image_view_widget.setCurrentIndex(current_index)
                    event.accept()
                    return

        super().keyPressEvent(event) # Pass on to default handler if not used

    def display_dicom_metadata(self, filepath):
        """Reads and displays key DICOM metadata in the metadata panel."""
        try:
            ds = pydicom.dcmread(filepath)
            metadata_str = f"File: {os.path.basename(filepath)}\n\n"
            metadata_str += f"Patient Name: {getattr(ds, 'PatientName', 'N/A')}\n"
            metadata_str += f"Patient ID: {getattr(ds, 'PatientID', 'N/A')}\n"
            metadata_str += f"Patient Birth Date: {getattr(ds, 'PatientBirthDate', 'N/A')}\n"
            metadata_str += f"Patient Sex: {getattr(ds, 'PatientSex', 'N/A')}\n\n"
            
            metadata_str += f"Study UID: {getattr(ds, 'StudyInstanceUID', 'N/A')}\n"
            metadata_str += f"Study Description: {getattr(ds, 'StudyDescription', 'N/A')}\n"
            metadata_str += f"Study Date: {getattr(ds, 'StudyDate', 'N/A')}\n"
            metadata_str += f"Study Time: {getattr(ds, 'StudyTime', 'N/A')}\n\n"
            
            metadata_str += f"Series UID: {getattr(ds, 'SeriesInstanceUID', 'N/A')}\n"
            metadata_str += f"Series Description: {getattr(ds, 'SeriesDescription', 'N/A')}\n"
            metadata_str += f"Series Number: {getattr(ds, 'SeriesNumber', 'N/A')}\n"
            metadata_str += f"Modality: {getattr(ds, 'Modality', 'N/A')}\n\n"
            
            metadata_str += f"SOP Instance UID: {getattr(ds, 'SOPInstanceUID', 'N/A')}\n"
            metadata_str += f"Instance Number: {getattr(ds, 'InstanceNumber', 'N/A')}\n"
            metadata_str += f"Image Type: {getattr(ds, 'ImageType', 'N/A')}\n"
            
            # Access the QTextEdit widget within the metadata_panel_dock
            metadata_text_edit = self.metadata_panel_dock.widget()
            if isinstance(metadata_text_edit, QTextEdit):
                metadata_text_edit.setText(metadata_str)
            else:
                print("Error: Metadata panel widget is not a QTextEdit.")
            print(f"Displayed metadata for: {filepath}")
        except Exception as e:
            print(f"Error reading or displaying DICOM metadata for {filepath}: {e}")
            metadata_text_edit = self.metadata_panel_dock.widget()
            if isinstance(metadata_text_edit, QTextEdit):
                metadata_text_edit.setText(f"Could not read metadata for {os.path.basename(filepath)}.\nError: {e}")

    def open_specific_file(self, filepath):
        """Opens a specific image file and displays it."""
        print(f"Opening specific file: {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return

        image_data = self.load_image(filepath) # load_image now calls display_dicom_metadata if it's a DICOM
        if image_data is not None:
            self.image_view_widget.setImage(image_data)
            print(f"Successfully displayed image from: {filepath}")
            # If it's not a DICOM, metadata won't be shown by load_image, clear panel or show basic info
            if not (filepath.lower().endswith('.dcm') or filepath.lower().endswith('.dicom')):
                metadata_text_edit = self.metadata_panel_dock.widget()
                if isinstance(metadata_text_edit, QTextEdit):
                     metadata_text_edit.setText(f"File: {os.path.basename(filepath)}\n\n(Non-DICOM file, no detailed metadata to display)")
        else:
            print(f"Failed to load image data from: {filepath}")


    def load_plugins(self): # New method in MainWindow
        # Example of explicit plugin loading
        from plugins.mrsi_fitting_plugin import MRSIFittingPlugin # Import here to avoid circularity if plugin imports MainWindow
        from plugins.dmri_fitting_plugin import DMRICorePlugin # Added import for dMRI plugin
        from plugins.pacs_plugin.pacs_plugin import PACSTransferPlugin # Added import for PACS plugin
        
        if not hasattr(self, 'loaded_plugins'):
            self.loaded_plugins = []

        # Load MRSI Plugin
        try:
            mrsi_plugin = MRSIFittingPlugin()
            mrsi_plugin.initialize(self)
            self.loaded_plugins.append(mrsi_plugin)
            print("Loaded MRSIFittingPlugin.")
        except ImportError:
            print("MRSIFittingPlugin not found or could not be imported.")
        except Exception as e:
            print(f"Error loading MRSIFittingPlugin: {e}")


        # Load dMRI Plugin
        try:
            dmri_plugin = DMRICorePlugin()
            dmri_plugin.initialize(self)
            self.loaded_plugins.append(dmri_plugin)
            print("Loaded DMRICorePlugin.")
        except ImportError:
            print("DMRICorePlugin not found or could not be imported.")
        except Exception as e:
            print(f"Error loading DMRICorePlugin: {e}")

        # Load PACS Transfer Plugin
        try:
            pacs_plugin = PACSTransferPlugin()
            pacs_plugin.initialize(self)
            self.loaded_plugins.append(pacs_plugin)
            print("Loaded PACSTransferPlugin.")
        except ImportError:
            print("PACSTransferPlugin not found or could not be imported.")
        except Exception as e:
            print(f"Error loading PACSTransferPlugin: {e}")
