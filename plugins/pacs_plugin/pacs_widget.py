# plugins/pacs_plugin/pacs_widget.py

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt # For Qt.ItemIsEditable if needed later

class PACSWidget(QWidget):
    """
    Widget for PACS Plugin UI.
    Provides interface for PACS configuration, querying, and retrieving DICOMs.
    """
    def __init__(self, plugin_instance, parent=None): # Added plugin_instance
        super().__init__(parent)
        self.plugin_instance = plugin_instance # Store plugin_instance

        # Main layout
        main_layout = QVBoxLayout(self)

        # PACS Configuration Group
        pacs_config_group = QGroupBox("PACS Configuration")
        pacs_config_layout = QFormLayout()

        self.hostname_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.our_aet_edit = QLineEdit()
        self.server_aet_edit = QLineEdit()
        self.test_connection_button = QPushButton("Test Connection")

        pacs_config_layout.addRow(QLabel("Hostname:"), self.hostname_edit)
        pacs_config_layout.addRow(QLabel("Port:"), self.port_edit)
        pacs_config_layout.addRow(QLabel("AE Title (Our):"), self.our_aet_edit)
        pacs_config_layout.addRow(QLabel("AE Title (Server):"), self.server_aet_edit)
        pacs_config_layout.addRow(self.test_connection_button)
        pacs_config_group.setLayout(pacs_config_layout)
        main_layout.addWidget(pacs_config_group)

        # Query & Retrieve Group
        query_retrieve_group = QGroupBox("Query & Retrieve")
        query_retrieve_layout = QVBoxLayout() # Using QVBoxLayout as per suggestion

        # Search criteria area (simple for now)
        search_area_layout = QHBoxLayout() # Use QHBoxLayout for label and line edit
        self.query_terms_edit = QLineEdit()
        self.search_dicoms_button = QPushButton("Search DICOMs")
        search_area_layout.addWidget(QLabel("Search Terms:")) # Added label for clarity
        search_area_layout.addWidget(self.query_terms_edit)
        search_area_layout.addWidget(self.search_dicoms_button)
        query_retrieve_layout.addLayout(search_area_layout)

        # Storage Directory Selection Area
        storage_dir_form_layout = QFormLayout() # Renamed for clarity
        self.storage_dir_line_edit = QLineEdit("./dicom_storage")
        self.browse_storage_dir_button = QPushButton("Browse...")
        self.browse_storage_dir_button.clicked.connect(self.browse_storage_directory)
        
        storage_dir_hbox = QHBoxLayout() 
        storage_dir_hbox.addWidget(self.storage_dir_line_edit)
        storage_dir_hbox.addWidget(self.browse_storage_dir_button)
        storage_dir_form_layout.addRow(QLabel("Storage Directory:"), storage_dir_hbox)
        query_retrieve_layout.addLayout(storage_dir_form_layout)

        # View Saved DICOM Button
        self.view_saved_dicom_button = QPushButton("View Saved DICOM")
        self.view_saved_dicom_button.clicked.connect(self.view_saved_dicom_file)
        query_retrieve_layout.addWidget(self.view_saved_dicom_button) # Add button to the QVBoxLayout

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7) # Patient Name, Study ID, Series ID, Modality, Description, Date, Select
        self.results_table.setHorizontalHeaderLabels([
            "Patient Name", "Study ID", "Series ID", "Modality",
            "Description", "Date", "Select"
        ])
        # Allow row selection
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        query_retrieve_layout.addWidget(self.results_table)

        self.retrieve_dicoms_button = QPushButton("Retrieve Selected DICOMs")
        query_retrieve_layout.addWidget(self.retrieve_dicoms_button)

        query_retrieve_group.setLayout(query_retrieve_layout)
        main_layout.addWidget(query_retrieve_group)

        # Log/Status Area
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        main_layout.addWidget(QLabel("Log/Status:")) # Added a label for the log area
        main_layout.addWidget(self.log_text_edit)

        self.setLayout(main_layout)

    def view_saved_dicom_file(self):
        """Opens a file dialog to select a DICOM file from the storage directory and loads it."""
        if not self.plugin_instance or not self.plugin_instance.main_window:
            print("Error: Main window reference not available.") # Or log to widget
            return

        storage_dir = self.storage_dir_line_edit.text().strip()
        if not storage_dir:
            storage_dir = "./dicom_storage/" # Default if empty
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Saved DICOM File",
            storage_dir, # Start directory
            "DICOM files (*.dcm *.dicom);;All files (*)"
        )
        
        if filepath:
            main_win = self.plugin_instance.main_window
            if hasattr(main_win, 'open_specific_file'):
                main_win.open_specific_file(filepath)
            else:
                # Fallback or error if method doesn't exist (should be created in main_window.py)
                print(f"Error: MainWindow.open_specific_file method not found. Attempting generic load.")
                # As a fallback, could try to use the existing open_file logic if it were refactored
                # or simply load via main_win.load_image and main_win.image_view_widget.setImage
                image_data = main_win.load_image(filepath)
                if image_data is not None:
                    main_win.image_view_widget.setImage(image_data)
                    if hasattr(main_win, 'display_dicom_metadata'): # Check if metadata display exists
                         main_win.display_dicom_metadata(filepath)
                else:
                    print(f"Failed to load image data for {filepath} via fallback.")


    def browse_storage_directory(self):
        """Opens a dialog to select a directory for storing DICOM files."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Storage Directory",
            self.storage_dir_line_edit.text() # Start from current directory in line edit
        )
        if directory: # If a directory was selected
            self.storage_dir_line_edit.setText(directory)

    def clear_results_table(self):
        """Clears all rows from the results table."""
        self.results_table.setRowCount(0)

    def add_search_result_row(self, data_dict):
        """
        Adds a new row to the results_table and populates it with data from data_dict.
        Expected keys in data_dict: "PatientName", "StudyID", "SeriesID", 
                                     "Modality", "Description", "Date".
        The "Select" column is added as a placeholder.
        """
        row_position = self.results_table.rowCount()
        self.results_table.insertRow(row_position)

        # Order of columns in the table:
        # "Patient Name", "Study ID", "Series ID", "Modality", "Description", "Date", "Select"
        self.results_table.setItem(row_position, 0, QTableWidgetItem(data_dict.get("PatientName", "")))
        self.results_table.setItem(row_position, 1, QTableWidgetItem(data_dict.get("StudyID", "")))
        self.results_table.setItem(row_position, 2, QTableWidgetItem(data_dict.get("SeriesID", "")))
        self.results_table.setItem(row_position, 3, QTableWidgetItem(data_dict.get("Modality", "")))
        self.results_table.setItem(row_position, 4, QTableWidgetItem(data_dict.get("Description", ""))) # Using StudyDescription for "Description"
        self.results_table.setItem(row_position, 5, QTableWidgetItem(data_dict.get("Date", ""))) # Using StudyDate for "Date"
        
        # Placeholder for "Select" - could be a checkbox or button later
        # For now, store SeriesInstanceUID here if available, or StudyInstanceUID as fallback
        # This hidden data can be useful for retrieval
        select_text = data_dict.get("SeriesInstanceUID", data_dict.get("StudyInstanceUID", "Select"))
        item = QTableWidgetItem(select_text)
        # Make it non-editable if it's just acting as a button or data store
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.results_table.setItem(row_position, 6, item)


if __name__ == '__main__':
    # This is a simple way to test the widget if run directly
    # For a real application, this widget would be integrated into the main app
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    # This test block needs plugin_instance, so it needs adjustment if run standalone.
    # For now, assume it's not run directly or a mock plugin_instance is provided.
    class MockPlugin:
        main_window = None # Mock main_window
    
    plugin = MockPlugin() # Create a mock plugin instance
    widget = PACSWidget(plugin_instance=plugin) # Pass the mock instance
    widget.setWindowTitle("PACS Plugin Test")
    widget.show()
    sys.exit(app.exec_())
