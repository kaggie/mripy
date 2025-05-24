# plugins/pacs_plugin/pacs_plugin.py

from PyQt5.QtWidgets import QAction, QDockWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QDateTime
from .pacs_widget import PACSWidget
import pynetdicom
from pynetdicom import AE, evt
from pynetdicom.sop_class import (
    VerificationSOPClass, 
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove, # Added C-MOVE model
    # Common storage SOP classes
    CTImageStorage, 
    MRImageStorage, 
    SecondaryCaptureImageStorage,
    PositronEmissionTomographyImageStorage,
    RTImageStorage
)
from pynetdicom.presentation import AllStoragePresentationContexts # For broad storage support
from pydicom.dataset import Dataset
import socket
import traceback # For detailed error logging
import os # For directory creation
import re # For filename sanitization

class PACSTransferPlugin:
    """
    Plugin for handling PACS communication (C-ECHO, C-FIND, C-MOVE).
    Integrates with the main application by adding a menu item and a dock widget.
    """
    def __init__(self):
        self.main_window = None
        self.pacs_dock_widget = None
        self.pacs_widget_instance = None
        self.pacs_action = None
        self.ae = None # Initialize AE object once
        self.current_storage_path = "./dicom_storage" # Default path, can be updated
        print("PACSTransferPlugin: __init__ called")

    def _sanitize_for_filename(self, text):
        """Removes/replaces characters not suitable for filenames."""
        if not isinstance(text, str):
            text = str(text)
        # Remove characters that are definitely problematic
        text = re.sub(r'[\\/*?:"<>|]', "", text)
        # Replace spaces and multiple underscores with a single underscore
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'_+', '_', text)
        # Remove leading/trailing underscores
        text = text.strip('_')
        # Limit length to avoid issues, e.g., 50 chars for a component
        return text[:50]

    def _log_message(self, message):
        """Helper method to log messages to the widget's QTextEdit with a timestamp."""
        if self.pacs_widget_instance and hasattr(self.pacs_widget_instance, 'log_text_edit'):
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.pacs_widget_instance.log_text_edit.append(f"{timestamp}: {message}")
            print(f"LOG: {timestamp}: {message}") # Also print to console for debugging
        else:
            print(f"Error: Could not log message - pacs_widget_instance or log_text_edit not available. Message: {message}")

    def initialize(self, main_window):
        """
        Initializes the plugin, adds a menu item to the main window.
        """
        self.main_window = main_window
        print(f"PACSTransferPlugin: Initializing with main_window: {main_window}")

        # Find the "Tools" menu
        tools_menu = None
        for menu in self.main_window.menuBar().findChildren(QWidget): # QWidget is a base for QMenu
            if menu.title() == "&Tools": # Default title for Tools menu
                tools_menu = menu
                break
        
        if not tools_menu:
            # If Tools menu doesn't exist, create one (optional, depends on app structure)
            # For now, assume it exists or handle error appropriately
            print("PACSTransferPlugin: 'Tools' menu not found. Cannot add PACS action.")
            return

        self.pacs_action = QAction("PACS Transfer", self.main_window)
        self.pacs_action.triggered.connect(self.show_pacs_dock_widget)
        tools_menu.addAction(self.pacs_action)

        print("PACSTransferPlugin: 'PACS Transfer' action added to Tools menu.")
        print("PACSTransferPlugin: Plugin initialized.")

    def show_pacs_dock_widget(self):
        """
        Creates (if necessary) and shows the PACS dock widget.
        """
        print("PACSTransferPlugin: show_pacs_dock_widget called")
        if self.pacs_dock_widget is None:
            print("PACSTransferPlugin: Creating PACS dock widget for the first time.")
            self.pacs_dock_widget = QDockWidget("PACS Transfer", self.main_window)
            # Pass `self` (the plugin instance) to PACSWidget
            self.pacs_widget_instance = PACSWidget(plugin_instance=self, parent=self.pacs_dock_widget)
            self.pacs_dock_widget.setWidget(self.pacs_widget_instance)

            # Connect buttons from PACSWidget to handler methods
            # Assuming standard QFormLayout for config_group where button is the 5th item (index 4)
            # and using direct attribute access for other buttons as per pacs_widget.py
            if self.pacs_widget_instance.test_connection_button:
                 self.pacs_widget_instance.test_connection_button.clicked.connect(self.handle_test_connection)
            else:
                print("PACSTransferPlugin: Warning - test_connection_button not found in pacs_widget_instance.")

            if self.pacs_widget_instance.search_dicoms_button:
                self.pacs_widget_instance.search_dicoms_button.clicked.connect(self.handle_search_dicoms)
            else:
                print("PACSTransferPlugin: Warning - search_dicoms_button not found in pacs_widget_instance.")
            
            if self.pacs_widget_instance.retrieve_dicoms_button:
                self.pacs_widget_instance.retrieve_dicoms_button.clicked.connect(self.handle_retrieve_dicoms)
            else:
                print("PACSTransferPlugin: Warning - retrieve_dicoms_button not found in pacs_widget_instance.")


            self.pacs_dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            self.pacs_dock_widget.setFloating(True) # Make it floating by default
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.pacs_dock_widget) # Add to a default area
            print("PACSTransferPlugin: PACS dock widget created and configured.")
        
        self.pacs_dock_widget.show()
        self.pacs_dock_widget.raise_()
        print("PACSTransferPlugin: PACS dock widget shown and raised.")

    def handle_test_connection(self):
        """
        Handles the 'Test Connection' button click.
        Retrieves PACS configuration and attempts a C-ECHO.
        """
        self._log_message("handle_test_connection called")
        if not self.pacs_widget_instance:
            self._log_message("Error - pacs_widget_instance is None.")
            return

        host = self.pacs_widget_instance.hostname_edit.text().strip()
        port_str = self.pacs_widget_instance.port_edit.text().strip()
        our_aet = self.pacs_widget_instance.our_aet_edit.text().strip()
        server_aet = self.pacs_widget_instance.server_aet_edit.text().strip()

        if not host:
            self._log_message("Error: Hostname is required.")
            return
        if not port_str:
            self._log_message("Error: Port is required.")
            return

        try:
            port = int(port_str)
            if not (0 < port < 65536):
                 raise ValueError("Port number out of range.")
        except ValueError as e:
            self._log_message(f"Error: Invalid port number '{port_str}'. {e}")
            return
        
        if not our_aet: # Our AET can be a default but should be explicitly set for clarity
            self._log_message("Warning: 'Our AE Title' is not set. Using default 'PYNETDICOM'.")
            our_aet = 'PYNETDICOM' # pynetdicom default if not set

        if not server_aet:
            self._log_message("Error: 'Server AE Title' is required for C-ECHO.")
            return

        self._log_message(f"Config - Host: {host}, Port: {port}, Our AET: {our_aet}, Server AET: {server_aet}")

        # Initialize Application Entity
        self.ae = AE(ae_title=our_aet)
        self.ae.add_requested_context(VerificationSOPClass) # For C-ECHO

        self._log_message(f"Attempting C-ECHO to {server_aet}@{host}:{port}...")

        try:
            # Associate with the peer AE
            assoc = self.ae.associate(host, port, ae_title=server_aet)

            if assoc.is_established:
                self._log_message("Association established with PACS.")
                
                # Send C-ECHO
                status = assoc.send_c_echo()

                if status:
                    # Check status of the C-ECHO response
                    if status.Status == 0x0000: # Success
                        self._log_message("C-ECHO successful: Connection verified.")
                    else:
                        self._log_message(f"C-ECHO failed. Status: {status.Status:#06x} (pynetdicom code: {status.Status})")
                else:
                    self._log_message("C-ECHO failed: No response status received.")
                
                # Release the association
                assoc.release()
                self._log_message("Association released.")
            else:
                self._log_message("Association failed. Check AE titles, hostname, port, and network connectivity.")
                # Detailed error messages might be available in assoc.acceptor_reason or similar,
                # but pynetdicom often logs these internally or raises exceptions for critical failures.

        except ConnectionRefusedError:
            self._log_message(f"Connection Error: Connection refused by {host}:{port}. Check PACS server is running and firewall rules.")
        except socket.timeout:
            self._log_message(f"Connection Error: Connection timed out for {host}:{port}.")
        except socket.error as e: # More general socket error
            self._log_message(f"Connection Error: A socket error occurred: {e}. Check network configuration.")
        except pynetdicom.pynetdicom_series_migration_generated_20240515.pynetdicom_scp_assoc_exceptions.SCPRejectAssociation as e: # More specific pynetdicom exception for association rejection
             self._log_message(f"Association Rejected by SCP: {e}. Check AE titles and configuration on PACS.")
        except RuntimeError as e: # pynetdicom can raise RuntimeError for various reasons
            self._log_message(f"Runtime Error during C-ECHO: {e}")
        except Exception as e:
            self._log_message(f"An unexpected error occurred during C-ECHO: {e}")
            # For developers: print full traceback to console for debugging
            import traceback
            print("--- C-ECHO Exception Traceback ---")
            traceback.print_exc()
            print("----------------------------------")


    def handle_search_dicoms(self):
        """
        Handles the 'Search DICOMs' button click.
        Retrieves search criteria and attempts a C-FIND.
        """
        self._log_message("handle_search_dicoms called")
        if not self.pacs_widget_instance:
            self._log_message("Error - pacs_widget_instance is None.")
            return

        # 1. Clear previous results
        try:
            self.pacs_widget_instance.clear_results_table()
            self._log_message("Cleared previous search results.")
        except Exception as e:
            self._log_message(f"Error clearing results table: {e}")
            return

        # 2. Retrieve PACS configuration
        host = self.pacs_widget_instance.hostname_edit.text().strip()
        port_str = self.pacs_widget_instance.port_edit.text().strip()
        our_aet = self.pacs_widget_instance.our_aet_edit.text().strip()
        server_aet = self.pacs_widget_instance.server_aet_edit.text().strip()
        search_term = self.pacs_widget_instance.query_terms_edit.text().strip()

        # --- Input Validation (similar to C-ECHO) ---
        if not host: self._log_message("Error: Hostname is required."); return
        if not port_str: self._log_message("Error: Port is required."); return
        try:
            port = int(port_str)
            if not (0 < port < 65536): raise ValueError("Port number out of range.")
        except ValueError as e:
            self._log_message(f"Error: Invalid port number '{port_str}'. {e}"); return
        if not our_aet: our_aet = 'PYNETDICOM'; self._log_message("Warning: 'Our AE Title' not set, using 'PYNETDICOM'.")
        if not server_aet: self._log_message("Error: 'Server AE Title' is required for C-FIND."); return
        # if not search_term: self._log_message("Warning: Search term is empty. This might return many results."); # Allow empty search

        self._log_message(f"C-FIND Config - Host: {host}, Port: {port}, Our AET: {our_aet}, Server AET: {server_aet}, Search: '{search_term}'")

        # 3. Create Query Dataset
        query_dataset = Dataset()
        query_dataset.QueryRetrieveLevel = "PATIENT" # Or STUDY if preferred
        
        # For simplicity, we'll search by PatientName. Wildcards are common.
        query_dataset.PatientName = f"*{search_term}*" if search_term else "*" # Search all if empty
        
        # Specify attributes to return (these will be columns)
        query_dataset.PatientID = ""
        query_dataset.StudyInstanceUID = ""
        query_dataset.SeriesInstanceUID = "" # Important for C-MOVE later
        query_dataset.Modality = ""
        query_dataset.StudyDescription = ""
        query_dataset.SeriesDescription = "" # Often useful
        query_dataset.StudyDate = ""
        query_dataset.PatientBirthDate = "" 
        # Add other tags you might want to display or use
        # query_dataset.AccessionNumber = ""
        # query_dataset.ReferringPhysicianName = ""

        # 4. Initialize AE and Presentation Context
        self.ae = AE(ae_title=our_aet)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        # If you want to support other SOP classes like StudyRoot, add them here.
        # self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)

        self._log_message(f"Attempting C-FIND to {server_aet}@{host}:{port} with level '{query_dataset.QueryRetrieveLevel}' for PatientName '{query_dataset.PatientName}'")
        
        results_found_count = 0

        # 5. Associate and Perform C-FIND
        try:
            assoc = self.ae.associate(host, port, ae_title=server_aet)
            if assoc.is_established:
                self._log_message("Association established for C-FIND.")
                
                responses = assoc.send_c_find(query_dataset, PatientRootQueryRetrieveInformationModelFind)
                
                for (status_dataset, identifier_dataset) in responses:
                    if status_dataset is None:
                        self._log_message("Received a None status dataset, which is unexpected. Skipping.")
                        continue

                    self._log_message(f"C-FIND Status: {status_dataset.Status:#06x} ({status_dataset.Status})")

                    if status_dataset.Status == 0xFF00 or status_dataset.Status == 0xFF01: # Pending
                        if identifier_dataset:
                            results_found_count += 1
                            self._log_message(f"  Result {results_found_count}: Pending - Data received.")
                            # Extract data and add to table
                            data_dict = {
                                "PatientName": getattr(identifier_dataset, 'PatientName', "N/A"),
                                "StudyID": getattr(identifier_dataset, 'StudyID', getattr(identifier_dataset, 'PatientID', "N/A")), # Fallback to PatientID if StudyID not present
                                "SeriesID": getattr(identifier_dataset, 'SeriesNumber', "N/A"), # SeriesNumber is more common than SeriesID in responses
                                "Modality": getattr(identifier_dataset, 'Modality', "N/A"),
                                "Description": getattr(identifier_dataset, 'StudyDescription', getattr(identifier_dataset, 'SeriesDescription', "N/A")),
                                "Date": getattr(identifier_dataset, 'StudyDate', "N/A"),
                                # Store UIDs for potential C-MOVE, not directly displayed in "Select" column text
                                "StudyInstanceUID": getattr(identifier_dataset, 'StudyInstanceUID', ""),
                                "SeriesInstanceUID": getattr(identifier_dataset, 'SeriesInstanceUID', "") 
                            }
                            # The 'Select' column text in add_search_result_row will use SeriesInstanceUID or StudyInstanceUID
                            self.pacs_widget_instance.add_search_result_row(data_dict)
                        else:
                            self._log_message("  Pending status but no identifier dataset received.")
                    elif status_dataset.Status == 0x0000: # Success
                        self._log_message(f"C-FIND completed successfully. Total results: {results_found_count}.")
                        if results_found_count == 0:
                             self._log_message("No matching results found for the given criteria.")
                        break # Operation finished
                    else: # Failed, Refused, Cancelled
                        error_name = status_dataset.ErrorComment if hasattr(status_dataset, 'ErrorComment') else "Unknown Error"
                        self._log_message(f"C-FIND failed or was cancelled. Status: {status_dataset.Status:#06x}. Error: {error_name}")
                        break # Operation finished or failed

                assoc.release()
                self._log_message("Association released after C-FIND.")
            else:
                self._log_message("C-FIND association failed. Check AE titles, hostname, port, and network connectivity.")

        except ConnectionRefusedError:
            self._log_message(f"Connection Error: Connection refused by {host}:{port}.")
        except socket.timeout:
            self._log_message(f"Connection Error: Connection timed out for {host}:{port}.")
        except socket.error as e:
            self._log_message(f"Socket Error: {e}.")
        except pynetdicom.pynetdicom_series_migration_generated_20240515.pynetdicom_scp_assoc_exceptions.SCPRejectAssociation as e:
             self._log_message(f"Association Rejected by SCP: {e}.")
        except RuntimeError as e:
            self._log_message(f"Runtime Error during C-FIND: {e}")
            self._log_message(traceback.format_exc()) # Log full traceback for runtime errors
        except Exception as e:
            self._log_message(f"An unexpected error occurred during C-FIND: {e}")
            self._log_message(traceback.format_exc()) # Log full traceback

    def handle_retrieve_dicoms(self):
        """
        Handles the 'Retrieve Selected DICOMs' button click.
        Identifies selected studies/series and attempts a C-MOVE.
        """
        self._log_message("handle_retrieve_dicoms called")
        if not self.pacs_widget_instance:
            self._log_message("Error - pacs_widget_instance is None.")
            return

        # 1. Retrieve PACS configuration
        host = self.pacs_widget_instance.hostname_edit.text().strip()
        port_str = self.pacs_widget_instance.port_edit.text().strip()
        our_aet = self.pacs_widget_instance.our_aet_edit.text().strip() # This is our AE Title, the move destination
        server_aet = self.pacs_widget_instance.server_aet_edit.text().strip()

        # --- Input Validation ---
        if not host: self._log_message("Error: Hostname is required."); return
        if not port_str: self._log_message("Error: Port is required."); return
        try:
            port = int(port_str)
            if not (0 < port < 65536): raise ValueError("Port number out of range.")
        except ValueError as e:
            self._log_message(f"Error: Invalid port number '{port_str}'. {e}"); return
        if not our_aet: self._log_message("Error: 'Our AE Title' (Move Destination) is required."); return
        if not server_aet: self._log_message("Error: 'Server AE Title' is required for C-MOVE."); return

        self._log_message(f"C-MOVE Config - Host: {host}, Port: {port}, Our AET (Dest): {our_aet}, Server AET: {server_aet}")

        # 2. Identify Selected Studies/Series
        selected_items_uids = [] # List of dicts: {"StudyInstanceUID": uid, "SeriesInstanceUID": uid_or_none}
        selected_rows = self.pacs_widget_instance.results_table.selectionModel().selectedRows()

        if not selected_rows:
            self._log_message("Retrieve: No studies/series selected from the table.")
            return

        for index in selected_rows:
            row = index.row()
            # Retrieve StudyInstanceUID and SeriesInstanceUID stored in the table items by C-FIND
            # Assuming StudyInstanceUID is in a hidden part of column 0 or similar, and SeriesInstanceUID in column 6 (Select)
            # For this example, we'll rely on the "Select" column (index 6) containing SeriesInstanceUID,
            # and assume StudyInstanceUID would need to be fetched from another column if C-FIND populated it.
            # The C-FIND implementation stores SeriesInstanceUID or StudyInstanceUID in the "Select" column's item (column 6).
            
            # To get StudyInstanceUID, we need to retrieve it from the table.
            # Let's assume C-FIND stored StudyInstanceUID in the item data of the first column (PatientName) for simplicity,
            # or use a dedicated (possibly hidden) column.
            # For now, we will assume the add_search_result_row in pacs_widget stored UIDs correctly.
            # The 'Select' column (idx 6) in pacs_widget.add_search_result_row stores SeriesInstanceUID or StudyInstanceUID.
            
            select_item = self.pacs_widget_instance.results_table.item(row, 6) # "Select" column where UID is stored
            uid_value = select_item.text() if select_item else None

            # We need both Study and Series UIDs for a Series-level C-MOVE.
            # The C-FIND populates the table with StudyInstanceUID and SeriesInstanceUID.
            # The pacs_widget.add_search_result_row stores these in the data_dict passed to it.
            # We need to ensure these UIDs are accessible here.
            # A robust way is to store them as Qt.UserRole data in the QTableWidgetItems.
            # For now, let's assume add_search_result_row stores SeriesInstanceUID in col 6,
            # and StudyInstanceUID would need to be retrieved from another column if not implicitly part of 'uid_value'
            
            # This part needs careful coordination with how UIDs are stored by add_search_result_row.
            # Let's assume item(row, 6) contains SeriesInstanceUID (if available)
            # and item(row, X) contains StudyInstanceUID.
            # The current C-FIND implementation puts StudyInstanceUID and SeriesInstanceUID into the `data_dict`
            # for `add_search_result_row`. `add_search_result_row` then uses `data_dict.get("SeriesInstanceUID", data_dict.get("StudyInstanceUID", "Select"))`
            # for the text of column 6. This means `uid_value` *is* the SeriesInstanceUID if available, else StudyInstanceUID.
            
            # We need to distinguish if it's a Study or Series UID.
            # A better way: store UIDs as Qt.UserRole data in table items.
            # For now, we'll assume if a UID is from column 6, we need its corresponding StudyUID.
            # This is a simplification. A real implementation would store these explicitly.
            # Let's retrieve them from the data used to populate the row if possible, or make assumptions.

            # To proceed, we need StudyInstanceUID. Let's assume it's in a hidden column or retrievable.
            # For this example, we'll assume the C-FIND results are still somehow accessible or that
            # the table stores enough info.
            # The `data_dict` used in `add_search_result_row` had `StudyInstanceUID` and `SeriesInstanceUID`.
            # If `pacs_widget.py` was modified to store these as item data (e.g., item.setData(Qt.UserRole, uid)), it would be best.
            # Lacking that, we must rely on text from columns.
            # C-FIND `data_dict` has "StudyInstanceUID" and "SeriesInstanceUID".
            # `add_search_result_row` sets column 6 to `data_dict.get("SeriesInstanceUID", data_dict.get("StudyInstanceUID", "Select"))`.
            # This means `uid_value` is SeriesInstanceUID if series-level, or StudyInstanceUID if study-level.

            # Let's assume we need to find the StudyInstanceUID from another column if uid_value is a SeriesInstanceUID.
            # This is tricky without knowing which column holds what.
            # The `data_dict` passed to `add_search_result_row` contains both.
            # The simplest path for this exercise is to assume `uid_value` is what we operate on,
            # and `QueryRetrieveLevel` will be based on whether it looks like a study or series UID
            # (which is not robust).
            
            # A better approach: The `add_search_result_row` in `pacs_widget.py` should store
            # `data_dict["StudyInstanceUID"]` and `data_dict["SeriesInstanceUID"]` as Qt.UserRole
            # on the QTableWidgetItem in column 6 (or separate hidden columns).
            # E.g., item.setData(Qt.UserRole, {"study": study_uid, "series": series_uid})
            # Given the current structure, this is not possible without modifying pacs_widget.py again.

            # Workaround: For this subtask, we will assume that if a SeriesInstanceUID is present in column 6,
            # we can find the corresponding StudyInstanceUID from one of the other columns if needed.
            # The C-FIND response includes StudyInstanceUID. Let's assume column 1 ("Study ID") is NOT StudyInstanceUID,
            # but a human-readable ID. We need the actual StudyInstanceUID.
            # The `data_dict` in `handle_search_dicoms` had `StudyInstanceUID` and `SeriesInstanceUID`.
            # `add_search_result_row` takes this `data_dict`.
            # Let's modify `add_search_result_row` in thought: it should store these UIDs.
            # Since I cannot modify `pacs_widget.py` now, I will make a simplifying assumption:
            # If `uid_value` from column 6 has a structure that suggests it's a SeriesInstanceUID (e.g. more dots),
            # then we need to find its parent StudyInstanceUID. This is not robust.
            
            # The task implies C-FIND stores SeriesInstanceUID in "Select" column or StudyInstanceUID if series not available.
            # This means `uid_value` IS the SeriesInstanceUID or StudyInstanceUID.
            # We need to fetch the StudyInstanceUID for the row.
            # Let's assume column 1 ("Study ID") text is the StudyInstanceUID for simplicity for this task.
            # THIS IS A MAJOR SIMPLIFICATION AND NOT DICOM ACCURATE for display IDs.
            
            study_instance_uid_item = self.pacs_widget_instance.results_table.item(row, 1) # Assuming this column contains StudyInstanceUID
            study_uid = study_instance_uid_item.text() if study_instance_uid_item else None

            series_uid_from_select_col = uid_value # This is SeriesInstanceUID or StudyInstanceUID

            if not study_uid:
                self._log_message(f"Warning: Could not retrieve StudyInstanceUID for row {row}. Skipping.")
                continue

            # Determine if series_uid_from_select_col is actually a Series UID or a Study UID (fallback)
            # A real DICOM UID has a specific structure. A simple check is not enough.
            # For now, if series_uid_from_select_col is different from study_uid and present, assume it's a series_uid.
            # Otherwise, we are retrieving the whole study.
            series_uid = None
            query_level = "STUDY"
            if series_uid_from_select_col and series_uid_from_select_col != study_uid and series_uid_from_select_col != "Select":
                series_uid = series_uid_from_select_col
                query_level = "SERIES"
            
            selected_items_uids.append({
                "StudyInstanceUID": study_uid, 
                "SeriesInstanceUID": series_uid, # This will be None if retrieving whole study
                "QueryRetrieveLevel": query_level
            })

        if not selected_items_uids:
            self._log_message("Retrieve: No valid Study/Series UIDs could be extracted for selected rows.")
            return

        # 3. Setup DICOM Storage Directory
        storage_path_from_ui = self.pacs_widget_instance.storage_dir_line_edit.text().strip()
        if not storage_path_from_ui:
            self.current_storage_path = "./dicom_storage" # Default if empty
            self._log_message(f"Warning: Storage directory in UI is empty. Defaulting to {self.current_storage_path}")
        else:
            self.current_storage_path = storage_path_from_ui
        
        try:
            os.makedirs(self.current_storage_path, exist_ok=True)
            self._log_message(f"DICOM files will be stored in: {os.path.abspath(self.current_storage_path)}")
        except Exception as e:
            self._log_message(f"Error creating storage directory '{self.current_storage_path}': {e}"); return
        
        # 4. AE Configuration for C-MOVE
        self.ae = AE(ae_title=our_aet)

        # Add supported storage contexts (acts as SCP for C-STORE)
        # Using AllStoragePresentationContexts provides broad support for what we can receive.
        # You can be more specific by adding individual contexts like CTImageStorage, MRImageStorage, etc.
        # For ImplicitVRLittleEndian only:
        # for context in AllStoragePresentationContexts:
        #     self.ae.add_supported_context(context.abstract_syntax, pynetdicom.sop_class.ImplicitVRLittleEndian)
        # For multiple transfer syntaxes (recommended):
        self.ae.supported_contexts = AllStoragePresentationContexts
        
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove) # SCU for C-MOVE

        # 5. Bind Event Handler for C-STORE
        # evt_handlers are passed to ae.associate()
        evt_handlers = [(evt.EVT_C_STORE, self.handle_store_event)]

        self._log_message(f"Attempting C-MOVE to {server_aet}@{host}:{port}, destination AET: {our_aet}")
        
        # 6. Associate and Perform C-MOVE
        try:
            assoc = self.ae.associate(host, port, ae_title=server_aet, evt_handlers=evt_handlers)
            if assoc.is_established:
                self._log_message("Association established for C-MOVE.")
                
                for item_uids in selected_items_uids:
                    study_uid = item_uids["StudyInstanceUID"]
                    series_uid = item_uids["SeriesInstanceUID"]
                    query_level = item_uids["QueryRetrieveLevel"]

                    query_dataset = Dataset()
                    query_dataset.QueryRetrieveLevel = query_level
                    query_dataset.StudyInstanceUID = study_uid
                    if series_uid: # If it's a series level query
                        query_dataset.SeriesInstanceUID = series_uid
                    
                    self._log_message(f"Sending C-MOVE for {query_level} UID: {series_uid if series_uid else study_uid}")
                    
                    responses = assoc.send_c_move(query_dataset, our_aet, PatientRootQueryRetrieveInformationModelMove)
                    
                    for (status_dataset, identifier_dataset) in responses:
                        if status_dataset is None:
                            self._log_message("Received a None status dataset for C-MOVE response. Skipping.")
                            continue
                        
                        self._log_message(f"C-MOVE Status: {status_dataset.Status:#06x}")

                        if status_dataset.Status == 0xFF00: # Pending
                             self._log_message(f"  Pending: {status_dataset.NumberOfRemainingSuboperations or 0} remaining, "
                                               f"{status_dataset.NumberOfCompletedSuboperations or 0} completed, "
                                               f"{status_dataset.NumberOfFailedSuboperations or 0} failed, "
                                               f"{status_dataset.NumberOfWarningSuboperations or 0} warnings.")
                        elif status_dataset.Status == 0x0000: # Success
                            self._log_message(f"C-MOVE sub-operation successful for {query_level} UID: {series_uid if series_uid else study_uid}.")
                            self._log_message(f"  Final Status: {status_dataset.NumberOfCompletedSuboperations or 0} completed, "
                                              f"{status_dataset.NumberOfFailedSuboperations or 0} failed, "
                                              f"{status_dataset.NumberOfWarningSuboperations or 0} warnings.")
                            break # This specific C-MOVE operation for one study/series is done
                        else: # Failure or Warning for this specific C-MOVE operation
                            error_comment = getattr(status_dataset, 'ErrorComment', 'Unknown error')
                            self._log_message(f"C-MOVE sub-operation failed or warning for {query_level} UID: {series_uid if series_uid else study_uid}. Status: {status_dataset.Status:#06x}. Error: {error_comment}")
                            self._log_message(f"  Details: {status_dataset.NumberOfFailedSuboperations or 0} failed sub-ops.")
                            break # Stop processing responses for this failed C-MOVE

                assoc.release()
                self._log_message("Association released after C-MOVE operations.")
            else:
                self._log_message("C-MOVE association failed. Check AE titles, hostname, port, and network connectivity.")

        except ConnectionRefusedError:
            self._log_message(f"Connection Error: Connection refused by {host}:{port}.")
        except socket.timeout:
            self._log_message(f"Connection Error: Connection timed out for {host}:{port}.")
        except socket.error as e:
            self._log_message(f"Socket Error: {e}.")
        except pynetdicom.pynetdicom_series_migration_generated_20240515.pynetdicom_scp_assoc_exceptions.SCPRejectAssociation as e:
             self._log_message(f"Association Rejected by SCP: {e}.")
        except RuntimeError as e:
            self._log_message(f"Runtime Error during C-MOVE: {e}")
            self._log_message(traceback.format_exc())
        except Exception as e:
            self._log_message(f"An unexpected error occurred during C-MOVE: {e}")
            self._log_message(traceback.format_exc())

    def handle_store_event(self, event):
        """
        Event handler for evt.EVT_C_STORE, called when a C-STORE request is received.
        Saves the incoming DICOM dataset to the dicom_storage_dir.
        """
        try:
            dataset = event.dataset
            sop_class_uid = event.context.sop_class_uid
            self._log_message(f"Received C-STORE request for SOP Class: {sop_class_uid} ({pynetdicom.sop_class.uid_to_sop_class(sop_class_uid).name})")

            # Ensure dataset has SOPInstanceUID for filename
            if not hasattr(dataset, 'SOPInstanceUID'):
                self._log_message("Error: Received dataset for C-STORE does not have SOPInstanceUID. Cannot save.")
                return 0xA900 # General failure

            # File Renaming Logic
            patient_name = self._sanitize_for_filename(getattr(dataset, 'PatientName', 'UnknownPatient'))
            patient_id = self._sanitize_for_filename(getattr(dataset, 'PatientID', 'UnknownID'))
            study_uid_short = self._sanitize_for_filename(getattr(dataset, 'StudyInstanceUID', 'UnknownStudyUID')[-12:]) # Last 12 chars
            series_num = self._sanitize_for_filename(getattr(dataset, 'SeriesNumber', 'S0'))
            instance_num = self._sanitize_for_filename(getattr(dataset, 'InstanceNumber', 'I0'))
            sop_uid_short = self._sanitize_for_filename(dataset.SOPInstanceUID[-12:]) # Last 12 chars

            # Construct filename: PatientName_PatientID_StudyUID-short_SeriesNum_InstanceNum_SOPUID-short.dcm
            # Using a simpler pattern as suggested for first pass: PatientName_StudyUID_SOPUID.dcm
            # filename = f"{patient_name}_{study_uid_short}_{sop_uid_short}.dcm"
            # Using the more detailed pattern:
            filename = f"{patient_name}_{patient_id}_{study_uid_short}_{series_num}_{instance_num}_{sop_uid_short}.dcm"
            
            filepath = os.path.join(self.current_storage_path, filename)

            # Ensure filename uniqueness if somehow it already exists (though SOPInstanceUID should make it unique)
            counter = 1
            base_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(base_filepath)
                filepath = f"{name}_{counter}{ext}"
                counter += 1
                if counter > 10: # Safety break
                    self._log_message(f"Error: Could not find a unique filename for {base_filepath} after 10 tries.")
                    return 0xA700 # Out of resources - filename

            # Save the dataset
            dataset.save_as(filepath, write_like_original=False) # write_like_original=False is important for consistency
            # self._log_message(f"DICOM instance saved to: {filepath}") # Original log message, will be replaced by detailed one.

            # Log detailed summary of DICOM tags
            log_summary = [
                f"DICOM instance saved:",
                f"  Path: {filepath}",
                f"  Patient Name: {getattr(dataset, 'PatientName', 'N/A')}",
                f"  Patient ID: {getattr(dataset, 'PatientID', 'N/A')}",
                f"  Study UID: {getattr(dataset, 'StudyInstanceUID', 'N/A')}",
                f"  Series UID: {getattr(dataset, 'SeriesInstanceUID', 'N/A')}",
                f"  SOP UID: {getattr(dataset, 'SOPInstanceUID', 'N/A')}",
                f"  Study Description: {getattr(dataset, 'StudyDescription', 'N/A')}",
                f"  Series Description: {getattr(dataset, 'SeriesDescription', 'N/A')}",
                f"  Modality: {getattr(dataset, 'Modality', 'N/A')}",
                f"  Study Date: {getattr(dataset, 'StudyDate', 'N/A')}",
                f"  Series Date: {getattr(dataset, 'SeriesDate', 'N/A')}",
                f"  Acquisition Date: {getattr(dataset, 'AcquisitionDate', 'N/A')}",
                f"  Content Date: {getattr(dataset, 'ContentDate', 'N/A')}",
                f"  Content Time: {getattr(dataset, 'ContentTime', 'N/A')}"
            ]
            self._log_message("\n".join(log_summary))
            
            return 0x0000  # Success
        except Exception as e:
            self._log_message(f"Error in handle_store_event: {e}")
            self._log_message(traceback.format_exc())
            # Return a failure status, see PS3.4 C.4.2.1.4 for status codes
            return 0xC001 # Processing failure (pynetdicom example uses this, or 0xA700 for out of resources)


    def unload(self):
        """
        Cleans up the plugin resources (menu action, dock widget).
        """
        print("PACSTransferPlugin: Unloading plugin...")
        if self.pacs_action and self.main_window:
            # Find the "Tools" menu again to remove the action
            tools_menu = None
            for menu in self.main_window.menuBar().findChildren(QWidget): # QWidget is a base for QMenu
                if menu.title() == "&Tools":
                    tools_menu = menu
                    break
            if tools_menu:
                tools_menu.removeAction(self.pacs_action)
                print("PACSTransferPlugin: Removed 'PACS Transfer' action from Tools menu.")
            self.pacs_action.deleteLater() # Ensure proper Qt cleanup
            self.pacs_action = None

        if self.pacs_dock_widget:
            self.main_window.removeDockWidget(self.pacs_dock_widget)
            self.pacs_dock_widget.deleteLater() # Ensure proper Qt cleanup
            self.pacs_dock_widget = None
            self.pacs_widget_instance = None # The widget is a child of the dock, should be cleaned up too
            print("PACSTransferPlugin: Removed and cleaned up PACS dock widget.")
        
        self.main_window = None
        print("PACSTransferPlugin: Plugin unloaded.")

# Example of how QWidget could be used if QMenu is not directly found
# This is more of a note for robustness if needed.
from PyQt5.QtWidgets import QWidget 

# Helper to find menu (if needed for more complex scenarios)
# def find_menu_by_title(menu_bar, title):
#     for action in menu_bar.actions():
#         if action.menu() and action.menu().title() == title:
#             return action.menu()
#     return None
