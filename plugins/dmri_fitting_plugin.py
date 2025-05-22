from PyQt5.QtWidgets import QAction, QDockWidget, QMenu, QLabel # Added QMenu, QLabel
from PyQt5.QtCore import Qt

# Attempting to make the import path work assuming 'plugins' is a package
# and 'dmri_plugin' is a sibling or discoverable package.
# If dmri_plugin is inside plugins, it would be:
# from .dmri_plugin.gui.dmrifittingwidget import DMRIFittingWidget
# For now, assume dmri_plugin is discoverable from the root.
# This will be tested by the actual import in show_dmri_dock_widget.

class DMRICorePlugin:
    def __init__(self):
        self.main_window = None
        self.dmri_dock_widget = None
        self.dmri_fitting_widget_instance = None # Hold instance
        self.dmri_action = None # Store the action

    def initialize(self, main_window):
        self.main_window = main_window
        tools_menu = None
        
        if hasattr(main_window, 'tools_menu') and main_window.tools_menu:
            tools_menu = main_window.tools_menu
        else:
            # Fallback: Iterate through all menus in the menu bar
            for menu in main_window.menuBar().findChildren(QMenu): # Corrected QtWidgets.QMenu to QMenu
                if menu.title().replace('&','') == "Tools": # Removed ampersand for comparison
                    tools_menu = menu
                    break
        
        if not tools_menu:
            print("DMRICorePlugin: 'Tools' menu not found.")
            return

        self.dmri_action = QAction("dMRI Fitting", main_window)
        self.dmri_action.triggered.connect(self.show_dmri_dock_widget)
        tools_menu.addAction(self.dmri_action)
        print("DMRICorePlugin: Added 'dMRI Fitting' action to Tools menu.")


    def show_dmri_dock_widget(self):
        if not self.main_window: 
            print("DMRICorePlugin: Main window not set.")
            return

        if self.dmri_dock_widget is None:
            self.dmri_dock_widget = QDockWidget("dMRI Fitting", self.main_window)
            
            try:
                # This is the critical import. Assumes dmri_plugin is in PYTHONPATH
                # or 'plugins' and 'dmri_plugin' are structured such that this works.
                from dmri_plugin.gui.dmrifittingwidget import DMRIFittingWidget
                self.dmri_fitting_widget_instance = DMRIFittingWidget(parent=self.dmri_dock_widget)
                self.dmri_dock_widget.setWidget(self.dmri_fitting_widget_instance)
                print("DMRICorePlugin: DMRIFittingWidget loaded successfully.")
            except ImportError as e:
                print(f"DMRICorePlugin: Error importing DMRIFittingWidget: {e}. Using placeholder.")
                placeholder = QLabel("DMRIFittingWidget could not be loaded. Check installation/path.")
                placeholder.setAlignment(Qt.AlignCenter)
                self.dmri_dock_widget.setWidget(placeholder)
            except Exception as e_general: # Catch other potential errors during widget init
                print(f"DMRICorePlugin: Error initializing DMRIFittingWidget: {e_general}. Using placeholder.")
                placeholder = QLabel(f"Error initializing DMRIFittingWidget: {e_general}")
                placeholder.setAlignment(Qt.AlignCenter)
                self.dmri_dock_widget.setWidget(placeholder)


            self.dmri_dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
            self.dmri_dock_widget.setFloating(True) # Default to floating as per example
            
            # Add to main window and show
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.dmri_dock_widget)
            self.dmri_dock_widget.show()
            print("DMRICorePlugin: Created and showed dMRI Fitting dock widget.")

        elif self.dmri_dock_widget.isHidden():
            # If it exists but is hidden, ensure it's part of the layout and show
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.dmri_dock_widget) # Re-adding is safe
            self.dmri_dock_widget.show()
            self.dmri_dock_widget.raise_() 
            print("DMRICorePlugin: Showed existing hidden dMRI Fitting dock widget.")
        else:
            # If already visible, just bring it to the front
            self.dmri_dock_widget.show() # Ensure it's not minimized
            self.dmri_dock_widget.raise_()
            print("DMRICorePlugin: Raised existing visible dMRI Fitting dock widget.")

    def unload(self):
        """ Optional: Clean up when the plugin is unloaded. """
        if self.main_window and self.dmri_action:
            tools_menu = None
            if hasattr(self.main_window, 'tools_menu') and self.main_window.tools_menu:
                tools_menu = self.main_window.tools_menu
            else:
                for menu in self.main_window.menuBar().findChildren(QMenu):
                    if menu.title().replace('&','') == "Tools":
                        tools_menu = menu
                        break
            if tools_menu:
                tools_menu.removeAction(self.dmri_action)
                print("DMRICorePlugin: Removed 'dMRI Fitting' action from Tools menu.")
        
        if self.dmri_dock_widget:
            self.dmri_dock_widget.hide()
            self.dmri_dock_widget.setParent(None) 
            self.dmri_dock_widget.deleteLater()
            self.dmri_dock_widget = None
            self.dmri_fitting_widget_instance = None
            print("DMRICorePlugin: Dock widget hidden and scheduled for deletion.")
        
        self.main_window = None
        print("DMRICorePlugin: Unloaded.")

```
