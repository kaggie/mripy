from PyQt5.QtWidgets import QAction, QDockWidget, QVBoxLayout, QWidget, QMenu # QLabel removed
from PyQt5.QtCore import Qt
# Import the new custom widget
from mrsi_plugin.gui.mrsifittingwidget import MRSIFittingWidget

class MRSIFittingPlugin:
    def __init__(self):
        self.main_window = None
        self.mrsi_dock_widget = None
        self.mrsi_plugin_action = None # Store the action

    def initialize(self, main_window):
        self.main_window = main_window

        tools_menu = None
        # Prioritize direct attribute access if MainWindow is structured that way
        if hasattr(main_window, 'tools_menu') and main_window.tools_menu:
            tools_menu = main_window.tools_menu
        else:
            # Fallback: Iterate through all menus in the menu bar
            # This is a more robust way to find a menu by its title
            for menu in main_window.menuBar().findChildren(QMenu):
                # Strip ampersands used for mnemonics (&File -> File) for comparison
                menu_title_stripped = menu.title().replace('&', '')
                if menu_title_stripped == "Tools":
                    tools_menu = menu
                    break
        
        if not tools_menu:
            print("MRSIFittingPlugin: 'Tools' menu not found in main_window.")
            return

        self.mrsi_plugin_action = QAction("MRSI Fitting", main_window)
        self.mrsi_plugin_action.triggered.connect(self.show_mrsi_dock_widget)
        tools_menu.addAction(self.mrsi_plugin_action)
        print("MRSIFittingPlugin: Added 'MRSI Fitting' action to Tools menu.")

    def show_mrsi_dock_widget(self):
        if not self.main_window:
            print("MRSIFittingPlugin: Main window not available.")
            return

        if self.mrsi_dock_widget is None:
            self.mrsi_dock_widget = QDockWidget("MRSI Fitting", self.main_window)
            
            # Instantiate and set the MRSIFittingWidget as the content
            self.fitting_widget_instance = MRSIFittingWidget() 
            self.mrsi_dock_widget.setWidget(self.fitting_widget_instance)

            self.mrsi_dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
            # self.mrsi_dock_widget.setFloating(True) # Optional: Default to docked
            
            
            # Important: Add the dock widget to the main window before showing if it's the first time
            # Or if it was closed (Qt might remove it from main window's list of dock widgets)
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.mrsi_dock_widget)
            self.mrsi_dock_widget.show()
            print("MRSIFittingPlugin: Created and showed MRSI Fitting dock widget.")

        elif self.mrsi_dock_widget.isHidden():
            # If the dock widget exists but is hidden, ensure it's added and then show
            # Check if it's still part of the main window's dock widgets
            # This check might be overly complex; usually, if not destroyed, show() is enough.
            # However, addDockWidget can be called again if it was removed.
            # For simplicity, let's assume if it's hidden, it might need re-adding or just showing.
            # A robust way is to just call addDockWidget again if it's not part of the window.
            # However, Qt usually just hides it.
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.mrsi_dock_widget) # Re-adding is safe
            self.mrsi_dock_widget.show()
            self.mrsi_dock_widget.raise_() # Bring to front
            print("MRSIFittingPlugin: Showed existing hidden MRSI Fitting dock widget.")
        else:
            # If already visible, just bring it to the front
            self.mrsi_dock_widget.raise_()
            self.mrsi_dock_widget.show() # Ensure it's not minimized or obscured
            print("MRSIFittingPlugin: Raised existing visible MRSI Fitting dock widget.")

    def unload(self):
        """
        Optional: Clean up when the plugin is unloaded.
        - Remove the action from the menu.
        - Hide and delete the dock widget.
        """
        if self.main_window and self.mrsi_plugin_action:
            # Find tools_menu again or assume it's still valid
            tools_menu = None
            if hasattr(self.main_window, 'tools_menu') and self.main_window.tools_menu:
                tools_menu = self.main_window.tools_menu
            else:
                for menu in self.main_window.menuBar().findChildren(QMenu):
                    if menu.title().replace('&','') == "Tools":
                        tools_menu = menu
                        break
            if tools_menu:
                tools_menu.removeAction(self.mrsi_plugin_action)
                print("MRSIFittingPlugin: Removed 'MRSI Fitting' action from Tools menu.")
        
        if self.mrsi_dock_widget:
            self.mrsi_dock_widget.hide()
            self.mrsi_dock_widget.setParent(None) # Allow garbage collection
            self.mrsi_dock_widget.deleteLater() # Schedule for deletion
            self.mrsi_dock_widget = None
            print("MRSIFittingPlugin: Dock widget hidden and scheduled for deletion.")
        
        self.main_window = None # Clear reference
        print("MRSIFittingPlugin: Unloaded.")

# Example of how this plugin might be managed if the application supports dynamic loading/unloading
# This is not part of the plugin itself but illustrates usage pattern.
# class PluginManager:
#     def __init__(self, main_window):
#         self.main_window = main_window
#         self.loaded_plugins = {}

#     def load_plugin(self, plugin_name, plugin_class):
#         if plugin_name not in self.loaded_plugins:
#             plugin_instance = plugin_class()
#             plugin_instance.initialize(self.main_window)
#             self.loaded_plugins[plugin_name] = plugin_instance
#             print(f"Loaded plugin: {plugin_name}")
#         else:
#             print(f"Plugin {plugin_name} already loaded.")

#     def unload_plugin(self, plugin_name):
#         if plugin_name in self.loaded_plugins:
#             plugin_instance = self.loaded_plugins.pop(plugin_name)
#             if hasattr(plugin_instance, 'unload'):
#                 plugin_instance.unload()
#             print(f"Unloaded plugin: {plugin_name}")
#         else:
#             print(f"Plugin {plugin_name} not found.")

```
