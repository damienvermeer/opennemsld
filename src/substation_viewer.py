# Standard library imports
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from threading import Thread
import yaml
import io
import math
import re


# Local application imports
import sld
from sld import Substation, DrawingParams, render_substation_svg

# Configuration
YAML_DIR = r"."  # Default directory, change this to your YAML file location
REFRESH_INTERVAL = 500  # milliseconds (0.5 seconds)


class SubstationViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Substation Viewer")
        self.geometry("800x600")
        self.minsize(600, 400)

        # Variables
        self.substation_name = tk.StringVar()
        self.yaml_path = tk.StringVar(value=YAML_DIR)
        self.status_text = tk.StringVar(value="Ready")
        self.last_update_time = tk.StringVar(value="Not updated yet")
        self.auto_refresh = tk.BooleanVar(value=True)
        self.substations_map = {}
        self.connection_status = {}

        # Setup UI
        self._create_ui()

        # Start refresh timer
        self._schedule_refresh()

    def _create_ui(self):
        """Create the user interface elements"""
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # YAML path
        ttk.Label(controls_frame, text="YAML Path:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(controls_frame, textvariable=self.yaml_path, width=40).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        # Substation name
        ttk.Label(controls_frame, text="Substation Name:").pack(
            side=tk.LEFT, padx=(0, 5)
        )
        name_entry = ttk.Entry(
            controls_frame, textvariable=self.substation_name, width=30
        )
        name_entry.pack(side=tk.LEFT, padx=(0, 10))
        name_entry.bind("<Return>", lambda e: self._refresh_diagram())

        # Auto-refresh toggle
        ttk.Checkbutton(
            controls_frame, text="Auto Refresh", variable=self.auto_refresh
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Refresh button
        ttk.Button(
            controls_frame, text="Refresh Now", command=self._refresh_diagram
        ).pack(side=tk.LEFT)

        # Content area - split into diagram and log
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Right side - log and connection status
        info_frame = ttk.Frame(content_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Connection status
        conn_frame = ttk.LabelFrame(info_frame, text="Connection Status")
        conn_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.conn_text = scrolledtext.ScrolledText(conn_frame, height=10, wrap=tk.WORD)
        self.conn_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log
        log_frame = ttk.LabelFrame(info_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT)
        ttk.Label(status_frame, text="Last Update:").pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Label(status_frame, textvariable=self.last_update_time).pack(side=tk.RIGHT)

    def _schedule_refresh(self):
        """Schedule the next refresh if auto-refresh is enabled"""
        if self.auto_refresh.get() and self.substation_name.get().strip():
            self._refresh_diagram()
        self.after(REFRESH_INTERVAL, self._schedule_refresh)

    def _refresh_diagram(self):
        """Refresh the substation diagram"""
        name = self.substation_name.get().strip()
        if not name:
            self.status_text.set("Please enter a substation name")
            return

        yaml_path = self.yaml_path.get().strip()
        if not os.path.exists(yaml_path):
            self.log_message(f"Warning: YAML path does not exist: {yaml_path}")
            self.status_text.set("Invalid YAML path")
            return

        # Run the refresh in a separate thread to avoid UI freezing
        Thread(
            target=self._load_and_render, args=(name, yaml_path), daemon=True
        ).start()

    def _load_and_render(self, name, yaml_path):
        """Load YAML data and render the substation diagram"""
        try:
            self.status_text.set("Loading YAML data...")
            self.update_idletasks()

            # Load YAML data
            if os.path.isdir(yaml_path):
                # If it's a directory, load all YAML files
                self.substations_map = {}
                for filename in os.listdir(yaml_path):
                    if filename.endswith((".yaml", ".yml")):
                        file_path = os.path.join(yaml_path, filename)
                        self.substations_map.update(
                            sld.load_substations_from_yaml(file_path)
                        )
            else:
                # If it's a file, load just that file
                self.substations_map = sld.load_substations_from_yaml(yaml_path)

            # Check if the substation exists
            if name not in self.substations_map:
                self.log_message(f"Substation '{name}' not found in YAML data")
                self.status_text.set(f"Substation '{name}' not found")
                return

            # Get the substation
            substation = self.substations_map[name]

            # Render the substation
            self.status_text.set("Rendering substation diagram...")
            self.update_idletasks()

            # Use the new function to render SVG with connection status
            temp_svg_file = "temp_substation.svg"
            (
                svg_content,
                connection_status,
                full_file_path,
            ) = sld.render_substation_with_connection_status(
                substation, all_substations=self.substations_map, filename=temp_svg_file
            )

            # Update the connection status display
            self.after(0, self._update_connection_status_display, connection_status)

            # SVG is generated, log path
            self.log_message(f"SVG generated at: {full_file_path}")
            self.status_text.set("SVG generated successfully")
            current_time = time.strftime("%H:%M:%S")
            self.last_update_time.set(current_time)

        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.status_text.set(f"Error: {str(e)}")

    def _update_connection_status_display(self, connection_status):
        """Update the connection status based on data from the rendering function"""
        self.connection_status = connection_status
        self.after(0, self._update_connection_status_text)

    def _update_connection_status_text(self):
        """Update the connection status text widget"""
        self.conn_text.delete(1.0, tk.END)

        if not self.connection_status:
            self.conn_text.insert(tk.END, "No connections defined")
            return

        for conn_name, (found, sub_name) in sorted(self.connection_status.items()):
            status = "✓ Found" if found else "✗ Not Found"
            color = "green" if found else "red"

            self.conn_text.insert(tk.END, f"{conn_name}: ", "conn")
            self.conn_text.insert(tk.END, status, color)

            if found and sub_name:
                self.conn_text.insert(tk.END, f" (in {sub_name})")

            self.conn_text.insert(tk.END, "\n")

        # Configure tags for colors
        self.conn_text.tag_configure("green", foreground="green")
        self.conn_text.tag_configure("red", foreground="red")
        self.conn_text.tag_configure("conn", font=("TkDefaultFont", 10, "bold"))

    # SVG rendering and display logic is no longer needed in the viewer.

    def log_message(self, message):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Scroll to the end


if __name__ == "__main__":
    app = SubstationViewer()
    app.mainloop()
