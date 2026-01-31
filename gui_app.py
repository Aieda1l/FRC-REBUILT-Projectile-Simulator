#!/usr/bin/env python3
"""
FRC 2026 Trajectory Simulator - GUI Application
================================================
A beautiful, modern interface for trajectory simulation and optimization.

Features:
- Real-time trajectory visualization
- Interactive parameter adjustment
- Error envelope visualization
- Comparison between ideal and realistic physics
- Export capabilities
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Arc, Circle, Polygon, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json
import csv

from trajectory_simulator import (
    PhysicsEngine, TrajectorySimulator, TrajectoryOptimizer,
    GamePiece, GamePieceProperties, EnvironmentConditions,
    LaunchParameters, Target, TrajectoryResult, ShooterConfig,
    compute_no_drag_angle, inches_to_meters, meters_to_inches,
    fps_to_mps, mps_to_fps, rpm_to_rads, rads_to_rpm
)


# Modern color palette
class Colors:
    # Primary colors
    BACKGROUND = '#0f0f1a'
    SURFACE = '#1a1a2e'
    SURFACE_LIGHT = '#252540'
    
    # Accent colors
    PRIMARY = '#6366f1'  # Indigo
    SECONDARY = '#22d3ee'  # Cyan
    SUCCESS = '#22c55e'  # Green
    WARNING = '#f59e0b'  # Amber
    ERROR = '#ef4444'  # Red
    
    # Trajectory colors
    TRAJECTORY_MAIN = '#6366f1'
    TRAJECTORY_IDEAL = '#22d3ee'
    TRAJECTORY_ERROR = '#f59e0b'
    TARGET = '#22c55e'
    LAUNCH = '#ef4444'
    
    # Text
    TEXT_PRIMARY = '#f8fafc'
    TEXT_SECONDARY = '#94a3b8'
    TEXT_MUTED = '#64748b'
    
    # Grid
    GRID = '#2d2d4a'
    GRID_MINOR = '#1f1f35'


class ModernStyle:
    """Apply modern styling to tkinter widgets."""
    
    @staticmethod
    def configure_ttk_style():
        style = ttk.Style()
        
        # Configure main theme
        style.theme_use('clam')
        
        # Frame styles
        style.configure('Modern.TFrame', background=Colors.SURFACE)
        style.configure('Dark.TFrame', background=Colors.BACKGROUND)
        
        # Label styles
        style.configure('Modern.TLabel',
            background=Colors.SURFACE,
            foreground=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 10))
        
        style.configure('Header.TLabel',
            background=Colors.SURFACE,
            foreground=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 12, 'bold'))
        
        style.configure('Value.TLabel',
            background=Colors.SURFACE,
            foreground=Colors.SECONDARY,
            font=('Consolas', 11, 'bold'))
        
        style.configure('Unit.TLabel',
            background=Colors.SURFACE,
            foreground=Colors.TEXT_MUTED,
            font=('Segoe UI', 9))
        
        # Button styles
        style.configure('Modern.TButton',
            background=Colors.PRIMARY,
            foreground=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 10, 'bold'),
            padding=(15, 8))
        
        style.map('Modern.TButton',
            background=[('active', Colors.SECONDARY)],
            foreground=[('active', Colors.BACKGROUND)])
        
        style.configure('Accent.TButton',
            background=Colors.SUCCESS,
            foreground=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 10, 'bold'),
            padding=(15, 8))
        
        # Entry styles
        style.configure('Modern.TEntry',
            fieldbackground=Colors.SURFACE_LIGHT,
            foreground=Colors.TEXT_PRIMARY,
            insertcolor=Colors.TEXT_PRIMARY,
            font=('Consolas', 10))
        
        # Scale styles
        style.configure('Modern.Horizontal.TScale',
            background=Colors.SURFACE,
            troughcolor=Colors.SURFACE_LIGHT,
            sliderthickness=20)
        
        # Notebook styles
        style.configure('Modern.TNotebook',
            background=Colors.BACKGROUND,
            tabmargins=[2, 5, 2, 0])
        
        style.configure('Modern.TNotebook.Tab',
            background=Colors.SURFACE,
            foreground=Colors.TEXT_SECONDARY,
            padding=[15, 8],
            font=('Segoe UI', 10))
        
        style.map('Modern.TNotebook.Tab',
            background=[('selected', Colors.PRIMARY)],
            foreground=[('selected', Colors.TEXT_PRIMARY)])
        
        # Checkbox
        style.configure('Modern.TCheckbutton',
            background=Colors.SURFACE,
            foreground=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 10))
        
        # Combobox
        style.configure('Modern.TCombobox',
            fieldbackground=Colors.SURFACE_LIGHT,
            background=Colors.SURFACE,
            foreground=Colors.TEXT_PRIMARY,
            arrowcolor=Colors.TEXT_PRIMARY,
            font=('Segoe UI', 10))
        
        # Separator
        style.configure('Modern.TSeparator',
            background=Colors.GRID)
        
        return style


class ParameterSlider(ttk.Frame):
    """Custom slider widget with label and value display."""
    
    def __init__(
        self,
        parent,
        label: str,
        min_val: float,
        max_val: float,
        initial: float,
        unit: str = "",
        resolution: float = 0.1,
        command=None
    ):
        super().__init__(parent, style='Modern.TFrame')
        
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        self.command = command
        
        self.var = tk.DoubleVar(value=initial)
        
        # Layout
        self.columnconfigure(1, weight=1)
        
        # Label
        ttk.Label(self, text=label, style='Modern.TLabel', width=15, anchor='w').grid(
            row=0, column=0, padx=(5, 10), pady=2, sticky='w')
        
        # Slider
        self.slider = ttk.Scale(
            self,
            from_=min_val,
            to=max_val,
            variable=self.var,
            orient='horizontal',
            style='Modern.Horizontal.TScale',
            command=self._on_change
        )
        self.slider.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        
        # Value display
        self.value_label = ttk.Label(self, text=f"{initial:.1f}", style='Value.TLabel', width=8, anchor='e')
        self.value_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Unit
        ttk.Label(self, text=unit, style='Unit.TLabel', width=6, anchor='w').grid(
            row=0, column=3, padx=(0, 5), pady=2)
    
    def _on_change(self, _=None):
        value = round(self.var.get() / self.resolution) * self.resolution
        self.value_label.configure(text=f"{value:.1f}")
        if self.command:
            self.command(value)
    
    def get(self) -> float:
        return round(self.var.get() / self.resolution) * self.resolution
    
    def set(self, value: float):
        self.var.set(value)
        self._on_change()


class ParameterEntry(ttk.Frame):
    """Custom entry widget with label and unit."""
    
    def __init__(
        self,
        parent,
        label: str,
        initial: float,
        unit: str = "",
        width: int = 10
    ):
        super().__init__(parent, style='Modern.TFrame')
        
        self.var = tk.StringVar(value=str(initial))
        
        # Layout
        self.columnconfigure(1, weight=1)
        
        # Label
        ttk.Label(self, text=label, style='Modern.TLabel', width=15, anchor='w').grid(
            row=0, column=0, padx=(5, 10), pady=2, sticky='w')
        
        # Entry
        self.entry = ttk.Entry(self, textvariable=self.var, style='Modern.TEntry', width=width)
        self.entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        
        # Unit
        ttk.Label(self, text=unit, style='Unit.TLabel', width=8, anchor='w').grid(
            row=0, column=2, padx=(0, 5), pady=2)
    
    def get(self) -> float:
        try:
            return float(self.var.get())
        except ValueError:
            return 0.0
    
    def set(self, value: float):
        self.var.set(str(value))


class TrajectoryApp:
    """Main application class."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FRC 2026 Trajectory Simulator")
        self.root.geometry("1600x900")
        self.root.configure(bg=Colors.BACKGROUND)
        
        # Apply modern styling
        self.style = ModernStyle.configure_ttk_style()
        
        # Initialize simulation components
        self.init_simulation()
        
        # Build UI
        self.build_ui()
        
        # Initial simulation
        self.update_simulation()
    
    def init_simulation(self):
        """Initialize physics and simulation components."""
        # Game piece (default to 2026 game piece)
        self.game_piece = GamePieceProperties.from_game_piece(GamePiece.CORAL)
        
        # Environment
        self.environment = EnvironmentConditions(
            temperature_celsius=22,
            altitude_meters=0
        )
        
        # Physics engine
        self.physics = PhysicsEngine(self.game_piece, self.environment)
        
        # Simulator
        self.simulator = TrajectorySimulator(self.physics, dt=0.0005, max_time=3.0)
        
        # Optimizer
        self.optimizer = TrajectoryOptimizer(self.simulator)
        
        # Default target (FRC-style hub/goal)
        # Based on typical FRC scoring element dimensions
        self.target = Target(
            name="Hub",
            position=(0, 2.64),  # ~8.5 ft high
            entry_radius=0.61,  # ~24 inch diameter opening
            funnel_radius=1.07,  # ~42 inch outer funnel
            height_at_funnel=2.89  # ~9.5 ft at outer edge
        )
        
        # Results storage
        self.current_result: Optional[TrajectoryResult] = None
        self.ideal_result: Optional[TrajectoryResult] = None
        self.error_results: List[TrajectoryResult] = []
    
    def build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_frame, style='Modern.TFrame', width=380)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.build_control_panel(left_panel)
        
        # Right panel (visualization)
        right_panel = ttk.Frame(main_frame, style='Dark.TFrame')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.build_visualization_panel(right_panel)
    
    def build_control_panel(self, parent):
        """Build the control panel with parameters."""
        # Title
        title_frame = ttk.Frame(parent, style='Modern.TFrame')
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(
            title_frame,
            text="⚡ FRC Trajectory Simulator",
            style='Header.TLabel',
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(anchor='w')
        
        subtitle = ttk.Label(
            title_frame,
            text="2026 Season • Advanced Physics",
            style='Modern.TLabel',
            foreground=Colors.TEXT_SECONDARY
        )
        subtitle.pack(anchor='w')
        
        # Separator
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=5)
        
        # Notebook for tabs
        notebook = ttk.Notebook(parent, style='Modern.TNotebook')
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Launch Parameters Tab
        launch_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(launch_tab, text='  Launch  ')
        self.build_launch_tab(launch_tab)
        
        # Target Tab
        target_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(target_tab, text='  Target  ')
        self.build_target_tab(target_tab)
        
        # Physics Tab
        physics_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(physics_tab, text='  Physics  ')
        self.build_physics_tab(physics_tab)
        
        # Results Tab
        results_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(results_tab, text='  Results  ')
        self.build_results_tab(results_tab)
        
        # Action buttons
        button_frame = ttk.Frame(parent, style='Modern.TFrame')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="🎯 Find Optimal Angle",
            style='Accent.TButton',
            command=self.find_optimal_angle
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            button_frame,
            text="⚡ Find Optimal Velocity",
            style='Modern.TButton',
            command=self.find_optimal_velocity
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            button_frame,
            text="🎯 Find Best V + Angle",
            style='Modern.TButton',
            command=self.find_optimal_both
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            button_frame,
            text="🔄 Simulate",
            style='Modern.TButton',
            command=self.update_simulation
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            button_frame,
            text="📊 Export Data",
            style='Modern.TButton',
            command=self.export_data
        ).pack(fill='x', pady=2)
    
    def build_launch_tab(self, parent):
        """Build launch parameters tab."""
        # Position section
        section_label = ttk.Label(parent, text="Launch Position", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.launch_x = ParameterSlider(
            parent, "Distance (X)", -5.0, -0.5, -3.0, "m", 0.05,
            command=lambda _: self.update_simulation()
        )
        self.launch_x.pack(fill='x', padx=5, pady=2)
        
        self.launch_y = ParameterSlider(
            parent, "Height (Y)", 0.1, 2.0, 0.5, "m", 0.05,
            command=lambda _: self.update_simulation()
        )
        self.launch_y.pack(fill='x', padx=5, pady=2)
        
        # Velocity section
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Launch Velocity", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.velocity = ParameterSlider(
            parent, "Speed", 5.0, 25.0, 12.0, "m/s", 0.1,
            command=lambda _: self.update_simulation()
        )
        self.velocity.pack(fill='x', padx=5, pady=2)
        
        # Convert display
        velocity_info = ttk.Frame(parent, style='Modern.TFrame')
        velocity_info.pack(fill='x', padx=15, pady=2)
        self.velocity_fps_label = ttk.Label(
            velocity_info, text="≈ 39.4 ft/s", style='Unit.TLabel'
        )
        self.velocity_fps_label.pack(side='left')
        
        self.angle = ParameterSlider(
            parent, "Angle", 10, 80, 45, "°", 0.5,
            command=lambda _: self.update_simulation()
        )
        self.angle.pack(fill='x', padx=5, pady=2)
        
        # Spin section
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Backspin", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.spin = ParameterSlider(
            parent, "Spin Rate", 0, 5000, 2000, "RPM", 50,
            command=lambda _: self.update_simulation()
        )
        self.spin.pack(fill='x', padx=5, pady=2)
        
        # Error margins
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Error Margins (for envelope)", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.velocity_error = ParameterSlider(
            parent, "Velocity ±", 0, 3.0, 0.5, "m/s", 0.1,
            command=lambda _: self.update_simulation()
        )
        self.velocity_error.pack(fill='x', padx=5, pady=2)
        
        self.angle_error = ParameterSlider(
            parent, "Angle ±", 0, 5.0, 1.0, "°", 0.1,
            command=lambda _: self.update_simulation()
        )
        self.angle_error.pack(fill='x', padx=5, pady=2)
    
    def build_target_tab(self, parent):
        """Build target configuration tab."""
        section_label = ttk.Label(parent, text="Target Configuration", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        # Target presets
        preset_frame = ttk.Frame(parent, style='Modern.TFrame')
        preset_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(preset_frame, text="Preset:", style='Modern.TLabel').pack(side='left', padx=5)
        
        self.target_preset = ttk.Combobox(
            preset_frame,
            values=["FRC 2026 Hub (Est.)", "FRC 2024 Speaker", "FRC 2022 Hub", "Custom"],
            state='readonly',
            style='Modern.TCombobox',
            width=20
        )
        self.target_preset.set("FRC 2026 Hub (Est.)")
        self.target_preset.pack(side='left', padx=5)
        self.target_preset.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        
        # Target position
        section_label = ttk.Label(parent, text="Target Position", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.target_x = ParameterEntry(parent, "X Position", 0, "m")
        self.target_x.pack(fill='x', padx=5, pady=2)
        
        self.target_y = ParameterEntry(parent, "Y Position", 2.64, "m")
        self.target_y.pack(fill='x', padx=5, pady=2)
        
        # Target dimensions
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Target Dimensions", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.entry_diameter = ParameterEntry(parent, "Entry Diameter", 48, "in")
        self.entry_diameter.pack(fill='x', padx=5, pady=2)
        
        self.funnel_diameter = ParameterEntry(parent, "Funnel Diameter", 84, "in")
        self.funnel_diameter.pack(fill='x', padx=5, pady=2)
        
        # Apply button
        ttk.Button(
            parent,
            text="Apply Target Settings",
            style='Modern.TButton',
            command=self.apply_target_settings
        ).pack(fill='x', padx=10, pady=10)
    
    def build_physics_tab(self, parent):
        """Build physics configuration tab."""
        section_label = ttk.Label(parent, text="Game Piece Properties", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        # Game piece selector
        piece_frame = ttk.Frame(parent, style='Modern.TFrame')
        piece_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(piece_frame, text="Game Piece:", style='Modern.TLabel').pack(side='left', padx=5)
        
        self.piece_selector = ttk.Combobox(
            piece_frame,
            values=["Coral (2026)", "Note (2024)", "Cargo (2022)", "Power Cell (2020)", "Custom"],
            state='readonly',
            style='Modern.TCombobox',
            width=18
        )
        self.piece_selector.set("Coral (2026)")
        self.piece_selector.pack(side='left', padx=5)
        self.piece_selector.bind('<<ComboboxSelected>>', self.on_piece_change)
        
        # Piece properties
        self.piece_mass = ParameterEntry(parent, "Mass", 235, "g")
        self.piece_mass.pack(fill='x', padx=5, pady=2)
        
        self.piece_radius = ParameterEntry(parent, "Radius", 4.0, "in")
        self.piece_radius.pack(fill='x', padx=5, pady=2)
        
        self.drag_coeff = ParameterEntry(parent, "Drag Coefficient", 0.47, "")
        self.drag_coeff.pack(fill='x', padx=5, pady=2)
        
        self.lift_coeff = ParameterEntry(parent, "Lift Coefficient", 0.25, "")
        self.lift_coeff.pack(fill='x', padx=5, pady=2)
        
        # Environment
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Environment", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.temperature = ParameterEntry(parent, "Temperature", 22, "°C")
        self.temperature.pack(fill='x', padx=5, pady=2)
        
        self.altitude = ParameterEntry(parent, "Altitude", 0, "m")
        self.altitude.pack(fill='x', padx=5, pady=2)
        
        # Display calculated air density
        self.air_density_label = ttk.Label(
            parent,
            text="Air Density: 1.225 kg/m³",
            style='Modern.TLabel'
        )
        self.air_density_label.pack(anchor='w', padx=15, pady=5)
        
        # Toggles
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Physics Options", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.enable_drag = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Enable Air Drag",
            variable=self.enable_drag,
            style='Modern.TCheckbutton',
            command=self.update_simulation
        ).pack(anchor='w', padx=15, pady=2)
        
        self.enable_magnus = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Enable Magnus Effect",
            variable=self.enable_magnus,
            style='Modern.TCheckbutton',
            command=self.update_simulation
        ).pack(anchor='w', padx=15, pady=2)
        
        self.show_ideal = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Show Ideal Trajectory (no drag)",
            variable=self.show_ideal,
            style='Modern.TCheckbutton',
            command=self.update_simulation
        ).pack(anchor='w', padx=15, pady=2)
        
        self.show_envelope = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Show Error Envelope",
            variable=self.show_envelope,
            style='Modern.TCheckbutton',
            command=self.update_simulation
        ).pack(anchor='w', padx=15, pady=2)
        
        # Apply button
        ttk.Button(
            parent,
            text="Apply Physics Settings",
            style='Modern.TButton',
            command=self.apply_physics_settings
        ).pack(fill='x', padx=10, pady=10)
        
        # Backspin Estimator Section
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=5)
        section_label = ttk.Label(parent, text="Backspin Estimator", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        self.flywheel_rpm = ParameterEntry(parent, "Flywheel RPM", 3500, "RPM")
        self.flywheel_rpm.pack(fill='x', padx=5, pady=2)
        
        self.flywheel_diameter = ParameterEntry(parent, "Flywheel Dia.", 4.0, "in")
        self.flywheel_diameter.pack(fill='x', padx=5, pady=2)
        
        self.ball_diameter = ParameterEntry(parent, "Ball Diameter", 5.0, "in")
        self.ball_diameter.pack(fill='x', padx=5, pady=2)
        
        self.hood_compression = ParameterEntry(parent, "Compression", 0.5, "in")
        self.hood_compression.pack(fill='x', padx=5, pady=2)
        
        ttk.Button(
            parent,
            text="Estimate & Apply Backspin",
            style='Modern.TButton',
            command=self.estimate_backspin
        ).pack(fill='x', padx=10, pady=5)
        
        self.estimated_backspin_label = ttk.Label(
            parent,
            text="Estimated backspin: -- RPM",
            style='Value.TLabel'
        )
        self.estimated_backspin_label.pack(anchor='w', padx=15, pady=2)
    
    def build_results_tab(self, parent):
        """Build results display tab."""
        section_label = ttk.Label(parent, text="Simulation Results", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        # Results frame
        results_frame = ttk.Frame(parent, style='Modern.TFrame')
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create result labels
        self.result_labels = {}
        
        results_data = [
            ("Hit Target", "hit_target", ""),
            ("Flight Time", "flight_time", "s"),
            ("Max Height", "max_height", "m"),
            ("Range", "range", "m"),
            ("Entry Velocity", "entry_velocity", "m/s"),
            ("Entry Angle", "entry_angle", "°"),
            ("Impact X", "impact_x", "m"),
            ("Impact Y", "impact_y", "m"),
            ("", "", ""),
            ("Optimal Angle", "optimal_angle", "°"),
            ("(No Drag)", "ideal_angle", "°"),
        ]
        
        for label, key, unit in results_data:
            if not label:
                ttk.Separator(results_frame, style='Modern.TSeparator').pack(fill='x', pady=5)
                continue
                
            row_frame = ttk.Frame(results_frame, style='Modern.TFrame')
            row_frame.pack(fill='x', pady=2)
            
            ttk.Label(row_frame, text=label, style='Modern.TLabel', width=15).pack(side='left')
            
            value_label = ttk.Label(row_frame, text="--", style='Value.TLabel', width=10, anchor='e')
            value_label.pack(side='left', padx=5)
            self.result_labels[key] = value_label
            
            ttk.Label(row_frame, text=unit, style='Unit.TLabel', width=6).pack(side='left')
        
        # Comparison section
        ttk.Separator(parent, style='Modern.TSeparator').pack(fill='x', padx=10, pady=10)
        section_label = ttk.Label(parent, text="Physics Comparison", style='Header.TLabel')
        section_label.pack(anchor='w', padx=10, pady=(5, 5))
        
        comp_frame = ttk.Frame(parent, style='Modern.TFrame')
        comp_frame.pack(fill='x', padx=10, pady=5)
        
        self.drag_effect_label = ttk.Label(
            comp_frame,
            text="Drag reduces range by: --",
            style='Modern.TLabel'
        )
        self.drag_effect_label.pack(anchor='w', pady=2)
        
        self.magnus_effect_label = ttk.Label(
            comp_frame,
            text="Magnus lift effect: --",
            style='Modern.TLabel'
        )
        self.magnus_effect_label.pack(anchor='w', pady=2)
    
    def build_visualization_panel(self, parent):
        """Build the visualization panel with matplotlib."""
        # Create figure with dark theme
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(12, 8), facecolor=Colors.BACKGROUND)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
        
        # Main trajectory plot
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(Colors.SURFACE)
        
        # Configure grid
        self.ax.grid(True, color=Colors.GRID, linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.grid(True, which='minor', color=Colors.GRID_MINOR, linestyle=':', linewidth=0.3, alpha=0.3)
        self.ax.minorticks_on()
        
        # Labels
        self.ax.set_xlabel('Horizontal Distance (m)', color=Colors.TEXT_PRIMARY, fontsize=11)
        self.ax.set_ylabel('Height (m)', color=Colors.TEXT_PRIMARY, fontsize=11)
        self.ax.set_title('Projectile Trajectory', color=Colors.TEXT_PRIMARY, fontsize=14, fontweight='bold')
        
        # Tick colors
        self.ax.tick_params(colors=Colors.TEXT_SECONDARY)
        for spine in self.ax.spines.values():
            spine.set_color(Colors.GRID)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        
        # Toolbar
        toolbar_frame = ttk.Frame(parent, style='Dark.TFrame')
        toolbar_frame.pack(side='bottom', fill='x')
        
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        toolbar.configure(background=Colors.SURFACE)
        
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_simulation(self):
        """Run simulation and update visualization."""
        # Get launch parameters
        launch = LaunchParameters(
            position=(self.launch_x.get(), self.launch_y.get()),
            velocity=self.velocity.get(),
            angle=self.angle.get(),
            spin_rate=rpm_to_rads(self.spin.get()) if self.enable_magnus.get() else 0
        )
        
        # Update velocity display
        fps = mps_to_fps(launch.velocity)
        self.velocity_fps_label.configure(text=f"≈ {fps:.1f} ft/s")
        
        # Temporarily modify physics if drag/magnus disabled
        original_drag = self.physics.drag_factor
        original_magnus = self.physics.magnus_factor
        
        if not self.enable_drag.get():
            self.physics.drag_factor = 0
        if not self.enable_magnus.get():
            self.physics.magnus_factor = 0
        
        # Run main simulation
        self.current_result = self.simulator.simulate(launch, self.target)
        
        # Compute error envelope
        if self.show_envelope.get():
            self.error_results = self.optimizer.compute_error_envelope(
                launch, self.target,
                velocity_error=self.velocity_error.get(),
                angle_error=self.angle_error.get()
            )
        else:
            self.error_results = []
        
        # Restore physics
        self.physics.drag_factor = original_drag
        self.physics.magnus_factor = original_magnus
        
        # Compute ideal (no drag) trajectory for comparison
        if self.show_ideal.get():
            # Save and zero drag
            saved_drag = self.physics.drag_factor
            saved_magnus = self.physics.magnus_factor
            self.physics.drag_factor = 0
            self.physics.magnus_factor = 0
            
            ideal_launch = LaunchParameters(
                position=launch.position,
                velocity=launch.velocity,
                angle=launch.angle,
                spin_rate=0
            )
            self.ideal_result = self.simulator.simulate(ideal_launch, self.target)
            
            # Restore
            self.physics.drag_factor = saved_drag
            self.physics.magnus_factor = saved_magnus
        else:
            self.ideal_result = None
        
        # Update visualization
        self.update_plot()
        
        # Update results
        self.update_results()
    
    def update_plot(self):
        """Update the trajectory plot."""
        self.ax.clear()
        
        # Reconfigure axes
        self.ax.set_facecolor(Colors.SURFACE)
        self.ax.grid(True, color=Colors.GRID, linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.grid(True, which='minor', color=Colors.GRID_MINOR, linestyle=':', linewidth=0.3, alpha=0.3)
        self.ax.minorticks_on()
        
        # Draw target
        self.draw_target()
        
        # Draw error envelope first (so it's behind main trajectory)
        if self.error_results and self.show_envelope.get():
            for result in self.error_results:
                t, x, y = result.get_arrays()
                self.ax.plot(x, y, color=Colors.TRAJECTORY_ERROR, alpha=0.15, linewidth=1)
        
        # Draw ideal trajectory
        if self.ideal_result and self.show_ideal.get():
            t, x, y = self.ideal_result.get_arrays()
            self.ax.plot(x, y, color=Colors.TRAJECTORY_IDEAL, linewidth=2,
                        linestyle='--', label='Ideal (no drag)', alpha=0.7)
        
        # Draw main trajectory
        if self.current_result:
            t, x, y = self.current_result.get_arrays()
            
            # Color gradient based on velocity
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Calculate velocities for coloring
            velocities = [p.speed for p in self.current_result.points]
            
            # Normalize velocities for color mapping
            norm = plt.Normalize(min(velocities), max(velocities))
            
            lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3)
            lc.set_array(np.array(velocities[:-1]))
            self.ax.add_collection(lc)
            
            # Add colorbar
            if hasattr(self, 'colorbar') and self.colorbar:
                self.colorbar.remove()
            self.colorbar = self.fig.colorbar(lc, ax=self.ax, label='Velocity (m/s)',
                                              shrink=0.6, pad=0.02)
            self.colorbar.ax.yaxis.label.set_color(Colors.TEXT_PRIMARY)
            self.colorbar.ax.tick_params(colors=Colors.TEXT_SECONDARY)
        
        # Draw launch point
        launch_x, launch_y = self.launch_x.get(), self.launch_y.get()
        self.ax.plot(launch_x, launch_y, 'o', color=Colors.LAUNCH, markersize=12,
                    markeredgecolor='white', markeredgewidth=2, label='Launch', zorder=5)
        
        # Draw velocity vector
        angle_rad = np.radians(self.angle.get())
        arrow_scale = 0.15 * self.velocity.get()
        dx = arrow_scale * np.cos(angle_rad)
        dy = arrow_scale * np.sin(angle_rad)
        self.ax.annotate('', xy=(launch_x + dx, launch_y + dy), xytext=(launch_x, launch_y),
                        arrowprops=dict(arrowstyle='->', color=Colors.LAUNCH, lw=2))
        
        # Draw impact point
        if self.current_result and self.current_result.impact_point:
            ix, iy = self.current_result.impact_point
            marker_color = Colors.SUCCESS if self.current_result.hit_target else Colors.ERROR
            self.ax.plot(ix, iy, 'X', color=marker_color, markersize=15,
                        markeredgecolor='white', markeredgewidth=2, zorder=5)
        
        # Set axis limits
        self.ax.set_xlim(launch_x - 0.5, 1.5)
        self.ax.set_ylim(-0.2, max(self.target.height_at_funnel + 0.5,
                                   self.current_result.max_height + 0.3 if self.current_result else 4))
        
        # Labels and legend
        self.ax.set_xlabel('Horizontal Distance (m)', color=Colors.TEXT_PRIMARY, fontsize=11)
        self.ax.set_ylabel('Height (m)', color=Colors.TEXT_PRIMARY, fontsize=11)
        
        hit_status = "✓ HIT" if (self.current_result and self.current_result.hit_target) else "✗ MISS"
        hit_color = Colors.SUCCESS if (self.current_result and self.current_result.hit_target) else Colors.ERROR
        
        self.ax.set_title(f'Projectile Trajectory — {hit_status}',
                         color=hit_color, fontsize=14, fontweight='bold')
        
        self.ax.tick_params(colors=Colors.TEXT_SECONDARY)
        for spine in self.ax.spines.values():
            spine.set_color(Colors.GRID)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color=Colors.TRAJECTORY_MAIN, linewidth=3, label='Trajectory'),
            Line2D([0], [0], color=Colors.TRAJECTORY_IDEAL, linewidth=2, linestyle='--', label='Ideal (no drag)'),
            Line2D([0], [0], color=Colors.TRAJECTORY_ERROR, linewidth=1, alpha=0.5, label='Error envelope'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=Colors.LAUNCH, markersize=10, label='Launch'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=Colors.SUCCESS, markersize=10, label='Impact'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right',
                      facecolor=Colors.SURFACE, edgecolor=Colors.GRID,
                      labelcolor=Colors.TEXT_PRIMARY, fontsize=9)
        
        self.canvas.draw()
    
    def draw_target(self):
        """Draw the target on the plot."""
        target = self.target
        
        # Draw funnel shape
        funnel_left = target.position[0] - target.funnel_radius
        funnel_right = target.position[0] + target.funnel_radius
        entry_left = target.position[0] - target.entry_radius
        entry_right = target.position[0] + target.entry_radius
        
        # Left funnel wall
        self.ax.plot([funnel_left, entry_left],
                    [target.height_at_funnel, target.position[1]],
                    color=Colors.TARGET, linewidth=3)
        
        # Right funnel wall
        self.ax.plot([entry_right, funnel_right],
                    [target.position[1], target.height_at_funnel],
                    color=Colors.TARGET, linewidth=3)
        
        # Entry line (dashed to show opening)
        self.ax.plot([entry_left, entry_right],
                    [target.position[1], target.position[1]],
                    color=Colors.TARGET, linewidth=2, linestyle='--', alpha=0.7)
        
        # Fill funnel area
        funnel_vertices = [
            (funnel_left, target.height_at_funnel),
            (entry_left, target.position[1]),
            (entry_right, target.position[1]),
            (funnel_right, target.height_at_funnel),
        ]
        funnel_poly = Polygon(funnel_vertices, alpha=0.15, facecolor=Colors.TARGET,
                             edgecolor='none')
        self.ax.add_patch(funnel_poly)
        
        # Target center marker
        self.ax.plot(target.position[0], target.position[1], 's',
                    color=Colors.TARGET, markersize=8, alpha=0.8)
        
        # Label
        self.ax.annotate(target.name, xy=target.position,
                        xytext=(target.position[0], target.position[1] + 0.3),
                        color=Colors.TEXT_PRIMARY, fontsize=10,
                        ha='center', fontweight='bold')
    
    def update_results(self):
        """Update the results display."""
        result = self.current_result
        
        if result:
            self.result_labels['hit_target'].configure(
                text="YES" if result.hit_target else "NO",
                foreground=Colors.SUCCESS if result.hit_target else Colors.ERROR
            )
            self.result_labels['flight_time'].configure(text=f"{result.flight_time:.3f}")
            self.result_labels['max_height'].configure(text=f"{result.max_height:.3f}")
            self.result_labels['range'].configure(text=f"{result.range_distance:.3f}")
            
            if result.entry_velocity:
                self.result_labels['entry_velocity'].configure(text=f"{result.entry_velocity:.2f}")
            else:
                self.result_labels['entry_velocity'].configure(text="--")
            
            if result.entry_angle:
                self.result_labels['entry_angle'].configure(text=f"{result.entry_angle:.1f}")
            else:
                self.result_labels['entry_angle'].configure(text="--")
            
            if result.impact_point:
                self.result_labels['impact_x'].configure(text=f"{result.impact_point[0]:.3f}")
                self.result_labels['impact_y'].configure(text=f"{result.impact_point[1]:.3f}")
        
        # Compute and display optimal angles
        ideal_angle = compute_no_drag_angle(
            self.launch_x.get(), self.launch_y.get(),
            self.target.position[0], self.target.position[1],
            self.velocity.get()
        )
        
        if ideal_angle:
            self.result_labels['ideal_angle'].configure(text=f"{ideal_angle:.1f}")
        else:
            self.result_labels['ideal_angle'].configure(text="N/A")
        
        # Calculate physics comparison
        if self.current_result and self.ideal_result:
            range_diff = self.ideal_result.range_distance - self.current_result.range_distance
            self.drag_effect_label.configure(
                text=f"Drag reduces range by: {range_diff:.2f} m ({range_diff/self.ideal_result.range_distance*100:.1f}%)"
            )
            
            height_diff = self.current_result.max_height - (self.ideal_result.max_height if self.ideal_result else 0)
            if self.enable_magnus.get() and self.spin.get() > 0:
                self.magnus_effect_label.configure(
                    text=f"Magnus effect on max height: {height_diff:+.3f} m"
                )
            else:
                self.magnus_effect_label.configure(text="Magnus effect: Disabled")
    
    def find_optimal_angle(self):
        """Find and apply the optimal launch angle."""
        optimal = self.optimizer.find_optimal_angle(
            self.launch_x.get(),
            self.launch_y.get(),
            self.velocity.get(),
            rpm_to_rads(self.spin.get()) if self.enable_magnus.get() else 0,
            self.target
        )
        
        if optimal:
            self.angle.set(optimal)
            self.result_labels['optimal_angle'].configure(text=f"{optimal:.1f}")
            self.update_simulation()
            messagebox.showinfo("Optimal Angle Found", f"Optimal angle: {optimal:.1f}°")
        else:
            messagebox.showwarning("No Solution", 
                "Could not find an angle to hit the target with current parameters.\n"
                "Try increasing velocity or moving closer to the target.")
    
    def find_optimal_velocity(self):
        """Find and apply the optimal launch velocity for current angle."""
        optimal = self.optimizer.find_optimal_velocity(
            self.launch_x.get(),
            self.launch_y.get(),
            self.angle.get(),
            rpm_to_rads(self.spin.get()) if self.enable_magnus.get() else 0,
            self.target
        )
        
        if optimal:
            self.velocity.set(optimal)
            self.update_simulation()
            messagebox.showinfo("Optimal Velocity Found", 
                f"Optimal velocity: {optimal:.2f} m/s ({mps_to_fps(optimal):.1f} ft/s)")
        else:
            messagebox.showwarning("No Solution", 
                "Could not find a velocity to hit the target at this angle.\n"
                "Try adjusting the angle or moving closer to the target.")
    
    def find_optimal_both(self):
        """Find and apply both optimal velocity and angle."""
        result = self.optimizer.find_optimal_parameters(
            self.launch_x.get(),
            self.launch_y.get(),
            rpm_to_rads(self.spin.get()) if self.enable_magnus.get() else 0,
            self.target,
            optimize_for="accuracy"
        )
        
        if result:
            v, a, traj = result
            self.velocity.set(v)
            self.angle.set(a)
            self.update_simulation()
            messagebox.showinfo("Optimal Parameters Found", 
                f"Optimal velocity: {v:.2f} m/s ({mps_to_fps(v):.1f} ft/s)\n"
                f"Optimal angle: {a:.1f}°\n"
                f"Entry velocity: {traj.entry_velocity:.1f} m/s\n"
                f"Entry angle: {traj.entry_angle:.1f}°")
        else:
            messagebox.showwarning("No Solution", 
                "Could not find parameters to hit the target.\n"
                "Try moving closer to the target.")
    
    def estimate_backspin(self):
        """Estimate backspin from flywheel parameters and apply it."""
        shooter = ShooterConfig.create_single_flywheel(
            flywheel_diameter=self.flywheel_diameter.get(),
            flywheel_rpm=self.flywheel_rpm.get(),
            ball_diameter=self.ball_diameter.get(),
            compression=self.hood_compression.get()
        )
        
        backspin_rpm = shooter.estimate_backspin_rpm()
        exit_vel = shooter.estimate_exit_velocity()
        
        # Apply to spin slider
        self.spin.set(backspin_rpm)
        
        # Update label
        self.estimated_backspin_label.configure(
            text=f"Estimated: {backspin_rpm:.0f} RPM, Exit vel: {exit_vel:.1f} m/s"
        )
        
        self.update_simulation()
        
        messagebox.showinfo("Backspin Estimated",
            f"Flywheel: {self.flywheel_diameter.get()}\" @ {self.flywheel_rpm.get()} RPM\n"
            f"Ball: {self.ball_diameter.get()}\" diameter\n"
            f"Compression: {self.hood_compression.get()}\"\n\n"
            f"Estimated backspin: {backspin_rpm:.0f} RPM\n"
            f"Estimated exit velocity: {exit_vel:.1f} m/s ({mps_to_fps(exit_vel):.1f} ft/s)"
        )
    
    def on_preset_change(self, event):
        """Handle target preset selection."""
        preset = self.target_preset.get()
        
        presets = {
            "FRC 2026 Hub (Est.)": (0, 2.64, 48, 84),
            "FRC 2024 Speaker": (0, 2.05, 42, 72),
            "FRC 2022 Hub": (0, 2.64, 48, 84),
        }
        
        if preset in presets:
            x, y, entry_d, funnel_d = presets[preset]
            self.target_x.set(x)
            self.target_y.set(y)
            self.entry_diameter.set(entry_d)
            self.funnel_diameter.set(funnel_d)
            self.apply_target_settings()
    
    def on_piece_change(self, event):
        """Handle game piece selection."""
        piece_name = self.piece_selector.get()
        
        piece_map = {
            "Coral (2026)": GamePiece.CORAL,
            "Note (2024)": GamePiece.NOTE_2024,
            "Cargo (2022)": GamePiece.CARGO_2022,
            "Power Cell (2020)": GamePiece.POWER_CELL_2020,
        }
        
        if piece_name in piece_map:
            self.game_piece = GamePieceProperties.from_game_piece(piece_map[piece_name])
            self.piece_mass.set(self.game_piece.mass * 1000)  # kg to g
            self.piece_radius.set(meters_to_inches(self.game_piece.radius))
            self.drag_coeff.set(self.game_piece.drag_coefficient)
            self.lift_coeff.set(self.game_piece.lift_coefficient)
            self.apply_physics_settings()
    
    def apply_target_settings(self):
        """Apply target configuration from UI."""
        self.target = Target(
            name="Hub",
            position=(self.target_x.get(), self.target_y.get()),
            entry_radius=inches_to_meters(self.entry_diameter.get() / 2),
            funnel_radius=inches_to_meters(self.funnel_diameter.get() / 2),
            height_at_funnel=self.target_y.get() + 0.25  # Slight rise at funnel edge
        )
        self.update_simulation()
    
    def apply_physics_settings(self):
        """Apply physics configuration from UI."""
        # Update game piece
        self.game_piece = GamePieceProperties(
            name="Custom",
            mass=self.piece_mass.get() / 1000,  # g to kg
            radius=inches_to_meters(self.piece_radius.get()),
            drag_coefficient=self.drag_coeff.get(),
            lift_coefficient=self.lift_coeff.get(),
            moment_of_inertia=0.4 * (self.piece_mass.get() / 1000) * inches_to_meters(self.piece_radius.get())**2
        )
        
        # Update environment
        self.environment = EnvironmentConditions(
            temperature_celsius=self.temperature.get(),
            altitude_meters=self.altitude.get()
        )
        
        self.air_density_label.configure(text=f"Air Density: {self.environment.air_density:.4f} kg/m³")
        
        # Recreate physics engine and simulator
        self.physics = PhysicsEngine(self.game_piece, self.environment)
        self.simulator = TrajectorySimulator(self.physics, dt=0.0005, max_time=3.0)
        self.optimizer = TrajectoryOptimizer(self.simulator)
        
        self.update_simulation()
    
    def export_data(self):
        """Export trajectory data to CSV."""
        if not self.current_result:
            messagebox.showwarning("No Data", "Run a simulation first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export Trajectory Data'
        )
        
        if filepath:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'X (m)', 'Y (m)', 'Vx (m/s)', 'Vy (m/s)', 'Speed (m/s)', 'Spin (rad/s)'])
                
                for point in self.current_result.points:
                    writer.writerow([
                        f"{point.time:.6f}",
                        f"{point.x:.6f}",
                        f"{point.y:.6f}",
                        f"{point.vx:.6f}",
                        f"{point.vy:.6f}",
                        f"{point.speed:.6f}",
                        f"{point.spin:.6f}"
                    ])
            
            messagebox.showinfo("Export Complete", f"Data exported to:\n{filepath}")


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Create and run app
    app = TrajectoryApp(root)
    
    # Handle window close
    def on_closing():
        plt.close('all')
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
