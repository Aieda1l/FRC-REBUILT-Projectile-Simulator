# FRC 2026 Projectile Trajectory Simulator

A comprehensive physics simulator for FIRST Robotics Competition (FRC) teams to calculate and optimize shooting trajectories with realistic effects including air resistance and Magnus effect (backspin).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FRC](https://img.shields.io/badge/FRC-2026-red.svg)

## Features

### Realistic Physics Modeling
- **Gravitational acceleration** - Standard 9.81 m/s² with altitude correction
- **Quadratic air drag** - Realistic drag force using Cd × ½ρAv²
- **Magnus effect** - Lift from backspin using spin parameter model
- **Spin decay** - Models how spin rate decreases during flight
- **Environmental conditions** - Temperature and altitude affect air density

### Interactive GUI
- Real-time trajectory visualization with velocity color mapping
- Interactive parameter sliders for instant feedback
- Error envelope visualization for consistency analysis
- Comparison between ideal (no drag) and realistic trajectories
- Multiple game piece presets (2024 Note, 2022 Cargo, etc.)

### Optimization Tools
- Automatic optimal angle finder
- Minimum velocity calculator
- Error margin analysis
- Data export to CSV

## Installation

### Prerequisites
- Python 3.8 or higher
- tkinter (usually included with Python)

### Quick Start

```bash
# Clone or download the project
cd frc_trajectory_simulator

# Install dependencies
pip install -r requirements.txt

# Run the application
python gui_app.py
```

### Dependencies
- `numpy` - Numerical computations
- `matplotlib` - Visualization

## Usage

### GUI Application

Launch the GUI with:
```bash
python gui_app.py
```

#### Launch Tab
Configure the shooter parameters:
- **Distance (X)**: Horizontal distance from robot to target (negative = behind target)
- **Height (Y)**: Launch height above ground
- **Speed**: Initial projectile velocity in m/s
- **Angle**: Launch angle in degrees from horizontal
- **Spin Rate**: Backspin in RPM (positive = backspin for lift)

#### Target Tab
Configure the scoring target:
- Choose from presets (FRC 2026 Hub, 2024 Speaker, 2022 Hub)
- Or enter custom dimensions

#### Physics Tab
Fine-tune the physics model:
- Select game piece type or enter custom properties
- Adjust environmental conditions
- Enable/disable drag and Magnus effect
- Show comparison trajectories

#### Results Tab
View simulation results:
- Hit/miss status
- Flight time and range
- Entry velocity and angle
- Physics effect comparisons

### Programmatic Usage

```python
from trajectory_simulator import (
    PhysicsEngine, TrajectorySimulator, TrajectoryOptimizer,
    GamePiece, GamePieceProperties, EnvironmentConditions,
    LaunchParameters, Target, rpm_to_rads
)

# Create game piece and environment
piece = GamePieceProperties.from_game_piece(GamePiece.CORAL)
env = EnvironmentConditions(temperature_celsius=22, altitude_meters=100)

# Create physics engine
physics = PhysicsEngine(piece, env)
simulator = TrajectorySimulator(physics, dt=0.001)

# Define target
target = Target(
    name="Hub",
    position=(0, 2.64),  # x, y in meters
    entry_radius=0.61,    # opening radius
    funnel_radius=1.07,   # outer funnel radius
    height_at_funnel=2.89
)

# Define launch parameters
launch = LaunchParameters(
    position=(-3.0, 0.5),  # 3m back, 0.5m high
    velocity=12.0,          # m/s
    angle=45.0,             # degrees
    spin_rate=rpm_to_rads(2000)  # 2000 RPM backspin
)

# Run simulation
result = simulator.simulate(launch, target)

print(f"Hit target: {result.hit_target}")
print(f"Flight time: {result.flight_time:.3f} s")
print(f"Max height: {result.max_height:.3f} m")
print(f"Range: {result.range_distance:.3f} m")
```

### Finding Optimal Parameters

```python
# Create optimizer
optimizer = TrajectoryOptimizer(simulator)

# Find optimal angle for given velocity
optimal_angle = optimizer.find_optimal_angle(
    launch_x=-3.0,
    launch_y=0.5,
    velocity=12.0,
    spin=rpm_to_rads(2000),
    target=target
)

print(f"Optimal angle: {optimal_angle:.1f}°")

# Find minimum velocity needed
min_vel, angle = optimizer.find_minimum_velocity(
    launch_x=-3.0,
    launch_y=0.5,
    spin=rpm_to_rads(2000),
    target=target
)

print(f"Minimum velocity: {min_vel:.1f} m/s at {angle:.1f}°")
```

## Physics Model Details

### Air Drag
The simulator uses the standard quadratic drag model:

```
F_drag = -½ ρ A Cd v² v̂
```

Where:
- ρ = air density (kg/m³)
- A = cross-sectional area (πr²)
- Cd = drag coefficient (~0.47 for spheres)
- v = velocity magnitude

### Magnus Effect
The Magnus force from spin is modeled as:

```
F_magnus = ½ ρ A Cl v² (ω × v̂)
```

Where:
- Cl = lift coefficient (depends on spin parameter S = ωr/v)
- ω = angular velocity (rad/s)

For backspin (top of ball moving opposite to flight direction), this creates an upward force that increases range and allows for a flatter trajectory.

### Integration Method
The simulator uses 4th-order Runge-Kutta (RK4) integration for accurate trajectory computation with configurable time steps.

## Tips for FRC Teams

### Tuning Your Shooter
1. **Start with ideal trajectory** - Disable drag to find theoretical optimal angle
2. **Enable drag** - Note how much the angle needs to increase
3. **Add spin** - See how backspin can recover some of the lost range
4. **Analyze error envelope** - Ensure your consistency margins still hit the target

### Common Issues
- **Can't reach target**: Increase velocity or move closer
- **Overshooting**: Reduce velocity or use lower angle
- **Inconsistent shots**: Check error envelope - may need tighter tolerances

### Recommended Spin Rates
- Light game pieces (200g): 1500-2500 RPM
- Medium game pieces (270g): 2000-3500 RPM
- The Magnus effect is most noticeable at higher velocities

## File Structure

```
frc_trajectory_simulator/
├── trajectory_simulator.py  # Core physics engine
├── gui_app.py               # GUI application
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Customization

### Adding New Game Pieces
Edit `trajectory_simulator.py` and add to the `GamePiece` enum and `GamePieceProperties.from_game_piece()` method.

### Adjusting Physics
Modify the `PhysicsEngine` class to add additional effects like:
- Wind resistance
- Variable spin axis
- Ground bounce physics

## Contributing

Contributions welcome! Areas for improvement:
- 3D trajectory visualization
- Robot chassis integration
- Sensor feedback simulation
- Network tables integration

## License

MIT License - feel free to use in your FRC projects!

## Acknowledgments

- Inspired by Desmos trajectory calculators used by FRC teams
- Physics models based on standard aerodynamics literature
- Built for the FIRST Robotics Competition community

---

**Good luck at competition!** 🤖
