# FRC 2026 Projectile Trajectory Simulator

A comprehensive web-based physics simulator for FIRST Robotics Competition (FRC) teams to calculate and optimize shooting trajectories. Features realistic effects including air resistance, Magnus effect (backspin), and interactive error analysis.

![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688.svg?logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### Realistic Physics Modeling
- **Gravitational acceleration**: Standard 9.81 m/s² with altitude correction.
- **Quadratic air drag**: Realistic drag force using $C_d \times \frac{1}{2}\rho Av^2$.
- **Magnus effect**: Lift from backspin using spin parameter models.
- **Spin decay**: Models how spin rate decreases during flight due to air friction.
- **Environment**: Adjust calculations based on air density (temperature/altitude).

### Interactive Web GUI
- **Real-time Visualization**: Instant SVG trajectory plotting using React.
- **Backspin Estimator**: Calculate approximate RPM based on flywheel specs (diameter, gearing, compression).
- **Optimization Tools**:
    - Auto-calculate optimal launch angle.
    - Find minimum required velocity.
    - "Best Fit" mode to optimize both velocity and angle simultaneously.
- **Error Envelope**: Visualize how small inconsistencies in shooter speed or angle affect accuracy.

## Project Structure

This project is structured as a modern full-stack application designed for Vercel deployment:

```text
frc-simulator/
├── api/                   # Python Backend (FastAPI) & Physics Engine
│   ├── index.py           # API Entry point
│   └── trajectory_simulator.py  # Core physics logic
├── src/                   # Frontend (React + Vite)
│   ├── components/
│   │   └── TrajectorySimulator.jsx # Main UI & Client-side simulation
│   ├── App.jsx
│   └── index.css          # Tailwind styling
├── public/                # Static assets
└── vercel.json            # Deployment configuration
```

## Installation & Local Development

### Prerequisites
- Node.js (v18+)
- Python (v3.9+)

### 1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/frc-trajectory-simulator.git
cd frc-trajectory-simulator

# Install Frontend Dependencies
npm install

# Install Backend Dependencies (Optional for local API testing)
pip install -r requirements.txt
```

### 2. Run Locally

To run the frontend (which handles the simulation visualization):

```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

> **Note:** The current React component performs physics calculations client-side for zero-latency feedback. The Python API is set up to allow for advanced server-side calculations or data logging in the future.

## Deployment

This project is optimized for **Vercel**.

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Deploy:
   ```bash
   vercel
   ```
3. Use default settings:
   - Build Command: `npm run build`
   - Output Directory: `dist`

Alternatively, push to GitHub and connect your repository to Vercel. The included `vercel.json` will automatically handle the Python/React hybrid build.

## Physics Model Details

### Air Drag
The simulator uses the standard quadratic drag model:
$$F_{drag} = -\frac{1}{2} \rho A C_d v^2 \hat{v}$$

### Magnus Effect
The Magnus force from backspin creates upward lift:
$$F_{magnus} = \frac{1}{2} \rho A C_l v^2 (\hat{\omega} \times \hat{v})$$
*Where $C_l$ (Lift Coefficient) varies based on the spin parameter (surface speed vs. translational speed).*

## Python API Usage

While the web interface is the primary tool, you can still use the physics engine programmatically for data analysis or scriptable optimizations.

```python
from api.trajectory_simulator import (
    PhysicsEngine, TrajectorySimulator, 
    GamePieceProperties, GamePiece, EnvironmentConditions,
    LaunchParameters, Target
)

# Setup
piece = GamePieceProperties.from_game_piece(GamePiece.CORAL)
env = EnvironmentConditions()
physics = PhysicsEngine(piece, env)
sim = TrajectorySimulator(physics)

# Run Simulation
launch = LaunchParameters(
    position=(-3.0, 0.5), 
    velocity=12.0, 
    angle=45.0, 
    spin_rate=209 # rad/s (~2000 RPM)
)
result = sim.simulate(launch)

print(f"Range: {result.range_distance:.2f}m")
print(f"Hit Target: {result.hit_target}")
```

## Tips for FRC Teams

1.  **Flywheel Tuning**: Use the "Backspin Calculator" toggle. Enter your wheel diameter and compression to see estimated backspin.
2.  **Error Envelopes**: Don't just find the perfect angle. Turn on the "Error Envelope" to see if a +/- 2° variance causes a miss. A robust shot is better than a perfect theoretical shot.
3.  **Ideal vs. Real**: Toggle "Show Ideal" to see how much gravity-only physics differs from the drag+lift model. This helps explain why standard kinematic equations fail for light game pieces like the 2024 Note or 2026 Coral.

## License

MIT License - Free to use for all FRC teams.

---
*Good luck at competition!* 🤖