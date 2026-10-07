# Milestone 2: RK4/RK45 3-D Projectile Engine Design

**Date:** 2026-10-07  
**Repository:** `Aieda1l/FRC-REBUILT-Projectile-Simulator`  
**Branch:** `milestone-2-rk45-3d-engine`

## Purpose

Milestone 2 replaces the duplicated 2-D integration logic with matching 3-D physics cores in Python and JavaScript. The new engine must support true fixed-step RK4, adaptive Dormand-Prince RK45, arbitrary spin axes, robot motion, wind/air-relative velocity, and event interpolation while preserving the current simulator UI and legacy API behavior.

This milestone improves the numerical and geometric foundation only. It must not invent new FUEL aerodynamic measurements. The Milestone 1 FUEL assumptions remain explicit uncalibrated baselines until real trajectory/spin data exists.

## Success Criteria

The milestone is complete when:

1. Python and browser simulations use the same state definition, force equations, coefficient law, coordinate convention, and solver semantics.
2. The Python engine supports true RK4 and adaptive Dormand-Prince RK45.
3. The JavaScript engine supports the same RK4 and RK45 algorithms and agrees with checked-in Python golden fixtures within defined tolerances.
4. 3-D launch conditions support arbitrary position, muzzle velocity, robot field velocity, spin vector, and wind vector.
5. Drag uses air-relative velocity and Magnus lift uses only spin perpendicular to the airflow.
6. Optional calibrated spin decay is integrated as part of the ODE instead of applied after a translational step.
7. Ground and horizontal-plane crossings are interpolated to the crossing rather than reported at the first discrete sample beyond it.
8. Existing 2-D UI inputs continue to work through an adapter with x = downrange and z = vertical.
9. The legacy `/api/simulate` response remains compatible; a new 3-D endpoint exposes full state data and solver selection.
10. Existing Milestone 1 tests, new 3-D tests, Python/JavaScript parity tests, and the production frontend build all pass.

## Coordinate System and Units

Use one right-handed field coordinate system everywhere:

- **x:** forward/downrange, meters.
- **y:** left lateral, meters.
- **z:** up, meters.
- Linear velocity: m/s.
- Angular velocity: rad/s.
- Time: seconds.
- Mass: kg.
- Forces: N.

With a projectile moving in +x:

- backspin that produces upward lift is `omega = (0, negative, 0)`;
- positive z sidespin produces +y lateral lift.

The current 2-D UI maps its existing `(x, y_vertical)` positions to 3-D `(x, 0, z)`. Its scalar launch speed and elevation angle map to muzzle velocity `(v cos(theta), 0, v sin(theta))`. Its positive backspin RPM maps to angular velocity `(0, -omega, 0)`.

## Canonical State

The canonical state is a nine-element vector:

```text
[x, y, z, vx, vy, vz, omega_x, omega_y, omega_z]
```

Python stores this as a NumPy `float64` array. JavaScript stores the same ordered values in a numeric array.

A trajectory sample contains:

- `time: float`
- `state: state-vector`

Convenience serialization may expose position, velocity, spin, and speed separately, but the solver operates on the canonical state.

## Launch Composition

A launch is constructed from field position, shooter-relative muzzle velocity, spin vector, and robot field velocity.

```text
v_initial_field = muzzle_velocity_shooter + robot_velocity_field
```

For Milestone 2 the caller is responsible for expressing the shooter-relative muzzle vector in field axes. Robot orientation transforms are outside this milestone.

The legacy 2-D adapter constructs the field-axis muzzle vector from its launch angle and speed, then adds an optional robot velocity vector that defaults to zero.

## Flight Parameters

The Python core exposes a `FlightParameters` dataclass. The JavaScript core accepts an object with the same semantic fields.

Required/default fields:

- `mass = 0.215`
- `radius = 0.075`
- `drag_coefficient = 0.47`
- `lift_coefficient = 0.25`
- `air_density = 1.204`
- `gravity = 9.81`
- `wind = (0, 0, 0)`
- `enable_drag = true`
- `enable_magnus = true`
- `spin_decay_time_constant = None/null`

These FUEL aerodynamic coefficients are uncalibrated Milestone 1 baselines. No Reynolds-dependent `Cd` curve or new empirical `Cl` curve is introduced here.

Cross-sectional area is `A = pi * radius^2`.

## Force and ODE Model

Let:

```text
u = v - wind
speed = |u|
u_hat = u / speed
```

Gravity is:

```text
a_g = (0, 0, -gravity)
```

### Drag

When drag is enabled and `speed > epsilon`:

```text
F_drag = -0.5 * rho * A * Cd * speed^2 * u_hat
```

Otherwise drag is zero.

### Magnus Lift

Only angular velocity perpendicular to airflow contributes to lift:

```text
omega_perp = omega - dot(omega, u_hat) * u_hat
omega_perp_mag = |omega_perp|
S = radius * omega_perp_mag / speed
```

The existing Milestone 1 uncalibrated lift law remains:

```text
Cl_effective = lift_coefficient * min(S / 0.5, 1.0)
```

When Magnus is enabled, `speed > epsilon`, and `omega_perp_mag > epsilon`:

```text
lift_direction = normalize(cross(omega_perp, u_hat))
F_magnus = 0.5 * rho * A * Cl_effective * speed^2 * lift_direction
```

This convention produces +z lift for `v=(+x)` with backspin `omega=(0,-w,0)`, and +y lift for positive z sidespin.

### Spin Decay

If no calibrated spin-decay time constant is supplied:

```text
domega/dt = (0, 0, 0)
```

If `tau > 0` is supplied:

```text
domega/dt = -omega / tau
```

A non-positive configured time constant is invalid and must be rejected at parameter validation rather than silently treated as no decay.

### State Derivative

The pure derivative function returns:

```text
d(position)/dt = velocity
d(velocity)/dt = gravity + (F_drag + F_magnus) / mass
d(omega)/dt = spin-decay derivative
```

It must not mutate its input state or parameters.

## Numerical Integrators

### Fixed-Step RK4

The fixed-step solver is classical fourth-order Runge-Kutta over the entire nine-element state. All intermediate stages recompute drag, Magnus force, and spin derivative from the corresponding intermediate state.

`rk4_step(state, params, dt)` rejects `dt <= 0`.

RK4 is the default browser/UI solver because it is deterministic, fast for repeated optimization calls, and straightforward to compare against Python.

### Adaptive RK45

RK45 uses the Dormand-Prince 5(4) embedded pair. The accepted state is the fifth-order estimate; the fourth-order estimate is used only to calculate local error.

Default adaptive settings:

- initial `dt = 0.02 s`
- `rtol = 1e-6`
- `atol = 1e-9`
- `min_step = 1e-5 s`
- `max_step = 0.05 s`
- safety factor `0.9`
- minimum step factor `0.2`
- maximum step factor `5.0`

Error norm is a weighted RMS over all nine state elements:

```text
scale_i = atol + rtol * max(abs(y_i), abs(y5_i))
error_norm = sqrt(mean((error_i / scale_i)^2))
```

A step is accepted when `error_norm <= 1`. The next step size is multiplied by:

```text
clamp(0.2, 5.0, 0.9 * error_norm^(-1/5))
```

When `error_norm == 0`, use factor `5.0`.

If a rejected step would require a step smaller than `min_step`, raise a clear integration error rather than loop indefinitely.

The integrator must clip its final accepted step so an explicit `max_time` endpoint is represented exactly when no earlier terminal event occurs.

RK45 is the default for the new 3-D server endpoint.

## Integration API

Python module: `api/physics3d.py`.

Required public interfaces:

```python
@dataclass(frozen=True)
class FlightSample:
    time: float
    state: np.ndarray

@dataclass
class FlightParameters:
    ...

def launch_state(
    position: Sequence[float],
    muzzle_velocity: Sequence[float],
    spin: Sequence[float],
    robot_velocity: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray: ...

def derivatives(state: np.ndarray, params: FlightParameters) -> np.ndarray: ...

def rk4_step(state: np.ndarray, params: FlightParameters, dt: float) -> np.ndarray: ...

def integrate_trajectory(
    initial_state: np.ndarray,
    params: FlightParameters,
    method: str = "rk4",
    dt: float = 0.001,
    max_time: float = 5.0,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_step: float = 1e-5,
    max_step: float = 0.05,
    terminal_height: float = 0.0,
) -> list[FlightSample]: ...
```

Accepted methods are exactly `"rk4"` and `"rk45"`. Unknown methods raise `ValueError`.

All public vector inputs must contain exactly three finite numeric values. State vectors must contain exactly nine finite numeric values. Mass and radius must be positive, density and gravity non-negative, and aerodynamic coefficients non-negative.

JavaScript module: `src/physics3d.js`. It mirrors these semantics with camelCase names:

- `launchState(...)`
- `derivatives(state, params)`
- `rk4Step(state, params, dt)`
- `integrateTrajectory(initialState, params, options)`

The JS state ordering and defaults must match Python exactly.

## Crossing Events

Milestone 2 implements interpolated horizontal-plane crossings, not the complete physical HUB collision model.

For an accepted step from sample A to B, a crossing of plane `z = h` is bracketed when the signed height relative to the plane changes sign in the requested direction.

Interpolation uses:

```text
alpha = (h - z_A) / (z_B - z_A)
t_cross = t_A + alpha * (t_B - t_A)
state_cross = state_A + alpha * (state_B - state_A)
```

Ground `z=0` is a terminal descending crossing. The returned final sample must lie exactly on `z=0`.

Target evaluation for the legacy UI uses the descending crossing of `z = target_height`. It checks the interpolated horizontal `(x,y)` center position against the current approximate opening-center clearance. Full hex-edge/rim/funnel contact is deferred.

## Legacy Python Compatibility

`api/trajectory_simulator.py` remains the public home of the existing 2-D dataclasses, optimizer, and helper functions during this milestone.

Its `TrajectorySimulator.simulate()` must delegate the actual integration to `api.physics3d` through a 2-D adapter instead of maintaining a second integration implementation.

Mapping:

- legacy position `(x, y_vertical)` -> 3-D `(x, 0, z)`
- legacy velocity/angle -> 3-D muzzle velocity
- positive legacy spin -> `(0, -spin, 0)`
- 3-D sample `(x,z,vx,vz,omega_y)` -> legacy trajectory point fields
- legacy `method="rk4"` -> core RK4
- legacy `method="adaptive"` or `method="rk45"` -> core RK45
- legacy `method="euler"` is removed from supported simulation methods in Milestone 2 and must raise a clear error

Existing optimizers continue calling `TrajectorySimulator.simulate()`; therefore they automatically use the new engine without being rewritten.

## HTTP API

### Legacy endpoint

`POST /api/simulate` retains the current request and response shape.

Its implementation uses the 3-D core through the legacy adapter. Existing callers must not need changes.

### New endpoint

Add `POST /api/simulate3d`.

Request fields:

- `position: [x,y,z]`
- `muzzle_velocity: [vx,vy,vz]`
- `spin: [wx,wy,wz]`
- `robot_velocity: [vx,vy,vz]`, default zero
- `wind: [vx,vy,vz]`, default zero
- `method: "rk4" | "rk45"`, default `"rk45"`
- optional solver fields `dt`, `max_time`, `rtol`, `atol`, `min_step`, `max_step`
- optional physical overrides `mass`, `radius`, `drag_coefficient`, `lift_coefficient`, `air_density`, `gravity`, `spin_decay_time_constant`, `enable_drag`, `enable_magnus`

Response fields:

- `success: true`
- `method`
- `samples`, each with `time`, `position[3]`, `velocity[3]`, `spin[3]`, and `speed`
- `flight_time`
- `range` as horizontal displacement magnitude from the initial x/y position
- `max_height`

Validation errors use FastAPI/Pydantic's normal 422 response. Integration failures caused by an impossible adaptive step return a clear 400-level response rather than an unhandled 500.

## Browser Integration

Move simulation math out of `src/TrajectorySimulator.jsx` into `src/physics3d.js`.

The existing UI remains visually 2-D for this milestone. It calls a small adapter that creates a 3-D initial state with y=0, maps positive backspin to negative y angular velocity, uses zero robot velocity and wind unless supplied programmatically, and projects returned samples back to x/z for the SVG chart.

The current UI uses RK4 by default. RK45 is implemented and tested in the JS core but does not require a new UI control in Milestone 2.

The angle/velocity optimizers continue using the same UI-facing simulation function, which delegates to the new JS core. This avoids network round trips during parameter sweeps.

## Python/JavaScript Parity

Python is the reference implementation for checked-in golden fixtures, but neither language receives different physics rules.

Add a deterministic fixture generator that writes representative states/trajectories to `tests/fixtures/physics3d_golden.json`. Generated cases cover:

1. vacuum flight;
2. drag only;
3. backspin;
4. sidespin;
5. wind matching projectile velocity;
6. robot field velocity addition;
7. calibrated spin decay;
8. RK45 adaptive integration.

The fixture file includes its generator version/schema and exact input parameters.

JavaScript tests use Node's built-in `node:test`; no JS testing framework dependency is required.

Parity tolerances:

- derivative/state-step fixtures: absolute tolerance `1e-10`;
- fixed-step trajectory samples: absolute tolerance `1e-8`;
- RK45 final states: absolute tolerance `2e-6`.

Golden fixtures are regenerated only intentionally and committed with the code change that justifies the numerical difference.

## Testing Strategy

### Python invariant tests

Tests must cover:

- launch adds robot velocity;
- gravity-only RK4 matches the analytic projectile solution;
- RK4 shows fourth-order convergence on a non-trivial smooth flight interval;
- RK45 converges to a fine-step RK4 reference;
- adaptive integration reaches explicit `max_time` exactly when no event terminates it;
- drag acceleration opposes air-relative velocity;
- zero relative wind produces zero aerodynamic force;
- Magnus acceleration is perpendicular to air-relative velocity;
- backspin lifts and sidespin deflects in the documented directions;
- spin parallel to airflow produces zero Magnus lift;
- calibrated exponential spin decay is integrated correctly;
- no configured decay preserves spin;
- descending ground crossing is interpolated and terminal;
- invalid state/vector lengths, non-finite values, non-positive mass/radius, bad step sizes, invalid tolerances, and unsupported methods are rejected.

The existing `tests/test_physics3d.py` red tests on the milestone branch are retained and expanded rather than replaced.

### JavaScript tests

Tests cover the same core sign conventions and the checked-in parity fixtures. They run without a browser.

### Regression/build checks

CI must run:

```text
python -m unittest discover -s tests -v
node --test tests/physics3d_js.test.mjs
npm run build
```

## Error Handling

The core must fail early on invalid numeric inputs rather than producing NaNs deep in integration.

Near-zero air-relative speed produces zero drag and Magnus force without division by zero.

Near-zero perpendicular spin produces zero Magnus force.

Adaptive integration has a hard minimum step and raises an explicit error when tolerance cannot be met at that step.

A trajectory that reaches `max_time` without hitting the ground is valid; its last sample is the exact max-time state and is not fabricated as a ground impact.

## File Structure

Planned responsibilities:

- `api/physics3d.py` — canonical Python 3-D state, forces, RK4, RK45, and crossing integration.
- `api/trajectory_simulator.py` — legacy 2-D types/optimizers plus adapter into `physics3d`.
- `api/main.py` — legacy HTTP endpoint plus new `/api/simulate3d`.
- `src/physics3d.js` — JavaScript mirror of the canonical engine.
- `src/TrajectorySimulator.jsx` — UI, controls, projection, optimization orchestration; no force/integrator implementation.
- `tests/test_physics3d.py` — Python 3-D invariant and solver tests.
- `tests/physics3d_js.test.mjs` — JavaScript invariant and golden-parity tests.
- `tests/fixtures/physics3d_golden.json` — checked-in deterministic parity data.
- `scripts/generate_physics3d_fixtures.py` — explicit fixture generator.
- `.github/workflows/physics-regression.yml` — runs Python tests, JS parity tests, and frontend build.

## Non-Goals

Milestone 2 does not include:

- fitting `Cd(Re)` from FUEL data;
- replacing the simple `Cl(S)` baseline with an empirical FUEL curve;
- measuring or guessing a FUEL spin-decay constant;
- robot-heading/quaternion transforms;
- Coriolis/gyroscopic attitude modeling of a deformable ball;
- full hexagonal HUB rim/funnel collision contact;
- bounce, rebound, or ball deformation on target contact;
- a new 3-D UI renderer;
- redesigning the controls or visual language.

These can build on the 3-D state and event interfaces later.

## Acceptance Checklist

Milestone 2 is acceptable only if all of the following are true:

- [ ] Python 3-D core exists with documented coordinate/sign conventions.
- [ ] True whole-state RK4 passes analytic and convergence tests.
- [ ] Dormand-Prince RK45 adapts/rejects steps and passes tolerance tests.
- [ ] Robot velocity and wind are represented in field coordinates.
- [ ] Drag and Magnus use air-relative velocity.
- [ ] Arbitrary spin vectors produce correct backspin/sidespin behavior.
- [ ] Spin decay, when configured, is part of the integrated ODE.
- [ ] Ground and target-height plane crossings are interpolated.
- [ ] Legacy Python simulation delegates to the new core.
- [ ] Legacy `/api/simulate` remains compatible.
- [ ] New `/api/simulate3d` returns full 3-D samples.
- [ ] Browser simulation delegates to `src/physics3d.js`.
- [ ] JS RK4/RK45 match Python golden fixtures within stated tolerances.
- [ ] Existing optimizer behavior continues through the adapter.
- [ ] All Python tests pass.
- [ ] All Node parity tests pass.
- [ ] `npm run build` passes.
