# Calibrated Projectile Physics — Design

Date: 2026-10-07  
Status: Approved design, awaiting written-spec review  
Branch: `feat/calibrated-projectile-physics`

## 1. Purpose

Upgrade the FRC REBUILT projectile simulator from a physically reasonable but largely uncalibrated flight model into a calibration-ready, uncertainty-aware shooter model suitable for engineering decisions.

The objective is not to claim CFD-level fidelity. The objective is to make the simulator measurably more accurate for a specific robot, launcher, FUEL population, and operating envelope while preserving the current simple workflow for teams that do not yet have measured calibration data.

Success means:

1. Existing users can continue to run the current scalar-coefficient model without breaking changes.
2. Advanced users can supply calibrated aerodynamic models as functions of Reynolds number and spin parameter.
3. Browser and Python engines remain numerically consistent.
4. Moving-robot shots and wind are modeled through field-relative launch and air-relative flight velocity.
5. Optimizers can rank shots by robust clean-entry probability under parameter uncertainty.
6. Calibration tools can fit candidate models to recorded shot data and evaluate held-out validation error.
7. The UI distinguishes measured inputs from heuristic flywheel estimates.
8. Documentation makes the limits of calibrated models explicit and warns against extrapolation outside measured regimes.

## 2. Non-goals

This project will not:

- implement CFD or Navier–Stokes solvers;
- infer foam deformation or rebound coefficients without measurements;
- simulate detailed wheel/ball contact mechanics through finite-element methods;
- treat the current flywheel estimator as a physics-grounded substitute for measured exit velocity and spin;
- guarantee scoring from contact-heavy trajectories;
- replace conservative rigid-sphere HUB collision checks with speculative bounce behavior.

The existing rigid-sphere contact model remains the scoring authority for optimization. A panel or rim collision remains non-clean even if a real FUEL could deform and score.

## 3. Current system constraints

The canonical JavaScript engine in `src/physics3d.js` already has:

- 9-state 3-D integration: position, velocity, angular velocity;
- gravity;
- quadratic drag;
- vector Magnus direction;
- wind-relative aerodynamic velocity;
- optional exponential spin decay;
- fixed-step RK4;
- adaptive Dormand–Prince RK45;
- launch-state support for robot field velocity.

The browser path in `src/trajectory2d.js` currently exposes only a subset of that capability. It accepts scalar `dragCoeff` and `liftCoeff`, does not expose `robotVelocity` or wind in the normal UI path, and uses an uncalibrated lift model that ramps linearly with spin parameter and saturates at a fixed cap.

The optimizer in `src/optimizer.js` currently ranks deterministic candidates by geometric classification and clearance.

## 4. Architecture

The physics stack will be split into explicit responsibilities.

### 4.1 Aerodynamic model layer

Create `src/aerodynamics.js` and `api/aerodynamics.py`.

The layer exposes pure functions for:

- Reynolds number:
  [
  Re = \frac{\rho v D}{\mu}
  ]
- perpendicular spin parameter:
  [
  S = \frac{\omega_\perp r}{v}
  ]
- drag coefficient evaluation;
- lift coefficient evaluation;
- validation of model definitions;
- interpolation for tabulated calibration data.

Supported drag model kinds:

1. `constant`
   - compatibility mode;
   - returns one scalar `Cd`.

2. `table1d`
   - piecewise-linear interpolation over `Re -> Cd`;
   - clamps at the calibrated domain boundaries by default;
   - records that a clamp occurred so the UI/validation layer can flag extrapolation risk.

Supported lift model kinds:

1. `legacy-spin-cap`
   - reproduces current behavior exactly:
     [
     C_L = C_{L,max}\min(S/S_{sat},1)
     ]
   - default `S_sat = 0.5`.

2. `table1d`
   - `S -> Cl` interpolation.

3. `table2d`
   - bilinear interpolation over `(Re,S) -> Cl`.

This design deliberately avoids embedding a baseball, soccer-ball, or other unrelated empirical curve as a new default for FUEL. The simulator can ingest measured FUEL data instead.

### 4.2 Backward-compatible flight parameters

Extend flight parameters to accept either legacy scalars or explicit aerodynamic-model objects.

Conceptual shape:

```js
{
  mass,
  radius,
  airDensity,
  dynamicViscosity,
  gravity,
  wind,
  enableDrag,
  enableMagnus,
  spinDecayTimeConstant,

  // compatibility inputs
  dragCoefficient,
  liftCoefficient,

  // advanced inputs
  dragModel,
  liftModel
}
```

Precedence:

- if `dragModel` is supplied, it is authoritative;
- otherwise use `dragCoefficient` through a constant model;
- if `liftModel` is supplied, it is authoritative;
- otherwise reproduce the existing legacy spin-cap behavior using `liftCoefficient`.

This preserves existing call sites and golden fixtures.

### 4.3 Physics engine integration

`src/physics3d.js` and `api/physics3d.py` will remain responsible for ODE derivatives and integration.

At every derivative evaluation:

1. compute air-relative velocity;
2. compute speed;
3. compute Reynolds number using configurable dynamic viscosity;
4. compute spin perpendicular to airflow;
5. compute spin parameter;
6. evaluate `Cd(Re)`;
7. evaluate `Cl(Re,S)`;
8. apply drag and Magnus force using those coefficients;
9. integrate spin decay if configured.

The vector direction logic remains unchanged.

Default air dynamic viscosity will be a room-temperature approximation suitable for the existing default environment. Environment adapters may derive a more precise value later, but temperature-dependent viscosity is not required for the first implementation.

### 4.4 Moving robot and wind

Expose the canonical engine's existing launch transform through `simulateShot` and the UI.

New shot parameters:

```js
robotVelocity: [vx, vy, vz]
wind: [vx, vy, vz]
```

Semantics:

- muzzle velocity is shooter-relative;
- robot velocity is field-relative and added once at launch;
- wind is field-relative and subtracted from projectile field velocity inside aerodynamic calculations.

The browser UI will initially expose horizontal/downrange and lateral robot velocity plus horizontal/lateral wind. Vertical robot velocity/wind remains available through API-level parameters but does not need primary UI controls.

### 4.5 Measured launcher state versus estimator state

The UI will separate:

- **Measured launch state**
  - exit speed;
  - launch angle;
  - spin RPM.

- **Flywheel estimate**
  - flywheel diameter;
  - wheel RPM;
  - compression;
  - hood material;
  - estimated exit speed/spin.

The estimator remains convenience tooling. Applying an estimate requires an explicit UI action. It must not silently overwrite measured inputs when estimator controls change.

Labels and documentation will state that estimator values are heuristics.

## 5. Calibration subsystem

Create `src/calibration.js` for browser-safe model fitting/evaluation utilities and `api/calibration.py` for full fitting workflows.

The Python implementation is the authoritative calibration workflow because it is better suited to numerical fitting and offline datasets. JavaScript will support profile loading, validation metrics, and deterministic evaluation of already-fitted profiles.

### 5.1 Calibration shot schema

Calibration data is stored as JSON or CSV with SI-unit canonical fields.

Required per shot:

- unique shot id;
- launch position `x,y,z`;
- measured initial velocity vector or measured speed + direction;
- measured spin vector or backspin RPM for planar shots;
- timestamped observed projectile positions, or at minimum one or more measured checkpoints;
- ball mass;
- ball diameter.

Optional:

- robot velocity;
- wind;
- temperature;
- pressure/air density;
- ball identifier;
- wear state;
- observed HUB result;
- notes.

A compact planar CSV format may be supported for teams that only record side-view trajectories.

### 5.2 Fitting strategy

The fitting process is staged to reduce parameter confounding.

Stage A — drag calibration:

- use low-spin or zero-spin shots;
- fit `Cd(Re)`;
- start from a constant candidate and optionally fit a piecewise-linear table;
- regularize tables to avoid overfitting sparse data.

Stage B — lift calibration:

- hold the selected drag model fixed;
- use measured spinning shots;
- fit `Cl(S)` or `Cl(Re,S)`.

Stage C — optional spin decay:

- only fit spin-decay time constant if spin-versus-time measurements exist;
- otherwise keep spin decay disabled rather than infer it from positional residuals alone.

The optimizer will minimize weighted position residuals. When timestamp uncertainty is present, checkpoint weights can be supplied.

### 5.3 Training and validation

Calibration must support deterministic train/validation splitting by shot id and seed.

Reported metrics:

- RMS 3-D position error across observations;
- RMS vertical error;
- RMS downrange error;
- HUB-plane crossing-position error where applicable;
- entry-angle error where measured;
- predicted versus observed clean-entry confusion matrix where labels exist;
- calibration-domain ranges for `Re` and `S`;
- fraction of validation samples requiring coefficient clamping.

A profile is not declared "accurate" solely because training residuals are small.

### 5.4 Calibration profile format

Profiles are versioned JSON.

Conceptual shape:

```json
{
  "schema": "frc-projectile-calibration-v1",
  "name": "Team 0000 FUEL profile",
  "createdAt": "...",
  "gamePiece": {
    "diameter": 0.15,
    "massReference": 0.215
  },
  "environment": {
    "dynamicViscosity": 0.0000181
  },
  "dragModel": {...},
  "liftModel": {...},
  "spinDecayTimeConstant": null,
  "domain": {
    "reynolds": [70000, 180000],
    "spinParameter": [0.05, 0.7]
  },
  "validation": {...}
}
```

Profiles are data, not executable code.

## 6. Uncertainty and Monte Carlo

Create `src/uncertainty.js` and Python parity helpers as needed.

Supported uncertain variables:

- launch speed;
- elevation angle;
- azimuth angle where applicable;
- spin magnitude;
- ball mass;
- drag multiplier or coefficient perturbation;
- lift multiplier or coefficient perturbation;
- robot velocity;
- optional wind.

The first version will use independent distributions. Correlated uncertainty is explicitly deferred.

Distribution kinds:

- fixed;
- normal with optional truncation;
- uniform.

A seeded PRNG is required so tests and optimization runs are reproducible.

### 6.1 Robust shot metrics

For a candidate launch condition, Monte Carlo output includes:

- clean-entry probability;
- rim-collision probability;
- funnel-collision probability;
- miss probability;
- minimum/median/percentile clearance margin;
- distribution of entry velocity;
- distribution of entry angle.

The UI will clearly label the result as a simulation estimate based on the specified uncertainty model.

## 7. Robust optimization

Extend `src/optimizer.js`.

Deterministic optimization remains available and remains the default for speed.

Add robust optimization mode:

1. coarse deterministic search identifies promising candidate regions;
2. top candidates receive Monte Carlo evaluation;
3. ranking prioritizes:
   - highest clean-entry probability;
   - then stronger low-percentile clearance;
   - then lower required velocity / closeness to the user's reference according to the chosen optimizer mode;
4. winning candidates are re-evaluated with the configured full Monte Carlo sample count and `dt=0.001 s`.

This prevents evaluating thousands of candidates with expensive Monte Carlo unnecessarily.

The worker architecture remains in place so robust optimization does not block the React main thread.

## 8. UI changes

The default UI should stay approachable.

Add an expandable **Advanced Physics** area containing:

### Aerodynamics

- model source: baseline / calibration profile;
- current `Cd` and `Cl` model summary;
- calibrated `Re` and `S` domain;
- warning badge when current flight spends meaningful time outside the calibrated domain.

### Motion and environment

- robot forward velocity;
- robot lateral velocity;
- wind forward;
- wind lateral.

### Calibration profile

- load profile JSON;
- profile name;
- validation RMS;
- profile-domain summary;
- clear/revert-to-baseline action.

No arbitrary JavaScript expressions are accepted.

### Uncertainty

- enable robust analysis;
- standard deviation/range controls for launch speed, angle, spin, mass, drag, and lift;
- sample count with a bounded range;
- deterministic seed.

Display robust results as probabilities and percentile clearance rather than pretending there is one exact trajectory.

## 9. Error handling

Invalid model definitions fail fast with descriptive errors.

Examples:

- unsorted or duplicate interpolation axes;
- non-finite coefficients;
- negative Reynolds values;
- malformed table dimensions;
- invalid probability/distribution parameters;
- unknown profile schema;
- invalid mass/radius;
- Monte Carlo sample counts outside supported bounds.

Calibration profiles outside their measured domain do not crash. Values clamp to boundary coefficients and produce extrapolation/clamping diagnostics.

## 10. Testing strategy

Development follows TDD.

### Aerodynamics unit tests

- Reynolds number calculation;
- spin parameter using only perpendicular spin;
- 1-D interpolation;
- 2-D bilinear interpolation;
- boundary clamping;
- model validation;
- exact legacy lift-model compatibility;
- exact constant-drag compatibility.

### Engine tests

- existing golden fixtures remain valid in compatibility mode;
- table models affect force magnitude as expected;
- robot field velocity changes launch state exactly once;
- matching wind still removes aerodynamic force;
- JavaScript/Python parity for representative calibrated models;
- RK4/RK45 convergence remains intact.

### Calibration tests

Use synthetic trajectories generated from known coefficient models.

Verify:

- constant `Cd` recovery within tolerance;
- simple `Cd(Re)` recovery on adequate synthetic data;
- `Cl(S)` recovery after drag is fixed;
- held-out validation metrics;
- deterministic train/validation split;
- refusal or warning when data are insufficient for a requested model complexity.

### Uncertainty tests

- seeded PRNG determinism;
- fixed distributions reproduce deterministic simulation;
- probability totals sum to 1;
- increasing launch variance broadens outcome distributions;
- stable percentile calculations.

### Optimizer tests

- robust ranking can prefer a slightly lower-clearance nominal shot with higher clean-entry probability;
- deterministic mode preserves current behavior;
- final robust candidate is revalidated using full settings.

### UI contract tests

- estimator does not overwrite measured values without explicit apply action;
- baseline mode still works with no profile;
- profile validation errors are visible;
- moving-robot controls reach simulation parameters.

## 11. Documentation

Update the README and add a dedicated calibration guide.

The guide will cover:

1. measuring exit speed with high-frame-rate video;
2. measuring spin using marked balls;
3. camera placement and scale calibration;
4. capturing multiple speeds and spin rates;
5. separating low-spin drag shots from spinning lift shots;
6. fitting a calibration profile;
7. validating on held-out shots;
8. interpreting RMS error and score-rate metrics;
9. not extrapolating beyond measured `Re/S` ranges;
10. repeating calibration for materially different ball wear or launcher configuration.

The documentation will explicitly distinguish:

- numerical solver accuracy;
- model-form uncertainty;
- fitted parameter uncertainty;
- launch repeatability;
- collision-model simplifications.

## 12. Migration and compatibility

No existing public scalar-coefficient call site should need to change.

The current browser default remains the existing uncalibrated FUEL baseline unless the user loads a calibration profile.

Existing tests that encode current behavior remain useful compatibility tests.

New advanced model objects are additive.

Profile schema versioning enables future model additions without silently reinterpreting old profiles.

## 13. Performance

A single deterministic shot should remain effectively as responsive as today.

Interpolation cost is negligible relative to integration.

Monte Carlo is bounded and worker-based.

Robust optimization uses two-stage evaluation to control total simulations.

Default sample counts should favor interactive latency; users can opt into larger validation runs.

## 14. Security and data handling

Calibration profiles and shot datasets are treated as local user data.

Profile import parses JSON only and does not execute code.

The browser does not require remote upload for simulation.

Any future server-side persistence is outside this project scope.

## 15. Implementation sequence

The implementation plan should break work into testable increments in this order:

1. aerodynamic model primitives and parity;
2. flight-engine integration with legacy compatibility;
3. robot velocity and wind plumbing;
4. calibration profile schema and loading;
5. Python calibration fitter and validation metrics;
6. uncertainty engine;
7. robust optimizer;
8. UI integration;
9. documentation and end-to-end verification.

Each increment must keep the repository testable and avoid mixing unrelated refactors.
