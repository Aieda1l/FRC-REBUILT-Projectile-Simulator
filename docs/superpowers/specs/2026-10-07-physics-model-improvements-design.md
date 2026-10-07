# Physics Model Improvements Without FUEL Measurements — Design

Date: 2026-10-07  
Status: Approved design, awaiting written-spec review  
Branch: `physics-model-improvements`

## 1. Purpose

Improve the FRC REBUILT projectile simulator using physics that can be implemented defensibly without access to a 2026 FUEL, a shooter, or new experimental data.

The changes in this milestone target model-form limitations identified during review of the current calibrated projectile engine:

1. include aerodynamic buoyancy for a large, light foam ball;
2. make the drag-model interface capable of representing spin-dependent drag;
3. allow signed lift coefficients so calibrated models can represent reverse Magnus behavior if measurements ever show it;
4. prevent spinning shots from contaminating the initial drag calibration stage.

The milestone must not invent new FUEL-specific aerodynamic constants. Existing scalar defaults remain the same unless a force follows directly from known geometry and atmospheric state.

## 2. Success criteria

The work is successful when:

1. JavaScript and Python flight engines include buoyancy consistently.
2. Existing `constant` and `table1d` drag models remain valid and retain their current meaning.
3. A new `table2d` drag model can represent `Cd(Re, S)` through bilinear interpolation and the same boundary-clamping behavior used elsewhere.
4. Tabulated lift models can contain negative coefficients and the force solver preserves their sign.
5. The legacy `legacy-spin-cap` lift model remains unchanged and nonnegative.
6. Calibration CLI drag fitting uses only shots inside an explicit low-spin domain.
7. Calibration fails clearly when insufficient low-spin data exist instead of silently fitting drag against Magnus-affected shots.
8. JavaScript and Python remain behaviorally consistent for all new model cases.
9. Existing calibration profile JSON remains loadable without schema migration.
10. No new FUEL-specific `Cd`, `Cl`, spin-decay, deformation, rebound, or turbulence constants are introduced.

## 3. Non-goals

This milestone will not:

- guess a FUEL-specific `Cd(Re)`, `Cd(Re,S)`, or `Cl(Re,S)` curve;
- add a guessed spin-decay constant;
- model foam deformation, rim bounce, funnel rebound, or panel friction;
- add CFD, boundary-layer, seam-orientation, or turbulence-transition solvers;
- infer launcher exit speed or ball spin from motor RPM more accurately;
- add automatic fitting of a 2-D spin-dependent drag surface from trajectory data;
- change the RK4 or Dormand-Prince RK45 numerical methods;
- change HUB geometry or scoring policy;
- redesign the UI;
- change the calibration schema identifier solely for these additions.

The new 2-D drag representation is infrastructure for future measured data. It is not permission to populate a speculative table.

## 4. Current system

The canonical engines are `src/physics3d.js` and `api/physics3d.py`. They integrate a 9-state vector containing position, translational velocity, and angular velocity.

The aerodynamic layer in `src/aerodynamics.js` and `api/aerodynamics.py` currently supports:

- Reynolds number `Re = rho v D / mu`;
- perpendicular spin parameter `S = omega_perp r / v`;
- drag models:
  - `constant`;
  - `table1d` over `Re`;
- lift models:
  - `legacy-spin-cap`;
  - `table1d` over `S`;
  - `table2d` over `(Re,S)`.

The force solver currently applies gravity, quadratic drag, Magnus lift, and optional exponential spin decay. It does not include buoyancy. Lift-table validation currently rejects negative coefficients, and the Magnus force path skips any coefficient that is not positive.

The calibration CLI currently fits drag from the complete training set while disabling Magnus during that drag fit. That allows spinning trajectories to push the fitted drag coefficient away from the true drag behavior.

## 5. Buoyancy

### 5.1 Physical model

For a spherical game piece of radius `r`, volume is

[
V_b = \frac{4}{3}\pi r^3.
]

The upward buoyancy force is

[
F_b = \rho_{air} V_b g.
]

The acceleration contribution is

[
a_b = \frac{\rho_{air} V_b g}{m}.
]

The vertical acceleration therefore becomes

[
a_z = -g + a_b + a_{drag,z} + a_{magnus,z}.
]

For the nominal current defaults (`r = 0.075 m`, `m = 0.215 kg`, `rho = 1.204 kg/m^3`), this reduces effective downward acceleration by roughly one percent. The force depends only on quantities already present in the model.

### 5.2 Flight parameter

Add:

- JavaScript: `enableBuoyancy: true`;
- Python: `enable_buoyancy: bool = True`.

Buoyancy is enabled by default because it is a deterministic consequence of the modeled ball volume, air density, and gravity rather than a FUEL-specific empirical coefficient.

When disabled, the engine reproduces the previous gravity behavior exactly.

Buoyancy is zero when either air density or gravity is zero.

### 5.3 Compatibility implications

Enabling buoyancy by default intentionally changes aerodynamic trajectories, including trajectories using existing calibration profiles. The calibration JSON schema does not need to change because buoyancy is an engine-level force derived from existing profile/environment values rather than a new fitted field.

Any previously measured calibration profile should be revalidated after this change. A profile fitted before buoyancy existed may have partially absorbed the missing upward force into its fitted drag or lift coefficients.

No measured FUEL profile is bundled with the repository, so this change does not invalidate a checked-in production coefficient set.

Gravity-only numerical tests that are intended to verify the analytic no-air solution must explicitly set `enableBuoyancy: false` / `enable_buoyancy=False`, or set air density to zero. Tests intended to verify realistic atmospheric flight should leave buoyancy enabled.

## 6. Spin-aware drag model

### 6.1 Representation

Extend normalized drag models with:

```json
{
  "kind": "table2d",
  "reynolds": [50000, 100000, 200000],
  "spinParameters": [0.0, 0.25, 0.5],
  "coefficients": [
    [0.50, 0.49, 0.48],
    [0.47, 0.46, 0.45],
    [0.40, 0.39, 0.38]
  ]
}
```

The coefficient rows correspond to Reynolds-number entries and columns correspond to spin-parameter entries, matching the existing lift `table2d` orientation.

This model remains optional. The default drag model is still the current scalar `Cd = 0.47`.

### 6.2 Evaluation API

Change drag evaluation conceptually from:

```text
evaluateDragModel(model, reynolds)
```

to:

```text
evaluateDragModel(model, reynolds, spinParameter = 0)
```

and the equivalent Python function.

Keeping a default spin value preserves direct callers that currently pass only Reynolds number.

Behavior by model kind:

- `constant`: ignores `Re` and `S`;
- `table1d`: interpolates only over `Re`, ignoring `S`;
- `table2d`: bilinearly interpolates over `(Re,S)`.

Out-of-domain queries clamp independently on both axes and return `clamped: true` when either axis was clamped.

### 6.3 Force pipeline

The aerodynamic-state calculation already computes both Reynolds number and perpendicular spin parameter before force application. It will pass both values into drag evaluation.

At zero relative airflow:

- `Re = 0`;
- `S = 0`;
- aerodynamic acceleration remains zero;
- diagnostics remain finite;
- table boundary-clamping diagnostics continue to reflect model-domain behavior.

### 6.4 Calibration profiles

The existing calibration profile schema accepts a normalized drag-model object. Its parser will be extended to accept `table2d` drag models without changing the top-level schema string.

Existing `constant` and `table1d` profiles remain valid.

The calibration domain continues to report overall Reynolds and spin-parameter ranges. No new profile field is required to load a 2-D drag table.

## 7. Signed lift coefficients

### 7.1 Model validation

The aerodynamic layer currently uses one nonnegative coefficient validator for both drag and lift. Split the concepts:

- drag coefficients must remain finite and nonnegative;
- tabulated lift coefficients may be any finite signed value;
- `legacy-spin-cap.maxCoefficient` remains finite and nonnegative.

This keeps the legacy model behavior unchanged while allowing measured tables to express force reversal.

### 7.2 Force application

Current logic applies Magnus acceleration only when `liftCoefficient > 0`. Replace that sign gate with a nonzero-magnitude check.

The unit Magnus direction remains

[
\hat m = \frac{\omega_\perp \times \hat u}{|\omega_\perp \times \hat u|}.
]

Acceleration becomes

[
\mathbf a_M =
\frac{1}{m}
\left(\frac12 \rho A v^2\right)
C_L
\hat m.
]

A negative `Cl` therefore reverses the force direction naturally.

No change is made to the spin vector convention. Positive UI backspin continues to map to the current negative-y spin axis.

### 7.3 Calibration fitting bounds

The lift fitter must be able to recover a signed table if future data require it. Change fitted table-coefficient bounds from nonnegative-only to symmetric signed bounds.

Use `[-3, 3]` as a numerical safety bound, matching the existing drag fitter's order-of-magnitude protection while permitting either sign. This is an optimizer guardrail, not a claim that FUEL physically reaches those coefficients.

Initial guesses remain positive so ordinary backspin datasets continue to start from the expected branch.

## 8. Low-spin drag calibration

### 8.1 Problem

The current CLI performs:

1. random train/validation split;
2. drag fit using all training shots with Magnus disabled;
3. lift fit using spinning training shots.

This is internally inconsistent. If a training shot has measurable spin, its observed curvature contains lift information. Turning Magnus off and then fitting `Cd` can make drag absorb part of the missing lift.

### 8.2 Shot partition helper

Add a shared calibration helper that computes each shot's initial `Re` and `S` using the same air-relative launch state already used for domain calculations.

Conceptual interface:

```python
partition_shots_by_spin_parameter(
    shots,
    base_params,
    max_drag_spin_parameter,
) -> (drag_shots, spinning_shots)
```

Classification:

- drag shot: `S <= max_drag_spin_parameter`;
- spinning/lift shot: `S > max_drag_spin_parameter`.

The comparison is inclusive on the drag boundary to make a configured threshold deterministic.

### 8.3 CLI control

Add:

```text
--drag-max-spin-parameter
```

with default `0.05`.

This value is a calibration-selection threshold, not a FUEL aerodynamic coefficient. It may be changed by the user when their measurement setup has a different definition of "low spin."

The CLI must print or record enough information to make the selection auditable: number of training shots used for drag, number reserved as spinning shots, and the configured threshold.

### 8.4 Failure behavior

Do not silently fall back to all shots when low-spin data are insufficient.

- Constant drag fitting requires at least one selected drag shot.
- `table1d` drag fitting retains its existing requirement for at least three distinct Reynolds conditions among selected drag shots.
- Lift fitting retains its current minimum-data requirements, applied to the selected spinning subset.

When drag data are insufficient, the CLI exits with a clear message that names the threshold and suggests either collecting low-spin data or explicitly raising the threshold.

When lift data are insufficient, the existing zero-lift fallback may remain for compatibility only when there are fewer than two spinning training shots. The generated profile must not pretend a measured lift model was fitted in that case.

### 8.5 No automatic `Cd(Re,S)` fitting yet

The engine and profile parser will support `table2d` drag, but `scripts/calibrate_fuel.py` will continue to offer only:

- `constant`;
- `table1d`.

Automatic 2-D drag fitting is intentionally deferred. Fitting spin-dependent drag and Magnus lift simultaneously from trajectory data introduces parameter identifiability problems and deserves a dedicated experimental design rather than an automatic extension of the current staged fitter.

## 9. Files and interfaces

Expected implementation touch points:

### JavaScript

- `src/aerodynamics.js`
  - signed lift-table validation;
  - `table2d` drag normalization and evaluation;
  - drag evaluation accepts spin parameter.
- `src/physics3d.js`
  - `enableBuoyancy` default and validation;
  - buoyancy acceleration;
  - pass `S` into drag evaluation;
  - signed Magnus application.
- `src/calibration.js`
  - accept `table2d` drag through existing normalization path;
  - preserve signed lift tables.
- `src/uncertainty.js`
  - scaling helper supports `table2d` drag;
  - scaling signed lift coefficients preserves sign.
- browser/worker parameter forwarding files as needed
  - preserve `enableBuoyancy` when constructing flight parameters.

### Python

- `api/aerodynamics.py`
  - parity with JavaScript aerodynamic model changes.
- `api/physics3d.py`
  - `enable_buoyancy`;
  - buoyancy acceleration;
  - signed Magnus;
  - spin-aware drag evaluation.
- `api/main.py`
  - expose/forward buoyancy option for 3-D API requests.
- `api/trajectory_simulator.py`
  - preserve buoyancy through the legacy adapter where appropriate.
- `api/uncertainty.py`
  - preserve the new engine parameter.
- `calibration/fitting.py`
  - signed lift fitting;
  - low-spin partition helper.
- `scripts/calibrate_fuel.py`
  - threshold argument;
  - drag/lift subset selection;
  - clear insufficient-data errors and selection reporting.

### Documentation

- `README.md`
  - list buoyancy;
  - document `Cd(Re,S)` support;
  - clarify that no measured 2-D drag surface is bundled.
- `docs/calibration-guide.md`
  - describe low-spin selection threshold;
  - state that pre-change calibration profiles should be revalidated after buoyancy is introduced.

## 10. Testing strategy

Implementation follows test-driven development. Tests are added or changed before production behavior.

### 10.1 Buoyancy tests

JavaScript and Python:

1. With drag and Magnus disabled, atmospheric air density, nonzero gravity, and buoyancy enabled, vertical acceleration equals:
   [
   -g + \rho (4\pi r^3/3)g/m.
   ]
2. Disabling buoyancy restores exactly `-g`.
3. Zero air density produces no buoyancy.
4. Zero gravity produces no buoyancy.
5. JS/Python golden fixtures remain consistent.

### 10.2 Drag-model tests

JavaScript and Python:

1. Existing constant drag returns the same result when `S` is omitted or supplied.
2. Existing `table1d` drag returns the same result for different `S`.
3. `table2d` drag interpolates correctly at:
   - exact grid points;
   - Reynolds midpoint;
   - spin midpoint;
   - bilinear interior point.
4. Either-axis out-of-domain queries clamp to the nearest boundary and flag `clamped`.
5. Negative drag coefficients remain invalid.

### 10.3 Signed-lift tests

JavaScript and Python:

1. Positive `Cl` preserves the existing Magnus direction.
2. Negative `Cl` produces equal-magnitude opposite-direction Magnus acceleration for the same state.
3. Zero `Cl` produces no Magnus acceleration.
4. Signed values survive table normalization/interpolation.
5. The legacy spin-cap model still rejects a negative maximum coefficient.

### 10.4 Calibration tests

Python:

1. Shot partitioning uses dimensionless spin parameter rather than raw RPM/rad/s.
2. A shot exactly on the threshold belongs to the drag subset.
3. High-spin shots are excluded from drag fitting.
4. CLI drag fitting fails when no training shot is below the threshold.
5. `table1d` drag fitting still enforces distinct-Reynolds requirements after filtering.
6. Lift fitting accepts signed coefficients within the new bounds.
7. Existing profile parsing remains backward-compatible.

### 10.5 Integration and regression

Run the complete existing suites plus build:

```bash
python -m pytest
npm test
npm run build
```

Any existing fixture whose expected trajectory changes only because buoyancy is now intentionally enabled must be regenerated from the Python canonical implementation and reviewed rather than hand-edited.

## 11. Backward compatibility

The following stay compatible:

- calibration schema string `frc-projectile-calibration-v1`;
- `constant` drag profiles;
- `table1d` drag profiles;
- legacy nonnegative `legacy-spin-cap` lift profiles;
- existing positive lift tables;
- callers of drag evaluation that omit spin parameter;
- RK4/RK45 method names and semantics;
- state-vector ordering;
- coordinate system;
- UI backspin sign convention;
- HUB geometry/classification behavior.

The intentional behavior change is atmospheric buoyancy being enabled by default. This affects trajectories even when drag and Magnus are disabled unless buoyancy is also disabled. Tests and documentation must make this explicit.

## 12. Error handling and validation

Reject:

- non-finite model axes or coefficients;
- non-monotonic `Re` or `S` axes;
- malformed table shapes;
- negative drag coefficients;
- negative legacy spin-cap maximum coefficient;
- invalid buoyancy toggle types where runtime validation already enforces booleans;
- negative calibration spin thresholds.

Allow:

- signed finite lift-table coefficients;
- zero lift coefficients;
- zero drag coefficients;
- 2-D drag tables with any nonnegative finite coefficients.

No imported calibration profile executes code.

## 13. Deferred work requiring measurements

These remain explicitly deferred until FUEL/shooter access is available:

- replacing `Cd = 0.47` with measured FUEL data;
- populating `Cd(Re,S)`;
- replacing the legacy lift curve with measured `Cl(Re,S)`;
- determining whether reverse Magnus occurs for FUEL in the relevant Reynolds range;
- measuring spin decay;
- validating launcher exit-speed/spin heuristics;
- modeling wear-state effects;
- fitting deformation/rebound behavior;
- establishing held-out HUB-plane RMS and scoring-classification accuracy.

## 14. Acceptance criteria

The milestone is ready for implementation review when all of the following are true:

- [ ] buoyancy has JS/Python parity and is on by default;
- [ ] buoyancy can be disabled for analytic/regression cases;
- [ ] `table2d` drag parses, interpolates, clamps, and reports diagnostics in both languages;
- [ ] old drag models are unchanged;
- [ ] signed lift tables parse and produce signed Magnus forces;
- [ ] legacy lift behavior is unchanged;
- [ ] calibration drag fitting excludes shots above the configured spin-parameter threshold;
- [ ] insufficient low-spin calibration data produces a clear failure;
- [ ] automatic 2-D drag fitting has not been added;
- [ ] no speculative FUEL coefficient data have been introduced;
- [ ] documentation explains buoyancy compatibility implications;
- [ ] Python tests pass;
- [ ] JavaScript tests pass;
- [ ] production build succeeds.
