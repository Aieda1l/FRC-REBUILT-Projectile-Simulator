# Calibrated Projectile Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add calibrated Reynolds/spin-dependent aerodynamics, moving-robot/wind support, measurement-driven calibration, uncertainty simulation, and robust shot optimization without breaking the current FRC simulator defaults.

**Architecture:** Keep `physics3d` as the canonical ODE/integration core and introduce pure aerodynamic-model evaluators that both JavaScript and Python consume. Layer calibration profiles, Monte Carlo uncertainty, and robust optimization above the deterministic simulator; keep the current scalar coefficients and deterministic optimizer as compatibility/default paths.

**Tech Stack:** React 19, JavaScript ES modules, Node built-in test runner, Python 3.9+, NumPy, SciPy `least_squares`, FastAPI/Pydantic, Vite.

**Spec:** `docs/superpowers/specs/2026-10-07-calibrated-projectile-physics-design.md`

## Global Constraints

- Existing scalar `dragCoeff` / `liftCoeff` and `drag_coefficient` / `lift_coefficient` call sites must remain valid.
- The baseline browser behavior remains the current uncalibrated FUEL model unless a calibration profile is explicitly loaded.
- Legacy Magnus behavior must remain exactly reproducible as `legacy-spin-cap` with `S_sat = 0.5`.
- Calibration profiles are JSON data only; no executable expressions or code evaluation.
- HUB scoring remains the current conservative rigid-sphere clean-entry model; do not add speculative bounce/deformation physics.
- Moving-robot muzzle velocity is shooter-relative; robot field velocity is added exactly once at launch; wind is applied only through air-relative velocity.
- Spin decay remains disabled unless a positive calibrated time constant is explicitly supplied.
- Robust/Monte Carlo analysis must be reproducible from a deterministic seed and must run through the worker path when invoked from the UI.
- Calibrated coefficient tables clamp at their measured domain boundaries and expose clamping diagnostics; they must not silently extrapolate.
- Do not replace the FUEL baseline with baseball, soccer-ball, or other unrelated empirical coefficients.

## Review Focus

- **Degenerate calibration tables:** one-point, unsorted, duplicate-axis, or mismatched 2-D tables must fail with descriptive validation errors; pinned in Task 1 tests.
- **Zero/near-zero airflow:** Reynolds/spin calculations must not divide by zero or emit NaN; pinned in Task 1 and Task 2 tests.
- **Out-of-domain calibrated flights:** coefficients clamp deterministically and diagnostics report the clamp rather than extrapolating; pinned in Task 1 and Task 4 tests.
- **Monte Carlo pathological inputs:** zero variance must reproduce the deterministic shot exactly; invalid sample counts/distributions must fail fast; pinned in Task 6 tests.
- **Profile/legacy coexistence:** clearing or rejecting a profile must return the UI and engine to byte-for-byte legacy model semantics, not leave stale advanced parameters; pinned in Task 4 and Task 8 tests.

---

### Task 1: Aerodynamic Model Primitives and JS/Python Parity

**Files:**
- Create: `src/aerodynamics.js`
- Create: `api/aerodynamics.py`
- Create: `tests/aerodynamics_js.test.mjs`
- Create: `tests/test_aerodynamics.py`

**Interfaces:**
- Consumes: plain JSON-like model objects from the approved spec.
- Produces JS:
  - `DEFAULT_DYNAMIC_VISCOSITY = 1.81e-5`
  - `reynoldsNumber({airDensity, speed, diameter, dynamicViscosity}) -> number`
  - `spinParameter({radius, perpendicularSpin, speed}) -> number`
  - `normalizeDragModel(model, fallbackCoefficient) -> normalized model`
  - `normalizeLiftModel(model, fallbackCoefficient) -> normalized model`
  - `evaluateDragModel(model, reynolds) -> {coefficient, clamped}`
  - `evaluateLiftModel(model, reynolds, spinParameterValue) -> {coefficient, clamped}`
- Produces Python snake_case equivalents with identical model schemas and numeric semantics.

  Exact model JSON shapes:
  - Drag constant: `{kind: "constant", coefficient: number}`
  - Drag table: `{kind: "table1d", reynolds: number[], coefficients: number[]}`
  - Legacy lift: `{kind: "legacy-spin-cap", maxCoefficient: number, saturationSpin?: number}`
  - Lift 1-D table: `{kind: "table1d", spinParameters: number[], coefficients: number[]}`
  - Lift 2-D table: `{kind: "table2d", reynolds: number[], spinParameters: number[], coefficients: number[][]}`, where rows follow `reynolds` and columns follow `spinParameters`.

- [ ] **Step 1: Write failing parity/validation tests**
  - Assert `Re = rho * speed * diameter / mu`.
  - Assert spin parameter is zero for `speed <= 1e-12`.
  - Assert `constant`, `table1d`, `legacy-spin-cap`, and `table2d` values at interior points.
  - Assert 1-D and 2-D boundary clamping sets `clamped: true`.
  - Assert unsorted/duplicate axes and malformed 2-D matrix shapes throw `RangeError` in JS / `ValueError` in Python.
  - Assert a one-point table is rejected rather than treated as an interpolator.

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
node --test tests/aerodynamics_js.test.mjs
python -m unittest tests.test_aerodynamics -v
```

Expected: FAIL because the aerodynamic modules do not exist.

- [ ] **Step 3: Implement the aerodynamic primitives**
  - Use piecewise-linear interpolation for `table1d`.
  - Use bilinear interpolation for `table2d`.
  - Clamp query axes to the first/last calibrated coordinate and set `clamped`.
  - `legacy-spin-cap` computes `coefficient = maxCl * min(S / saturationSpin, 1)`, with default `saturationSpin = 0.5`.
  - Require all coefficients and axes to be finite; require non-negative Reynolds/spin axes and coefficients.

- [ ] **Step 4: Run focused tests to verify pass**

Run the two commands from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aerodynamics.js api/aerodynamics.py tests/aerodynamics_js.test.mjs tests/test_aerodynamics.py
git commit -m "feat: add calibrated aerodynamic model primitives"
```

---

### Task 2: Integrate Calibrated Aerodynamics Into the Canonical Flight Engines

**Files:**
- Modify: `src/physics3d.js`
- Modify: `api/physics3d.py`
- Modify: `tests/physics3d_js.test.mjs`
- Modify: `tests/test_physics3d.py`
- Modify: `tests/fixtures/physics3d_golden.json` only if a new fixture schema case is required; do not rewrite existing legacy expected values.

**Interfaces:**
- Consumes: Task 1 model evaluators.
- Produces:
  - JS flight params accept `dynamicViscosity`, `dragModel`, `liftModel` in addition to legacy scalar coefficients.
  - Python `FlightParameters` accepts `dynamic_viscosity: float`, `drag_model: Optional[dict]`, `lift_model: Optional[dict]`.
  - JS `aerodynamicDiagnostics(state, params) -> {reynolds, spinParameter, dragCoefficient, liftCoefficient, dragClamped, liftClamped}`.
  - Python `aerodynamic_diagnostics(state, params) -> dict` with equivalent fields.

- [ ] **Step 1: Write failing engine compatibility tests**
  - Existing legacy derivative/golden cases must remain numerically unchanged within existing tolerances.
  - A `table1d` drag model must change drag magnitude according to current Reynolds number.
  - A `table2d` lift model must change Magnus magnitude according to both Reynolds and spin parameter.
  - At zero relative airflow diagnostics return finite zeros and no aerodynamic acceleration.
  - Explicit model objects take precedence over scalar fallback coefficients.

- [ ] **Step 2: Run focused tests to verify failure**

Run:
```bash
node --test tests/physics3d_js.test.mjs
python -m unittest tests.test_physics3d -v
```

Expected: new model tests FAIL while legacy tests remain green.

- [ ] **Step 3: Extend flight parameter normalization and derivative evaluation**
  - Add `dynamicViscosity` / `dynamic_viscosity` defaulting to Task 1's constant.
  - Normalize legacy scalars into model objects only when an explicit advanced model is absent.
  - Evaluate coefficients on every derivative call from air-relative speed and perpendicular spin.
  - Keep drag/Magnus vector direction code unchanged.

- [ ] **Step 4: Run focused tests and cross-language fixture checks**

Run the commands from Step 2 plus:
```bash
npm run test:physics
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physics3d.js api/physics3d.py tests/physics3d_js.test.mjs tests/test_physics3d.py tests/fixtures/physics3d_golden.json
git commit -m "feat: evaluate calibrated aerodynamics in flight engine"
```

---

### Task 3: Plumb Robot Velocity and Wind Through Shot Simulation and APIs

**Files:**
- Modify: `src/trajectory2d.js`
- Modify: `api/main.py`
- Modify: `tests/physics3d_js.test.mjs`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: existing `launchState(position, muzzleVelocity, spin, robotVelocity)` and `FlightParameters.wind`.
- Produces:
  - `simulateShot(params, options)` accepts `robotVelocity = [0,0,0]` and `wind = [0,0,0]`.
  - `Sim3DRequest` accepts `dynamic_viscosity`, `drag_model`, `lift_model` while preserving scalar fields.
  - Existing `/api/simulate3d` request semantics remain backward compatible.

- [ ] **Step 1: Write failing moving-shot/API tests**
  - A +2 m/s downrange robot velocity increases initial field velocity by exactly +2 m/s, with no second addition later.
  - Lateral robot velocity produces lateral flight displacement.
  - Wind is passed into the engine and matching field velocity still eliminates aerodynamic force.
  - The API accepts a calibrated drag/lift model payload and rejects malformed models with HTTP 422 or a descriptive 400.

- [ ] **Step 2: Run focused tests to verify failure**

Run:
```bash
node --test tests/physics3d_js.test.mjs
python -m unittest tests.test_api -v
```

- [ ] **Step 3: Implement shot/API plumbing**
  - Pass `robotVelocity` to `launchState`.
  - Pass `wind` and advanced aerodynamic fields into flight parameters.
  - Add Pydantic fields using JSON-compatible dictionaries; validate model structure by constructing `FlightParameters` and convert model validation failures to client errors.

- [ ] **Step 4: Run focused tests to verify pass**

Run the commands from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory2d.js api/main.py tests/physics3d_js.test.mjs tests/test_api.py
git commit -m "feat: expose moving robot and wind shot physics"
```

---

### Task 4: Calibration Profile Schema, Loading, and Domain Diagnostics

**Files:**
- Create: `src/calibration.js`
- Create: `api/calibration.py`
- Create: `tests/calibration_js.test.mjs`
- Create: `tests/test_calibration.py`
- Modify: `src/trajectory2d.js`

**Interfaces:**
- Consumes: Task 1 model schemas and Task 2 diagnostics.
- Produces:
  - JS `parseCalibrationProfile(value) -> normalizedProfile`
  - JS `applyCalibrationProfile(baseParams, profile) -> params`
  - JS `summarizeCalibrationDomain(samples, flightParams, profile) -> {sampleCount, clampedSamples, clampedFraction, reynoldsRange, spinParameterRange}`
  - Python `parse_calibration_profile(value) -> dict`
  - Schema identifier exactly `frc-projectile-calibration-v1`.

- [ ] **Step 1: Write failing profile tests**
  - Accept a complete valid v1 profile.
  - Reject unknown schema, missing model definitions, non-finite domain values, and reversed domain bounds.
  - Applying a profile must override aerodynamic models/spin-decay settings but preserve launch geometry and unrelated shot parameters.
  - Clearing/reverting the profile must reproduce legacy parameter objects and legacy trajectory results.
  - A trajectory outside profile domain reports a positive `clampedFraction`.

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
node --test tests/calibration_js.test.mjs
python -m unittest tests.test_calibration -v
```

- [ ] **Step 3: Implement profile parsing/application and trajectory diagnostics**
  - Treat profiles as data only.
  - Normalize drag/lift models through Task 1 validators.
  - Compute diagnostics by sampling the already-produced trajectory; do not run a second flight integration.

- [ ] **Step 4: Run focused tests to verify pass**

Run the commands from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/calibration.js api/calibration.py src/trajectory2d.js tests/calibration_js.test.mjs tests/test_calibration.py
git commit -m "feat: add calibration profile schema and diagnostics"
```

---

### Task 5: Python Calibration Fitting and Held-Out Validation

**Files:**
- Modify: `api/calibration.py`
- Modify: `requirements.txt`
- Modify: `tests/test_calibration.py`
- Create: `scripts/calibrate_fuel.py`

**Interfaces:**
- Consumes: calibration-shot records and canonical Python `integrate_trajectory`.
- Produces:
  - `split_shots(shots, validation_fraction: float, seed: int) -> (train, validation)`
  - `fit_drag_model(shots, base_params, model_kind="constant"|"table1d") -> dict`
  - `fit_lift_model(shots, base_params, drag_model, model_kind="table1d"|"table2d") -> dict`
  - `fit_spin_decay(shots, base_params) -> Optional[float]`, requiring measured spin-over-time data.
  - `validate_profile(shots, profile, base_params) -> metrics dict`
  - CLI: `python scripts/calibrate_fuel.py INPUT.(json|csv) --output profile.json --validation-fraction 0.2 --seed 2026`.

- [ ] **Step 1: Add SciPy and write synthetic-data failing tests**
  - Generate synthetic low-spin trajectories from a known constant `Cd`; fit recovers it within a stated tolerance.
  - Generate spinning trajectories after fixing drag; `Cl(S)` fit reduces held-out RMS versus the zero-lift baseline.
  - Deterministic split returns the same shot IDs for the same seed.
  - Requesting spin decay without spin-versus-time observations returns `None` or a clear insufficiency result, never infers it from position alone.
  - Table complexity exceeding available independent data points raises a descriptive error.

- [ ] **Step 2: Run calibration tests to verify failure**

Run:
```bash
python -m unittest tests.test_calibration -v
```

Expected: FAIL because fitting functions/CLI do not exist.

- [ ] **Step 3: Implement fitting with `scipy.optimize.least_squares`**
  - Optimize bounded non-negative coefficient parameters.
  - Fit drag on low-spin data first, then lift with drag fixed.
  - Add smoothness residuals for multi-knot tables so sparse data cannot produce arbitrary oscillation.
  - Compute RMS 3-D, vertical, downrange, HUB-plane, clean-entry confusion counts, domain ranges, and clamp fraction when those observations are present.

- [ ] **Step 4: Implement the CLI as a thin adapter**
  - Parse JSON and documented planar CSV.
  - Write the versioned profile JSON and validation metrics.
  - Exit non-zero with a clear message for malformed/insufficient datasets.

- [ ] **Step 5: Run tests to verify pass**

Run:
```bash
python -m unittest tests.test_calibration -v
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/calibration.py requirements.txt tests/test_calibration.py scripts/calibrate_fuel.py
git commit -m "feat: fit and validate FUEL calibration profiles"
```

---

### Task 6: Seeded Uncertainty and Monte Carlo Shot Evaluation

**Files:**
- Create: `src/uncertainty.js`
- Create: `api/uncertainty.py`
- Create: `tests/uncertainty_js.test.mjs`
- Create: `tests/test_uncertainty.py`

**Interfaces:**
- Consumes: deterministic `simulateShot` parameters and distribution definitions.
- Produces JS:
  - `createSeededRng(seed: number) -> () => number`
  - `sampleDistribution(definition, rng) -> number`
  - `sampleShotParams(baseParams, uncertainty, rng) -> params`
  - `evaluateShotUncertainty(baseParams, uncertainty, {sampleCount, seed, dt}) -> robustResult`
- Produces Python equivalents with the same 32-bit PRNG algorithm and distribution semantics.
- `robustResult` contains counts/probabilities for all four HUB classifications plus clearance, entry-speed, and entry-angle percentile summaries.

- [ ] **Step 1: Write failing deterministic/validation tests**
  - Same seed yields the same sample sequence in JS and Python fixture cases.
  - `fixed`, `uniform`, and optionally truncated `normal` distributions validate and sample finite values.
  - Zero-variance/fixed uncertainty reproduces deterministic HUB classification and trajectory-derived entry metrics.
  - Classification probabilities sum to 1 within floating-point tolerance.
  - Invalid sample count, negative normal sigma, reversed uniform bounds, or non-finite values fail fast.

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
node --test tests/uncertainty_js.test.mjs
python -m unittest tests.test_uncertainty -v
```

- [ ] **Step 3: Implement seeded sampling and Monte Carlo aggregation**
  - Use one documented 32-bit PRNG algorithm in both languages.
  - Bound UI-facing sample counts in the evaluator to `1..10000`.
  - Perturb speed, elevation, spin magnitude, mass, drag/lift multipliers, robot velocity, and wind only when configured.
  - Percentiles use deterministic sorted-sample interpolation.

- [ ] **Step 4: Run focused tests to verify pass**

Run the commands from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uncertainty.js api/uncertainty.py tests/uncertainty_js.test.mjs tests/test_uncertainty.py
git commit -m "feat: add reproducible shot uncertainty simulation"
```

---

### Task 7: Robust Worker-Based Optimization

**Files:**
- Modify: `src/optimizer.js`
- Modify: `src/optimizer.worker.js`
- Modify: `src/optimizerClient.js`
- Modify: `tests/optimizer_js.test.mjs`
- Modify: `tests/optimizer_client_js.test.mjs`

**Interfaces:**
- Consumes: Task 6 `evaluateShotUncertainty`.
- Produces:
  - `rankRobustCandidate(a, b, reference) -> number`
  - `optimizeRobust(params, {mode, uncertainty, coarseSamples, finalSamples, seed}, callbacks) -> optimizationResult`
  - Worker mode `robust` carrying the deterministic mode as `options.mode = "angle"|"velocity"|"both"`.

- [ ] **Step 1: Write failing robust-ranking/worker tests**
  - Higher clean-entry probability outranks a nominally larger-clearance candidate.
  - Equal probability falls back to better 10th-percentile clearance.
  - Equal robust metrics fall back to the existing reference-distance rule.
  - Robust optimizer first screens deterministic candidates, Monte Carlo-evaluates only the retained shortlist, and re-evaluates the winner at `dt=0.001` using `finalSamples`.
  - Worker client routes progress/complete/error for robust mode and cancellation still terminates the active worker.

- [ ] **Step 2: Run focused tests to verify failure**

Run:
```bash
node --test tests/optimizer_js.test.mjs tests/optimizer_client_js.test.mjs
```

- [ ] **Step 3: Implement two-stage robust optimization**
  - Reuse existing coarse/refined deterministic searches to generate the shortlist.
  - Default `coarseSamples = 64`, `finalSamples = 512` when not provided.
  - Ranking order: clean-entry probability descending, 10th-percentile clearance descending, then existing deterministic tie-break.
  - Preserve existing `optimizeAngle`, `optimizeVelocity`, and `optimizeBoth` behavior unchanged.

- [ ] **Step 4: Run focused tests to verify pass**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.js src/optimizer.worker.js src/optimizerClient.js tests/optimizer_js.test.mjs tests/optimizer_client_js.test.mjs
git commit -m "feat: optimize shots for robust clean entry probability"
```

---

### Task 8: Advanced Physics UI and Explicit Estimator Application

**Files:**
- Create: `src/AdvancedPhysicsPanel.jsx`
- Modify: `src/TrajectorySimulator.jsx`
- Modify: `tests/ui_contract_js.test.mjs`
- Modify: `tests/physics3d_js.test.mjs`

**Interfaces:**
- Consumes: calibration profile parser/application, robust optimizer mode, robot velocity/wind shot fields.
- Produces:
  - `AdvancedPhysicsPanel` controlled component receiving baseline/profile state, motion/environment values, uncertainty config, profile diagnostics, and change callbacks.
  - UI state for `calibrationProfile | null`, `robotVelocity`, `wind`, `robustEnabled`, `uncertaintyConfig`, `uncertaintySeed`, and `robustSampleCount`.
  - Explicit **Apply Estimate** action for flywheel-derived speed/spin.

- [ ] **Step 1: Write failing UI contract tests**
  - Advanced Physics section exists and exposes robot forward/lateral velocity and wind forward/lateral controls.
  - Loading invalid JSON/profile displays an error and leaves the previous active profile unchanged.
  - Clearing a valid profile returns the simulator to baseline scalar coefficients.
  - The UI displays calibration profile name, validation RMS when present, domain, and out-of-domain/clamping warning.
  - Changing flywheel estimator controls alone does not call `setVelocity` or `setSpinRPM`; an explicit `Apply Estimate` control does.
  - Robust optimization invokes worker mode `robust` only when robust analysis is enabled; deterministic buttons preserve existing modes otherwise.

- [ ] **Step 2: Run UI tests to verify failure**

Run:
```bash
node --test tests/ui_contract_js.test.mjs tests/physics3d_js.test.mjs
```

- [ ] **Step 3: Implement the advanced panel and state plumbing**
  - Keep the default panel collapsed.
  - File/profile loading uses `File.text()` + JSON parsing only.
  - Map UI motion controls into 3-D vectors with vertical component zero.
  - Pass active profile models, wind, and robot velocity into `simulateTrajectory2D`.
  - Display robust outcome probabilities and percentile clearance when robust results are available.
  - Keep current simple deterministic result display intact.

- [ ] **Step 4: Run UI tests, lint, and build**

Run:
```bash
node --test tests/ui_contract_js.test.mjs tests/physics3d_js.test.mjs
npm run lint
npm run build
```

Expected: PASS / exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/AdvancedPhysicsPanel.jsx src/TrajectorySimulator.jsx tests/ui_contract_js.test.mjs tests/physics3d_js.test.mjs
git commit -m "feat: expose calibrated and robust physics controls"
```

---

### Task 9: Calibration Guide, README, and End-to-End Verification

**Files:**
- Create: `docs/calibration-guide.md`
- Modify: `README.md`
- Modify: `tests/test_physics3d_fixtures.py` only if fixture generation coverage needs the new advanced-model cases.
- Modify: `tests/fixtures/physics3d_golden.json` only to append calibrated-model parity cases, preserving existing entries.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented measurement/calibration workflow and final parity fixtures.

- [ ] **Step 1: Add calibrated-model cross-language golden cases**
  - Include at least one `table1d` drag trajectory and one `table2d` lift trajectory with wind and nonzero robot launch velocity.
  - Python-generated expected data must be consumed by the JavaScript golden-fixture test.

- [ ] **Step 2: Run parity tests and confirm any new case fails before fixture/consumer support is complete**

Run:
```bash
python -m unittest tests.test_physics3d_fixtures -v
node --test tests/physics3d_js.test.mjs
```

- [ ] **Step 3: Write the calibration documentation**
  - High-speed-video setup and scale calibration.
  - Measuring exit velocity and marked-ball spin.
  - Capturing multiple speed/spin conditions.
  - Separating low-spin drag data from lift data.
  - JSON/CSV dataset fields and SI units.
  - Running `scripts/calibrate_fuel.py`.
  - Reading RMS and classification validation metrics.
  - Calibrated `Re/S` domain and clamp warnings.
  - Recalibration guidance for materially different ball wear/launcher setup.
  - Clear distinction among numerical integration error, model-form uncertainty, fitted-parameter uncertainty, launch repeatability, and conservative collision physics.

- [ ] **Step 4: Update README**
  - Replace the baseline-only physics description with baseline + calibrated modes.
  - Keep baseline values explicitly labeled uncalibrated.
  - Add moving-robot, wind, profile, robust optimization, and calibration-guide links.
  - Do not claim measured FUEL coefficients are bundled.

- [ ] **Step 5: Run the complete repository verification**

Run:
```bash
npm run test:physics
python -m unittest discover -s tests -v
npm run lint
npm run build
```

Expected: all tests PASS, lint exits 0, build exits 0.

- [ ] **Step 6: Review requirements against the design spec**
  - Confirm every success criterion and non-goal is represented in code/tests/docs.
  - Confirm default scalar-coefficient behavior remains unchanged when no profile/robust settings are enabled.
  - Confirm no speculative collision/rebound physics was introduced.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/calibration-guide.md tests/test_physics3d_fixtures.py tests/fixtures/physics3d_golden.json
git commit -m "docs: add FUEL calibration and validation workflow"
```
