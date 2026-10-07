# Physics Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add defensible no-new-measurement physics improvements—buoyancy, spin-aware drag-model support, signed lift, and low-spin drag-calibration staging—without inventing FUEL aerodynamic data.

**Architecture:** Keep `physics3d` as the canonical ODE/integration core in both JavaScript and Python. Extend the pure aerodynamic-model layer first, then integrate buoyancy and signed/spin-aware forces, propagate the new compatibility parameter through API/shot/uncertainty paths, and finally correct the calibration staging so drag is fit only from low-spin shots.

**Tech Stack:** React 19, JavaScript ES modules, Node built-in test runner, Python 3.9+, NumPy, SciPy `least_squares`, FastAPI/Pydantic, Vite.

**Spec:** `docs/superpowers/specs/2026-10-07-physics-model-improvements-design.md`

## Global Constraints

- Do not introduce measured-looking or guessed FUEL-specific `Cd(Re)`, `Cd(Re,S)`, `Cl(Re,S)`, spin-decay, deformation, rebound, or turbulence constants.
- Keep the default scalar drag coefficient `Cd = 0.47` and legacy lift cap `Cl = 0.25` unchanged.
- Keep `legacy-spin-cap` behavior unchanged and nonnegative.
- Existing `constant` and `table1d` drag profiles and existing positive lift profiles must remain valid.
- Keep calibration schema string `frc-projectile-calibration-v1`; do not require a migration.
- `table2d` drag is representational support only; do not add automatic 2-D drag fitting.
- Buoyancy is enabled by default and is the one intentional nominal trajectory change.
- RK4/RK45 algorithms, state ordering, coordinate conventions, HUB geometry, scoring policy, and UI backspin sign convention remain unchanged.
- Calibration low-spin selection uses dimensionless spin parameter `S`, not raw RPM/rad/s.
- Default calibration threshold is `--drag-max-spin-parameter 0.05`.
- Calibration lift fitting bounds are `[-3, 3]`; this is an optimizer safety bound, not a physical FUEL claim.
- Imported calibration profiles remain data-only; no executable expressions or code evaluation.

## Review Focus

- **Zero relative airflow with a 2-D drag table:** diagnostics may clamp the coefficient lookup at `Re=0,S=0`, but translational aerodynamic acceleration must remain exactly zero; pinned in Task 2.
- **Negative lift under uncertainty scaling:** multiplying a signed `Cl` table by a nonnegative uncertainty factor must preserve coefficient sign and force direction; pinned in Task 3.
- **Buoyancy compatibility:** `enableBuoyancy=false` / `enable_buoyancy=False` must reproduce the old `-g` vertical acceleration exactly, while zero air density also removes buoyancy; pinned in Task 2.
- **Legacy drag-evaluator callers:** calling `evaluateDragModel(model, Re)` / `evaluate_drag_model(model, Re)` without spin must continue to work and behave as `S=0`; pinned in Task 1.
- **Calibration filtering after train/validation split:** if the training subset contains no shot at or below the threshold, the CLI must fail clearly rather than borrowing validation shots or falling back to all training shots; pinned in Task 4.

---

### Task 1: Extend Aerodynamic Model Semantics

**Files:**
- Modify: `src/aerodynamics.js`
- Modify: `api/aerodynamics.py`
- Modify: `tests/aerodynamics_js.test.mjs`
- Modify: `tests/test_aerodynamics.py`
- Modify tests only as needed: `tests/calibration_js.test.mjs`, `tests/test_calibration.py`

**Interfaces:**
- Consumes: existing normalized aerodynamic model schemas.
- Produces JavaScript:
  - `normalizeDragModel(model, fallbackCoefficient)` accepts `table2d`.
  - `evaluateDragModel(model, reynolds, spinParameterValue = 0) -> {coefficient, clamped}`.
  - `normalizeLiftModel(...)` accepts signed finite coefficients for `table1d` and `table2d`.
- Produces Python:
  - `normalize_drag_model(model, fallback_coefficient)` accepts `table2d`.
  - `evaluate_drag_model(model, reynolds, spin_parameter_value=0.0) -> dict`.
  - `normalize_lift_model(...)` accepts signed finite tabulated coefficients.
- Drag coefficients remain finite and nonnegative.
- `legacy-spin-cap.maxCoefficient` remains finite and nonnegative.

- [ ] **Step 1: Write failing JS aerodynamic tests**

Add tests named to cover:
- `table2d drag interpolates over reynolds and spin`;
- `table2d drag clamps either axis`;
- `legacy drag evaluation defaults omitted spin to zero`;
- `signed lift tables normalize and interpolate negative values`;
- `negative drag remains invalid`;
- `negative legacy lift cap remains invalid`.

Use a 2x2 drag table such as:
```js
{
  kind: 'table2d',
  reynolds: [100000, 200000],
  spinParameters: [0, 1],
  coefficients: [[0.5, 0.4], [0.3, 0.2]],
}
```
Assert the bilinear midpoint at `Re=150000,S=0.5` is `0.35`, and assert omitted `S` evaluates the same as `S=0`.

- [ ] **Step 2: Write equivalent failing Python aerodynamic tests**

Mirror the JavaScript assertions in `tests/test_aerodynamics.py` using snake_case APIs and the same numeric tables/tolerances.

- [ ] **Step 3: Run focused tests and confirm the intended failures**

Run:
```bash
node --test tests/aerodynamics_js.test.mjs
python -m unittest tests.test_aerodynamics -v
```

Expected: new `table2d` drag and signed-lift tests FAIL; existing tests remain otherwise green.

- [ ] **Step 4: Implement separate drag/lift coefficient validation**

In both aerodynamic modules:
- retain a nonnegative coefficient helper for drag and `legacy-spin-cap.maxCoefficient`;
- add a finite signed coefficient helper for tabulated lift coefficients;
- do not loosen axis validation: Reynolds/spin axes remain finite, nonnegative, and strictly increasing.

- [ ] **Step 5: Implement `table2d` drag normalization**

Use the same row/column convention already used for lift:
- rows follow `reynolds`;
- columns follow `spinParameters`;
- reject malformed matrix dimensions;
- reject negative drag entries.

- [ ] **Step 6: Extend drag evaluation signature and interpolation**

Implement:
```text
evaluateDragModel(model, reynolds, spinParameterValue = 0)
evaluate_drag_model(model, reynolds, spin_parameter_value=0.0)
```

Behavior:
- `constant`: ignore `Re,S`;
- `table1d`: use `Re`, ignore `S`;
- `table2d`: bilinear interpolation over `Re,S`;
- clamp either axis and OR the clamp flags.

- [ ] **Step 7: Add calibration-profile compatibility assertions**

In `tests/calibration_js.test.mjs` and `tests/test_calibration.py`, add one profile fixture/assertion proving schema v1 accepts:
- `dragModel.kind = "table2d"`;
- a lift table containing at least one negative coefficient.

No production parser change is required unless these tests reveal a parser-specific restriction beyond aerodynamic normalization.

- [ ] **Step 8: Run focused tests**

Run:
```bash
node --test tests/aerodynamics_js.test.mjs tests/calibration_js.test.mjs
python -m unittest tests.test_aerodynamics tests.test_calibration -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/aerodynamics.js api/aerodynamics.py tests/aerodynamics_js.test.mjs tests/test_aerodynamics.py tests/calibration_js.test.mjs tests/test_calibration.py
git commit -m "feat: extend aerodynamic coefficient models"
```

---

### Task 2: Add Buoyancy and Signed/Spin-Aware Forces to the Canonical Engines

**Files:**
- Modify: `src/physics3d.js`
- Modify: `api/physics3d.py`
- Modify: `tests/physics3d_js.test.mjs`
- Modify: `tests/test_physics3d.py`
- Modify: `scripts/generate_physics3d_fixtures.py`
- Regenerate: `tests/fixtures/physics3d_golden.json`

**Interfaces:**
- Consumes: Task 1 drag/lift evaluators.
- Produces JavaScript flight parameter `enableBuoyancy: true`.
- Produces Python `FlightParameters.enable_buoyancy: bool = True`.
- Canonical acceleration includes
  `+ airDensity * (4/3*pi*radius^3) * gravity / mass` on z when buoyancy is enabled.
- Canonical aerodynamic diagnostics pass current `spinParameter` into drag evaluation.
- Magnus force applies any nonzero signed `liftCoefficient`.

- [ ] **Step 1: Write failing buoyancy tests in JavaScript**

Add assertions that:
- with drag/Magnus off, `az = -g + rho*(4/3*pi*r^3)*g/m`;
- `enableBuoyancy:false` gives exactly `-g`;
- `airDensity:0` yields no buoyancy;
- `gravity:0` yields no buoyancy.

Use current nominal defaults where convenient, but compute the expected buoyancy term from parameters inside the test.

- [ ] **Step 2: Write failing buoyancy tests in Python**

Mirror Step 1 against `FlightParameters` and `derivatives`.

- [ ] **Step 3: Write failing signed-Magnus and spin-aware-drag engine tests**

For both languages:
- use the same state and two identical lift tables except `Cl=+0.2` vs `Cl=-0.2`; assert Magnus acceleration contributions are equal magnitude and opposite direction after subtracting gravity/buoyancy;
- use a `table2d` drag model with coefficients differing by spin column; assert the same translational speed with different perpendicular spin produces different drag acceleration;
- at matching wind / zero relative airflow, assert aerodynamic translational force is zero even if the model lookup reports a clamped 2-D coefficient.

- [ ] **Step 4: Update analytic-vacuum tests before running the new default**

Tests whose purpose is exact gravity-only analytic integration must set:
- JS: `enableBuoyancy:false`;
- Python: `enable_buoyancy=False`.

Do not disable buoyancy in ordinary atmospheric regression tests.

- [ ] **Step 5: Run focused tests and confirm intended failures**

Run:
```bash
node --test tests/physics3d_js.test.mjs
python -m unittest tests.test_physics3d -v
```

Expected: new buoyancy/signed/2-D-drag tests FAIL until production code changes.

- [ ] **Step 6: Implement buoyancy in both derivative functions**

Add the default parameter and apply the upward acceleration before drag/Magnus:
```text
volume = 4/3 * pi * radius^3
buoyant_accel = airDensity * volume * gravity / mass
```

Only apply it when `enableBuoyancy` / `enable_buoyancy` is true. Existing validation of mass, radius, density, and gravity supplies the required domains.

- [ ] **Step 7: Pass spin parameter into drag evaluation**

In each aerodynamic-state helper:
- zero-airflow branch calls drag evaluation with `S=0`;
- moving-air branch calls drag evaluation with current computed `spinParameter`.

Do not change the Reynolds or spin-parameter definitions.

- [ ] **Step 8: Apply signed Magnus coefficients**

Replace the positive-only gate with a nonzero check:
- keep `omega_perp_mag > EPS`;
- keep lift-direction normalization unchanged;
- multiply by signed `Cl` so negative coefficients reverse direction naturally.

- [ ] **Step 9: Update and regenerate canonical cross-language fixtures**

Modify `scripts/generate_physics3d_fixtures.py` so cases intended as vacuum explicitly disable buoyancy. Add representative cases for:
- buoyancy;
- signed lift;
- calibrated `table2d` drag.

Run:
```bash
python scripts/generate_physics3d_fixtures.py
```

Review the fixture diff; accept changed existing values only where default atmospheric buoyancy intentionally affects that case.

- [ ] **Step 10: Run engine and fixture tests**

Run:
```bash
node --test tests/physics3d_js.test.mjs
python -m unittest tests.test_physics3d tests.test_physics3d_fixtures -v
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/physics3d.js api/physics3d.py tests/physics3d_js.test.mjs tests/test_physics3d.py scripts/generate_physics3d_fixtures.py tests/fixtures/physics3d_golden.json
git commit -m "feat: add buoyancy and signed aerodynamic forces"
```

---

### Task 3: Propagate Buoyancy and Advanced Drag Through Shot, API, and Uncertainty Paths

**Files:**
- Modify: `src/trajectory2d.js`
- Modify: `src/uncertainty.js`
- Modify: `api/main.py`
- Modify: `api/uncertainty.py`
- Modify if required by parity tests: `api/trajectory_simulator.py`
- Modify: `tests/physics3d_js.test.mjs`
- Modify: `tests/test_api.py`
- Modify: `tests/uncertainty_js.test.mjs`
- Modify: `tests/test_uncertainty.py`
- Modify as needed: `tests/test_legacy_adapter.py`

**Interfaces:**
- Consumes: Task 2 engine parameter `enableBuoyancy` / `enable_buoyancy`.
- Produces:
  - `simulateShot(params, options)` defaults `enableBuoyancy = true` and forwards it exactly once to flight parameters.
  - `Sim3DRequest.enable_buoyancy: bool = True`.
  - Python uncertainty construction maps browser-style `enableBuoyancy` to `FlightParameters.enable_buoyancy`.
  - uncertainty drag scaling supports `table2d` coefficient matrices.
  - signed lift scaling multiplies values without taking absolute value or otherwise changing sign.

- [ ] **Step 1: Write failing shot/API propagation tests**

Add tests proving:
- `simulateShot` defaults buoyancy on;
- passing `enableBuoyancy:false` reaches the engine and restores old gravity-only behavior when drag/Magnus are also off;
- `Sim3DRequest(enable_buoyancy=False)` reaches `FlightParameters`;
- omitting the API field defaults it to true.

- [ ] **Step 2: Write failing uncertainty compatibility tests**

In JS and Python uncertainty tests:
- create a `table2d` drag model and apply fixed `dragMultiplier: 2`; assert every matrix coefficient doubles;
- create a signed lift table such as `[-0.2, 0.1]` and apply fixed `liftMultiplier: 0.5`; assert results are `[-0.1, 0.05]`, preserving sign;
- assert sampled parameter dictionaries preserve any explicit `enableBuoyancy:false`.

- [ ] **Step 3: Run focused tests and confirm failures**

Run:
```bash
node --test tests/physics3d_js.test.mjs tests/uncertainty_js.test.mjs
python -m unittest tests.test_api tests.test_uncertainty tests.test_legacy_adapter -v
```

Expected: new propagation/scaling tests FAIL.

- [ ] **Step 4: Forward buoyancy through browser shot simulation**

In `src/trajectory2d.js`:
- destructure `enableBuoyancy = true`;
- include it in `flightParams`;
- do not add a new UI toggle in this milestone.

- [ ] **Step 5: Add the API field and forwarding**

In `api/main.py`:
- add `enable_buoyancy: bool = True` to `Sim3DRequest`;
- pass it to `FlightParameters`.

Keep existing request payloads valid.

- [ ] **Step 6: Forward buoyancy through uncertainty and legacy construction**

In `api/uncertainty.py`, map:
```python
enable_buoyancy=params.get("enableBuoyancy", True)
```

In any legacy-adapter constructor that explicitly reconstructs all physical toggles, forward the new field or rely on the canonical default only if no user-facing override exists. Preserve existing adapter parity tests.

- [ ] **Step 7: Extend uncertainty model scalers**

Support `table2d` drag in both JS and Python by multiplying every matrix entry by the nonnegative multiplier.

Do not special-case negative lift values: ordinary multiplication by a nonnegative factor must preserve sign.

- [ ] **Step 8: Run focused compatibility tests**

Run the commands from Step 3.

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/trajectory2d.js src/uncertainty.js api/main.py api/uncertainty.py api/trajectory_simulator.py tests/physics3d_js.test.mjs tests/test_api.py tests/uncertainty_js.test.mjs tests/test_uncertainty.py tests/test_legacy_adapter.py
git commit -m "feat: propagate buoyancy through simulator interfaces"
```

---

### Task 4: Correct Calibration Staging and Signed Lift Fitting

**Files:**
- Modify: `calibration/fitting.py`
- Modify: `scripts/calibrate_fuel.py`
- Modify: `tests/test_calibration.py`
- Create: `tests/test_calibration_cli.py`

**Interfaces:**
- Consumes: existing `_shot_initial_conditions(shot, base_params) -> (Re, S)`.
- Produces:
  - `partition_shots_by_spin_parameter(shots, base_params, max_drag_spin_parameter) -> tuple[list, list]`.
  - CLI option `--drag-max-spin-parameter` with default `0.05`.
  - drag fit uses only `S <= threshold` training shots.
  - lift fit uses only `S > threshold` training shots.
  - lift-table least-squares bounds become `[-3, 3]`.
- The train/validation split occurs before spin partitioning; validation shots are never borrowed to satisfy training-data requirements.

- [ ] **Step 1: Write failing partition-helper tests**

In `tests/test_calibration.py` add tests that:
- threshold validation rejects negative or non-finite values;
- selection is based on dimensionless `S = omega_perp*r/v`, not raw spin magnitude;
- a shot exactly at the threshold is in the drag subset;
- a shot above the threshold is in the spinning subset.

Use simple launch states where `S` is easy to compute exactly.

- [ ] **Step 2: Write a failing signed-lift-fit regression**

Construct synthetic spinning shots from a truth model whose tabulated lift includes a negative coefficient in part of the sampled domain. Assert `fit_lift_model(...)` can return at least one negative fitted coefficient and improves validation error versus an all-zero lift model.

Keep the synthetic dataset within a regime where the fit is identifiable and deterministic.

- [ ] **Step 3: Write failing CLI tests**

Create `tests/test_calibration_cli.py` using `tempfile` plus `subprocess.run` against `scripts/calibrate_fuel.py`.

Cover:
- default threshold `0.05` selects only low-spin training shots;
- the CLI prints a selection summary containing drag count, spinning count, and threshold;
- if the post-split training set has zero drag shots at or below the threshold, the process exits nonzero and stderr contains both the threshold and guidance to collect low-spin data or raise `--drag-max-spin-parameter`;
- the script does not use validation shots to rescue an empty drag subset;
- with fewer than two spinning training shots, lift fitting is skipped and the output profile contains the explicit zero `legacy-spin-cap` fallback.

- [ ] **Step 4: Run focused tests and confirm failures**

Run:
```bash
python -m unittest tests.test_calibration tests.test_calibration_cli -v
```

Expected: new partition, signed-fit, and CLI tests FAIL.

- [ ] **Step 5: Implement `partition_shots_by_spin_parameter`**

Signature:
```python
def partition_shots_by_spin_parameter(
    shots: Sequence[Dict[str, Any]],
    base_params: Any,
    max_drag_spin_parameter: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
```

Validate the threshold as finite and nonnegative. Use `_shot_initial_conditions` so robot velocity, wind, radius, density, and viscosity semantics match the rest of calibration.

Partition rule:
- drag: `S <= threshold`;
- spinning: `S > threshold`.

- [ ] **Step 6: Allow signed lift fitting bounds**

For `table1d` and `table2d` lift fits:
- change lower bounds to `-3`;
- keep upper bounds `+3`;
- keep existing positive initial guesses;
- keep smoothness regularization unchanged.

Do not change drag-fit bounds.

- [ ] **Step 7: Add CLI threshold and selection**

In `scripts/calibrate_fuel.py`:
- add `--drag-max-spin-parameter`, `type=float`, default `0.05`;
- split train/validation first;
- partition only `train`;
- fit drag from `drag_shots`;
- fit lift from `spinning_shots`;
- print a deterministic selection summary before fitting, e.g. `Calibration selection: drag=<n> spinning=<n> max_drag_spin_parameter=<value>`.

If `drag_shots` is empty, exit with a clear error naming the threshold and suggesting low-spin data or an explicitly larger threshold.

- [ ] **Step 8: Preserve the no-lift-data fallback explicitly**

When fewer than two spinning training shots are available:
- keep `{"kind":"legacy-spin-cap","maxCoefficient":0.0,"saturationSpin":0.5}`;
- print a concise note that lift fitting was skipped for insufficient spinning shots.

Do not label the fallback as measured or fitted.

- [ ] **Step 9: Run calibration tests**

Run:
```bash
python -m unittest tests.test_calibration tests.test_calibration_cli -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add calibration/fitting.py scripts/calibrate_fuel.py tests/test_calibration.py tests/test_calibration_cli.py
git commit -m "fix: isolate low-spin drag calibration data"
```

---

### Task 5: Documentation, Full Regression, and Build Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/calibration-guide.md`
- Modify any tests/fixtures only if full-suite verification reveals an intentional buoyancy compatibility expectation that was not updated in Tasks 1–4.

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: user-facing documentation matching shipped behavior and a fully verified branch.

- [ ] **Step 1: Update README physics documentation**

Document:
- buoyancy is included by default;
- drag models may be `constant`, `Cd(Re)`, or supplied `Cd(Re,S)` tables;
- no measured FUEL `Cd(Re,S)` table is bundled;
- lift tables may be signed, but the default legacy lift model remains unchanged/nonnegative;
- old measured calibration profiles should be revalidated because buoyancy changes the modeled force balance.

Do not imply reverse Magnus has been observed on FUEL.

- [ ] **Step 2: Update the calibration guide**

Add:
- `--drag-max-spin-parameter 0.05`;
- explanation that the threshold is a data-selection rule, not an aerodynamic constant;
- drag fitting uses only low-spin training shots after the train/validation split;
- guidance for the insufficient-low-spin error;
- note that automatic `Cd(Re,S)` fitting is intentionally not provided;
- revalidation guidance for profiles fitted before buoyancy support.

- [ ] **Step 3: Run the complete JavaScript test suite**

Run:
```bash
npm run test:physics
```

Expected: all Node tests PASS.

- [ ] **Step 4: Run the complete Python test suite**

Run:
```bash
python -m unittest discover -s tests -v
```

Expected: all Python tests PASS.

- [ ] **Step 5: Run lint and production build**

Run:
```bash
npm run lint
npm run build
```

Expected: both exit 0.

- [ ] **Step 6: Run whitespace/diff integrity check**

Run:
```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 7: Re-read the approved spec against the final diff**

Verify explicitly:
- no speculative FUEL coefficient table/data was added;
- no automatic 2-D drag fitting was added;
- schema string is unchanged;
- default `Cd`/`Cl` constants are unchanged;
- buoyancy default and opt-out are documented;
- calibration partition uses `S` and threshold `0.05`.

- [ ] **Step 8: Commit documentation/final compatibility adjustments**

```bash
git add README.md docs/calibration-guide.md
git add -u
git commit -m "docs: document improved projectile physics model"
```

- [ ] **Step 9: Fresh post-commit verification**

Run again on the committed tree:
```bash
npm run test:physics
python -m unittest discover -s tests -v
npm run lint
npm run build
git diff --check
```

Expected: all commands exit 0; working tree is clean apart from any execution-workspace ledger files required by the chosen Superpowers execution mode.
