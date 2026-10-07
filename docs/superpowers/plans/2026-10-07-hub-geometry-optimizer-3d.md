# Milestone 2.1 HUB Geometry, Optimizer, and 3-D View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make target scoring geometrically consistent with the HUB, keep optimization responsive, repair the Physics Options switches, and add an interactive 3-D trajectory view without changing the Milestone 2 flight-physics model.

**Architecture:** A new pure `hubGeometry.js` module owns official HUB dimensions, regular-hex/frustum geometry, and swept-sphere interaction classification. `trajectory2d.js` integrates each shot once and analyzes those samples; a pure coarse-to-fine optimizer runs inside a module Web Worker managed by a testable lifecycle client. The React UI consumes the same geometry for both its existing 2-D view and a new SVG orthographic 3-D view.

**Tech Stack:** React 19, Vite 7, JavaScript ES modules, Node built-in `node:test`, browser module Web Workers, SVG; existing Python/NumPy/FastAPI physics remain unchanged.

**Spec:** `docs/superpowers/specs/2026-10-07-hub-geometry-optimizer-3d-design.md`

## Global Constraints

- Retain the Milestone 2 coordinate system: x = forward/downrange, y = left lateral, z = up.
- Do not change FUEL mass, radius, `Cd`, `Cl(S)`, spin-decay defaults, RK4/RK45 equations, or Python/JavaScript golden aerodynamic fixtures.
- HUB top opening across flats is `41.727 in`; top plane is `72.0 in` above carpet.
- Funnel panel bottom side is `18.92 in`; panel vertical height is `17.90 in`.
- One flat face of the regular hex faces the shooter at negative x.
- A valid optimizer hit is only `classification === "clean-entry"`; rim/funnel collisions never count.
- A displayed or optimized browser shot performs one flight integration; HUB analysis consumes those returned samples.
- Existing UI trajectory integration defaults to RK4 `dt=0.001`.
- Coarse optimizer passes may use `dt=0.005`; every returned solution is revalidated at `dt=0.001`.
- Optimizer search loops must execute in a Web Worker, never synchronously in a React event handler.
- Only one optimizer run may be active; cancellation uses `worker.terminate()`.
- Preserve the existing 2-D view as the default; do not add Three.js/WebGL or another runtime dependency.
- 3-D geometry/rendering and hit classification must derive from the same `hubGeometry.js` values/functions.
- Physics switches must use a native checkbox and support click/touch, label click, keyboard focus, and Space activation.
- `python -m unittest discover -s tests -v`, all Node physics/UI tests, and `npm run build` remain required gates.

## Review Focus

- **Hex-corner grazing:** a sphere near a top-opening vertex must use true edge/vertex distance and classify rim contact rather than slipping through a half-space-only test; pinned in Task 2.
- **Trajectories that start below or inside the funnel:** without a descending top-entry crossing they must never be classified `clean-entry`; pinned in Task 2.
- **Stale worker completion after cancel/restart:** messages from an old request must not overwrite a newer optimization result; pinned in Task 5.
- **No-solution optimizer run:** controls must remain unchanged, worker must terminate, and a best near-miss may be reported without presenting it as a solution; pinned in Tasks 4 and 5.
- **Degenerate/near-flat 3-D scene extents:** orthographic projection must remain finite when all samples share one lateral coordinate or nearly one axis value; pinned in Task 6.

---

### Task 1: Repair the Physics Option Toggle Component

**Files:**
- Create: `src/Toggle.jsx`
- Create: `tests/ui_contract_js.test.mjs`
- Modify: `src/TrajectorySimulator.jsx`
- Modify: `package.json`

**Interfaces:**
- Produces: `Toggle({label, checked, onChange})` default/named React component using a native controlled checkbox.
- Later tasks consume the existing `enableDrag`, `enableMagnus`, `showIdeal`, and `showEnvelope` state setters through this component.

- [ ] **Step 1: Write the failing source-structure regression**

Create `tests/ui_contract_js.test.mjs` and assert the production component contains the required native semantics:

```javascript
test('Toggle uses a controlled native checkbox wired to onChange', async () => {
  const source = await readFile(new URL('../src/Toggle.jsx', import.meta.url), 'utf8');
  assert.match(source, /type=["']checkbox["']/);
  assert.match(source, /checked=\{checked\}/);
  assert.match(source, /onChange=\{[^}]*onChange/);
  assert.match(source, /event\.target\.checked/);
});
```

Also read `src/TrajectorySimulator.jsx` and assert the four labels `Air Drag`, `Magnus Effect (Backspin)`, `Show Ideal (No Drag)`, and `Show Error Envelope` still render through `Toggle`.

- [ ] **Step 2: Run the regression and verify RED**

Run: `node --test tests/ui_contract_js.test.mjs`

Expected: FAIL because `src/Toggle.jsx` does not exist.

- [ ] **Step 3: Implement `Toggle` and replace the inline decorative component**

`src/Toggle.jsx` renders a native `<input type="checkbox">` with:

- `checked={checked}`;
- `onChange={(event) => onChange(event.target.checked)}`;
- an associated visible label/switch treatment;
- keyboard-visible focus styling;
- no custom click handler that duplicates native checkbox behavior.

Remove the old inline `Toggle` definition from `TrajectorySimulator.jsx` and import the new component.

- [ ] **Step 4: Make all Node test files part of `test:physics`**

Change `package.json`:

```json
"test:physics": "node --test tests/*_js.test.mjs"
```

- [ ] **Step 5: Verify GREEN and build**

Run:
- `npm run test:physics`
- `npm run build`

Expected: all Node tests PASS and Vite exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/Toggle.jsx src/TrajectorySimulator.jsx tests/ui_contract_js.test.mjs package.json
git commit -m "fix: make physics options interactive"
```

### Task 2: Add Official HUB Geometry and Swept-Sphere Classification

**Files:**
- Create: `src/hubGeometry.js`
- Create: `tests/hub_geometry_js.test.mjs`

**Interfaces:**
- Produces:
  - `INCH_TO_METER`
  - `HUB_DIMENSIONS`
  - `createHubGeometry({centerX = 0, centerY = 0} = {})`
  - `hexVertices(apothem, z, centerX = 0, centerY = 0)`
  - `classifyHubInteraction(samples, geometry, ballRadius)`
- `createHubGeometry` returns at least `topZ`, `bottomZ`, `topApothem`, `bottomApothem`, `slope`, `normals`, `topVertices`, and `bottomVertices`.
- `classifyHubInteraction` returns `{classification, topCrossing, bottomCrossing, collisionPoint, clearanceMargin, missDistance}`.

- [ ] **Step 1: Write failing dimension and vertex tests**

Pin the official values:

```javascript
test('HUB dimensions match official drawings', () => {
  assert.ok(Math.abs(HUB_DIMENSIONS.topAcrossFlats - 41.727 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.topZ - 72 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.bottomSide - 18.92 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.panelHeight - 17.90 * 0.0254) < 1e-12);
});
```

For the default geometry, assert six top and six bottom vertices and that all six top vertices satisfy the regular-hex half-space equations with one flat at negative x.

- [ ] **Step 2: Write failing interaction tests with synthetic samples**

Use simple linearly spaced states, not the aerodynamic solver, to isolate geometry:

```javascript
test('centerline descending passage is clean entry', () => {
  const result = classifyHubInteraction(centerlineSamples(), createHubGeometry(), 0.075);
  assert.equal(result.classification, 'clean-entry');
  assert.ok(result.clearanceMargin > 0);
  assert.ok(result.topCrossing);
  assert.ok(result.bottomCrossing);
});

test('top edge overlap is rim collision', () => {
  const x = createHubGeometry().topApothem - 0.03;
  const result = classifyHubInteraction(verticalSamplesAtX(x), createHubGeometry(), 0.075);
  assert.equal(result.classification, 'rim-collision');
  assert.ok(result.clearanceMargin < 0);
});

test('clean top entry that reaches shrinking side is funnel collision', () => {
  const result = classifyHubInteraction(verticalSamplesAtX(0.42), createHubGeometry(), 0.075);
  assert.equal(result.classification, 'funnel-collision');
  assert.ok(result.collisionPoint);
});

test('sphere fully outside top opening is miss', () => {
  const g = createHubGeometry();
  const result = classifyHubInteraction(verticalSamplesAtX(g.topApothem + 0.20), g, 0.075);
  assert.equal(result.classification, 'miss');
  assert.ok(result.missDistance > 0);
});
```

- [ ] **Step 3: Add Review Focus geometry tests before implementation**

Add:

- a point near a top hex vertex whose center is outside but whose sphere overlaps the vertex/adjacent edges → `rim-collision`;
- samples beginning below `topZ` with no descending top crossing → never `clean-entry`;
- bottom-rim overlap after otherwise clear passage → `funnel-collision`.

- [ ] **Step 4: Run and verify RED**

Run: `node --test tests/hub_geometry_js.test.mjs`

Expected: FAIL because `src/hubGeometry.js` does not exist.

- [ ] **Step 5: Implement the regular-hex/frustum geometry**

Use six horizontal outward normals at angles `0, 60, 120, 180, 240, 300°`, giving flats at `x = ±apothem`.

Compute:

```text
topApothem = 41.727 in / 2
bottomApothem = sqrt(3) * 18.92 in / 2
bottomZ = topZ - 17.90 in
slope = (topApothem - bottomApothem) / (topZ - bottomZ)
```

Implement exact minimum point-to-segment distance against six top hex edges for rim overlap, rather than using only side-plane signed distances.

- [ ] **Step 6: Implement `classifyHubInteraction`**

Requirements:

- locate the first descending top-plane crossing by segment interpolation;
- distinguish miss vs rim overlap at the top;
- clip subsequent segments to `[bottomZ, topZ]`;
- evaluate sloped-panel sphere clearance at clipped segment endpoints;
- interpolate the first zero-clearance collision;
- require descending bottom crossing with full bottom-edge ball clearance for `clean-entry`;
- never classify a trajectory lacking top entry as clean.

- [ ] **Step 7: Verify geometry tests**

Run:
- `node --test tests/hub_geometry_js.test.mjs`
- `npm run test:physics`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hubGeometry.js tests/hub_geometry_js.test.mjs
git commit -m "feat: model HUB funnel collision geometry"
```

### Task 3: Make Browser Shot Simulation Single-Pass and Geometry-Aware

**Files:**
- Modify: `src/trajectory2d.js`
- Modify: `tests/physics3d_js.test.mjs`
- Modify: `tests/ui_contract_js.test.mjs`

**Interfaces:**
- Consumes: `createHubGeometry` and `classifyHubInteraction` from Task 2; `integrateTrajectory` / `launchState` from Milestone 2.
- Produces:
  - `simulateShot(params, options = {})`
  - compatibility wrapper `simulateTrajectory2D(params, options = {})`
- `simulateShot` returns `{samples3d, points, hubInteraction, hitTarget, impactPoint, flightTime, maxHeight, range, entryVelocity, entryAngle}`.

- [ ] **Step 1: Write failing single-pass and result-shape tests**

Add tests:

```javascript
test('simulateShot returns canonical 3-D samples and x/z projection from the same flight', () => {
  const result = simulateShot(baseParams(), {dt: 0.002});
  assert.ok(result.samples3d.length > 2);
  assert.equal(result.points.length > 2, true);
  assert.equal(result.points[0].x, result.samples3d[0].state[0]);
  assert.equal(result.points[0].y, result.samples3d[0].state[2]);
  assert.ok(result.hubInteraction);
});

test('hitTarget is true only for clean entry', () => {
  const clean = knownCleanFixture();
  assert.equal(clean.hubInteraction.classification, 'clean-entry');
  assert.equal(clean.hitTarget, true);

  const collision = knownFunnelCollisionFixture();
  assert.notEqual(collision.hubInteraction.classification, 'clean-entry');
  assert.equal(collision.hitTarget, false);
});
```

- [ ] **Step 2: Pin one integration call structurally**

In `tests/ui_contract_js.test.mjs`, read `src/trajectory2d.js` and assert the source contains exactly one production call site matching `integrateTrajectory(`.

This test intentionally protects the acceptance requirement that target analysis consumes one integration rather than adding a second target-plane integration again.

- [ ] **Step 3: Pin optimizer timestep override semantics**

Compare:

```javascript
const fine = simulateShot(baseParams(), {dt: 0.001});
const coarse = simulateShot(baseParams(), {dt: 0.005});
assert.ok(coarse.samples3d.length < fine.samples3d.length);
```

Then call `simulateTrajectory2D(baseParams())` without options and assert its sample/projection timing is consistent with the `dt=0.001` default.

- [ ] **Step 4: Run and verify RED**

Run: `npm run test:physics`

Expected: FAIL because `simulateShot` and geometry-aware scoring are not implemented and the current adapter has two integration call sites.

- [ ] **Step 5: Implement one-pass `simulateShot`**

Perform one ground-terminated RK4 integration. Analyze that returned sample list using `classifyHubInteraction`. Do not integrate a second target trajectory.

Build geometry using `params.targetX ?? 0` and `params.targetLateralY ?? 0`. HUB vertical dimensions come from `HUB_DIMENSIONS`, not browser literals.

Preserve the old result fields through `simulateTrajectory2D`, but define `hitTarget` as `hubInteraction.classification === "clean-entry"`.

- [ ] **Step 6: Verify adapter and regression suites**

Run:
- `npm run test:physics`
- `npm run build`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trajectory2d.js tests/physics3d_js.test.mjs tests/ui_contract_js.test.mjs
git commit -m "refactor: score HUB from one integrated trajectory"
```

### Task 4: Implement Bounded Coarse-to-Fine Optimizers

**Files:**
- Create: `src/optimizer.js`
- Create: `tests/optimizer_js.test.mjs`

**Interfaces:**
- Consumes: `simulateShot(params, {dt})` from Task 3.
- Produces:
  - `rankCandidate(a, b, reference)`
  - `optimizeAngle(params, callbacks = {})`
  - `optimizeVelocity(params, callbacks = {})`
  - `optimizeBoth(params, callbacks = {})`
- Each optimizer returns `{solution, bestNearMiss, evaluatedCandidates}`.
- `callbacks.onProgress(progress)` is optional and receives bounded periodic updates.
- A successful `solution` contains `velocity`, `angle`, and the final `result` revalidated at `dt=0.001`.

- [ ] **Step 1: Write failing deterministic ranking tests**

Pin ranking order:

```javascript
test('clean entry always outranks collision and miss', () => { ... });
test('larger positive clearance wins between clean entries', () => { ... });
test('less-negative clearance wins between collisions', () => { ... });
test('smaller miss distance wins between misses', () => { ... });
test('exact ties prefer parameters closest to the reference', () => { ... });
```

Use hand-built candidate objects so ranking tests do not depend on flight physics.

- [ ] **Step 2: Write failing optimizer behavior tests**

Using real `simulateShot`, add one known-solvable fixture for each mode:

- angle-only returns a `clean-entry`;
- velocity-only returns a `clean-entry`;
- combined returns a `clean-entry`.

For combined search assert:

```javascript
assert.ok(result.evaluatedCandidates < 1100);
assert.equal(result.solution.result.hubInteraction.classification, 'clean-entry');
```

For every returned solution, inspect adjacent final `samples3d` times and assert no full integration step exceeds `0.001 + 1e-12`.

- [ ] **Step 3: Add no-solution Review Focus test**

Use deliberately unreachable bounds/parameters and assert:

- `solution === null`;
- `bestNearMiss` is present when candidates exist;
- `bestNearMiss.result.hubInteraction.classification !== "clean-entry"`.

- [ ] **Step 4: Run and verify RED**

Run: `node --test tests/optimizer_js.test.mjs`

Expected: FAIL because `src/optimizer.js` does not exist.

- [ ] **Step 5: Implement shared candidate evaluation/ranking**

Coarse grids:

- angle: 5..85 by 2° at `dt=0.005`;
- velocity: 5..25 by 1 m/s at `dt=0.005`;
- both: 5..25 by 1 m/s × 20..80 by 2° at `dt=0.005`.

Progress callbacks are emitted after bounded chunks (for example every 25 candidates), not every simulation.

- [ ] **Step 6: Implement local refinement**

Angle-only: refine around top coarse candidates with <=0.25° increments at `dt<=0.002`.

Velocity-only: refine around top coarse candidates with <=0.2 m/s increments at `dt<=0.002`.

Combined: retain at most 12 coarse candidates and evaluate a fixed local grid around each; total evaluations including final validation remain below 1100.

Deduplicate identical parameter pairs before simulation.

- [ ] **Step 7: Revalidate successful final candidate**

Every optimizer must re-run only its chosen clean candidate at `dt=0.001`. If that final result is no longer `clean-entry`, it may consider the next ranked refined clean candidate, also at `dt=0.001`; it must never return an unvalidated coarse hit.

- [ ] **Step 8: Verify optimizer and full Node suite**

Run:
- `node --test tests/optimizer_js.test.mjs`
- `npm run test:physics`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/optimizer.js tests/optimizer_js.test.mjs
git commit -m "feat: add bounded coarse-to-fine optimizer"
```

### Task 5: Move Optimization Into a Cancellable Web Worker

**Files:**
- Create: `src/optimizer.worker.js`
- Create: `src/optimizerClient.js`
- Create: `tests/optimizer_client_js.test.mjs`
- Modify: `src/TrajectorySimulator.jsx`
- Modify: `tests/ui_contract_js.test.mjs`

**Interfaces:**
- Consumes: Task 4 optimizer functions.
- Produces:
  - worker request: `{type: "optimize", requestId, mode, params}`
  - worker progress: `{type: "progress", requestId, progress}`
  - worker completion: `{type: "complete", requestId, result}`
  - worker failure: `{type: "error", requestId, message}`
  - `createOptimizerClient({workerFactory, onProgress, onComplete, onError})` returning `{start(mode, params), cancel(), dispose()}`.

- [ ] **Step 1: Write failing optimizer-client lifecycle tests with a fake Worker**

Pin:

```javascript
test('start posts one request and cancel terminates active worker', () => { ... });
test('starting a new optimization terminates the previous worker', () => { ... });
test('stale completion from an old request is ignored', () => { ... });
test('completion terminates and clears the active worker', () => { ... });
test('dispose terminates an active worker', () => { ... });
```

The fake worker implements only `postMessage`, `terminate`, and assignable `onmessage/onerror`; assertions are against real client behavior, not mocked optimizer logic.

- [ ] **Step 2: Run and verify RED**

Run: `node --test tests/optimizer_client_js.test.mjs`

Expected: FAIL because `src/optimizerClient.js` does not exist.

- [ ] **Step 3: Implement worker protocol**

`optimizer.worker.js` selects `optimizeAngle`, `optimizeVelocity`, or `optimizeBoth`, forwards bounded progress, and posts one complete/error message carrying the original `requestId`.

- [ ] **Step 4: Implement `createOptimizerClient`**

Default `workerFactory` creates:

```javascript
new Worker(new URL('./optimizer.worker.js', import.meta.url), {type: 'module'})
```

Use a monotonically increasing request ID. Ignore any message whose request ID is not the active request. `cancel`/ `dispose` terminate synchronously.

- [ ] **Step 5: Verify lifecycle GREEN**

Run: `node --test tests/optimizer_client_js.test.mjs`

Expected: PASS.

- [ ] **Step 6: Replace synchronous React optimizer loops**

Delete `findOptimalAngle`, `findOptimalVelocity`, and `findOptimalBoth` from `TrajectorySimulator.jsx`.

Add state for:

- active mode / running boolean;
- progress;
- best diagnostic candidate;
- error.

Start handlers use `createOptimizerClient`; completion updates controls only when `result.solution` is non-null.

No-solution completion leaves velocity/angle unchanged and displays a concise status with the best near-miss classification.

- [ ] **Step 7: Add Cancel/progress UI and source contracts**

While running:

- disable all three start buttons;
- show progress text/bar;
- show `Cancel Optimization`;
- leave sliders and physics toggles enabled.

Add source-structure assertions that `TrajectorySimulator.jsx` imports/uses `createOptimizerClient`, contains `Cancel Optimization`, and no longer contains the old nested synchronous optimizer loops/functions.

- [ ] **Step 8: Verify Node suite and build**

Run:
- `npm run test:physics`
- `npm run build`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/optimizer.worker.js src/optimizerClient.js tests/optimizer_client_js.test.mjs src/TrajectorySimulator.jsx tests/ui_contract_js.test.mjs
git commit -m "feat: run optimizer in cancellable web worker"
```

### Task 6: Add Shared-Geometry 3-D Visualization and Update the 2-D HUB Drawing

**Files:**
- Create: `src/trajectory3dProjection.js`
- Create: `src/Trajectory3DView.jsx`
- Create: `tests/trajectory3d_js.test.mjs`
- Modify: `src/TrajectorySimulator.jsx`
- Modify: `tests/ui_contract_js.test.mjs`

**Interfaces:**
- Consumes: `samples3d`, `hubInteraction`, and Task 2 HUB geometry.
- Produces:
  - `CAMERA_PRESETS` with `isometric`, `front`, `side`, `top`;
  - `projectScene(points, camera, {width, height, padding})` returning finite 2-D projected points and scale data;
  - `Trajectory3DView({samples, hubGeometry, interaction, ballRadius})`.

- [ ] **Step 1: Write failing projection tests**

Pin:

- every preset projects a representative HUB + trajectory to finite coordinates;
- top view separates positive/negative lateral y;
- side view represents x/z while collapsing lateral y as expected;
- a degenerate scene with all trajectory samples at y=0 and nearly constant x still produces finite scale/coordinates.

- [ ] **Step 2: Run and verify RED**

Run: `node --test tests/trajectory3d_js.test.mjs`

Expected: FAIL because the projection module does not exist.

- [ ] **Step 3: Implement orthographic projection helpers**

Represent camera orientation as yaw/pitch. Rotate around the HUB center, drop depth, then fit all projected HUB vertices + trajectory points into the SVG bounds.

Clamp any near-zero projected extent to a small positive epsilon before scaling so flat/degenerate scenes cannot produce Infinity/NaN.

- [ ] **Step 4: Implement `Trajectory3DView`**

Render SVG:

- top/bottom hexagons;
- six funnel panel faces/edges;
- trajectory polyline;
- FUEL marker at selected sample;
- collision marker when present;
- optional dashed ball-center clearance hexagons;
- small ground axes.

Add buttons: `Isometric`, `Front`, `Side`, `Top`.

Add a native range input for trajectory sample selection.

Pointer drag updates yaw/pitch only in the free/isometric view. Preset buttons remain the accessible alternative; respect reduced-motion preferences by not adding continuous animation.

- [ ] **Step 5: Add 2-D/3-D view selector to the graph panel**

Default to 2-D.

When 3-D is selected, pass `result.samples3d`, the shared geometry, and `result.hubInteraction` into `Trajectory3DView`.

Update status copy to exactly one of:

- `CLEAN ENTRY`
- `RIM COLLISION`
- `FUNNEL COLLISION`
- `MISS`.

- [ ] **Step 6: Replace independent 2-D HUB literals/drawing**

Remove `funnelRadius` and `targetRadius` as independent geometry constants from `TrajectorySimulator.jsx`.

Generate the side-profile drawing from Task 2 geometry:

- top edges at `±topApothem, topZ`;
- bottom edges at `±bottomApothem, bottomZ`;
- optional center-clearance guides subtracting the ball radius.

Add a source test asserting `TrajectorySimulator.jsx` imports Task 2 geometry and does not contain the legacy `0.529`/`0.454` target geometry literals.

- [ ] **Step 7: Verify all Node tests and build**

Run:
- `npm run test:physics`
- `npm run build`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/trajectory3dProjection.js src/Trajectory3DView.jsx tests/trajectory3d_js.test.mjs src/TrajectorySimulator.jsx tests/ui_contract_js.test.mjs
git commit -m "feat: visualize shared HUB geometry in 3-D"
```

### Task 7: Documentation, Regression Gates, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/physics-regression.yml` only if its Node command does not already call `npm run test:physics`
- Modify: PR #2 description after final verification.

**Interfaces:**
- Consumes: complete Milestone 2.1 implementation.
- Produces: user-facing documentation and permanent CI coverage.

- [ ] **Step 1: Update README**

Document:

- Physics Options are interactive native controls;
- target outcomes are `clean-entry`, `rim-collision`, `funnel-collision`, and `miss`;
- HUB dimensions come from official 2026 FIRST funnel-panel/top-assembly drawings;
- optimizer runs in a cancellable worker and uses bounded coarse-to-fine search;
- final optimizer candidates are revalidated at `dt=0.001`;
- 3-D SVG view and camera presets;
- FUEL remains a rigid sphere for collision clearance and aerodynamics remain uncalibrated.

- [ ] **Step 2: Confirm CI executes all Node tests**

The workflow's Node job must run:

```text
npm ci
npm run test:physics
npm run build
```

Do not add a second overlapping workflow.

- [ ] **Step 3: Run full fresh local verification**

Run:

```bash
python -m unittest discover -s tests -v
npm run test:physics
npm run build
python scripts/generate_physics3d_fixtures.py
git diff --exit-code -- tests/fixtures/physics3d_golden.json
git diff --check
```

Expected: every command exits 0; the aerodynamic golden fixture is byte-identical after regeneration.

- [ ] **Step 4: Review the branch against the Milestone 2.1 spec**

Verify specifically:

- no decorative-only toggle remains;
- no synchronous optimizer grid remains in React;
- no optimizer accepts collision as success;
- no browser shot performs a second target integration;
- 2-D and 3-D target renderers import shared HUB geometry;
- worker cancellation terminates work;
- no Three.js/WebGL dependency was added;
- no flight-physics coefficient/equation changed.

- [ ] **Step 5: Update PR #2 only after exact-head CI is green**

PR description must call out:

- the user-reported toggle and freeze regressions;
- official HUB geometry / conservative rigid-sphere collision model;
- worker + cancel/progress behavior;
- 3-D view;
- unchanged aerodynamic calibration caveat.

Do not merge PR #2 automatically.

