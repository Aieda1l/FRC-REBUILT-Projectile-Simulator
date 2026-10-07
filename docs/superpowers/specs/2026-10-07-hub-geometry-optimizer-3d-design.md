# Milestone 2.1: HUB Geometry, Responsive Optimizer, and 3-D Visualization

**Date:** 2026-10-07  
**Repository:** `Aieda1l/FRC-REBUILT-Projectile-Simulator`  
**Branch / PR:** `milestone-2-rk45-3d-engine` / PR #2  
**Builds on:** `docs/superpowers/specs/2026-10-07-rk45-3d-engine-design.md`

## Purpose

Milestone 2.1 fixes three user-visible problems found while testing Milestone 2:

1. the Physics Options switches look interactive but cannot be toggled;
2. optimizer buttons can freeze the browser because thousands of RK4 simulations run synchronously on the React main thread;
3. the simulator can report a target hit even when the displayed trajectory clips through the HUB side, because scoring currently checks only one horizontal plane instead of the funnel geometry.

The milestone also adds an interactive 3-D trajectory/HUB view. The 2-D and 3-D views, hit classification, and optimizer must use one shared HUB geometry definition so the visualization cannot silently disagree with scoring.

The aerodynamic model is unchanged. This milestone improves target geometry, optimization execution, and visualization; it does not invent new FUEL coefficients.

## Source Geometry

Use official FIRST 2026 materials as the geometry basis:

- The 2026 REBUILT Game Manual §5.4 states that the HUB top has a **41.7 in hexagonal opening** and that the **front edge of the opening is 72 in above the carpet**.
- FIRST's TE-26300 HUB Team Test Element documentation states that its six `GE-26329 Hub Funnel Side` polycarbonate panels are the **same panels used on the real field**.
- Drawing `TE-26302 HUB TOP ASSEMBLY` gives the assembled top opening across-flats dimension as **41.727 in**.
- Drawing `GE-26329 HUB FUNNEL SIDE` gives:
  - top side length: **24.09 in**;
  - bottom side length: **18.92 in**;
  - vertical panel height: **17.90 in**;
  - side angle: **73.9°**.

For computation, use the drawing value 41.727 in where it provides more precision than the rounded manual value. The displayed documentation may continue saying 41.7 in.

The regular-hex bottom opening across-flats is derived from the panel bottom side:

```text
bottom_across_flats = sqrt(3) * 18.92 in ~= 32.77 in
```

The top opening plane is:

```text
z_top = 72.0 in = 1.8288 m
```

The modeled bottom edge of the funnel panels is:

```text
z_bottom = z_top - 17.90 in ~= 54.10 in = 1.37414 m
```

This is a collision model of the six real-field funnel panels, not a rigid-body model of every fastener, light bar, frame member, net, or downstream HUB component.

## Coordinate Convention

Retain the Milestone 2 right-handed coordinate system:

- x = forward/downrange;
- y = left lateral;
- z = up.

The HUB is centered at `(targetX, targetYHorizontal)`, with the existing browser default at `(0, 0)`. One flat face of the regular hexagon faces the shooter at negative x.

Existing 2-D launches remain in the x/z plane with y = 0.

## Shared Browser HUB Geometry

Create `src/hubGeometry.js` as the single browser source of truth for:

- dimensional constants;
- top and bottom hex vertices;
- horizontal side normals;
- frustum side-plane equations;
- top/bottom opening clearance;
- trajectory/HUB interaction classification;
- geometry data consumed by both 2-D and 3-D renderers.

No HUB dimensions may remain duplicated as independent literals in `TrajectorySimulator.jsx`, `trajectory2d.js`, the optimizer, or the 3-D renderer.

### Frustum model

Let:

```text
a_top = top_across_flats / 2
a_bottom = bottom_across_flats / 2
k = (a_top - a_bottom) / (z_top - z_bottom)
a(z) = a_bottom + k * (z - z_bottom)
```

For each of the six outward horizontal unit normals `n_i`, the inside of the funnel is:

```text
n_i dot [x, y] <= a(z)
```

For a spherical FUEL center with radius `r`, perpendicular clearance to a sloped panel is:

```text
panel_clearance_i =
    (a(z) - n_i dot [x, y]) / sqrt(1 + k^2) - r
```

A negative clearance means the FUEL sphere intersects that panel.

At the top and bottom polygon edges, use horizontal regular-hex clearance:

```text
edge_clearance = min_i(a_plane - n_i dot [x, y]) - r
```

This prevents accepting a center trajectory that enters through the mathematical polygon while the physical ball overlaps its rim.

## Swept-Ball Interaction Classification

Create a pure function with semantics equivalent to:

```javascript
classifyHubInteraction(samples, hubGeometry, ballRadius)
```

It scans the already-integrated trajectory once. It must not run a second flight integration merely to evaluate the HUB.

Possible results:

- `clean-entry`
- `rim-collision`
- `funnel-collision`
- `miss`

The returned object contains at least:

- `classification`
- `topCrossing` or null
- `bottomCrossing` or null
- `collisionPoint` or null
- `clearanceMargin` in meters
- `missDistance` in meters when applicable.

### Top crossing

Find the first descending segment that crosses `z_top` and linearly interpolate the full state at the crossing.

At the top crossing, let `raw_edge_distance = min_i(a_top - n_i dot [x,y])` and `edge_clearance = raw_edge_distance - ballRadius`:

- if `edge_clearance >= 0`, entry is geometrically valid;
- if `edge_clearance < 0` but the sphere overlaps/touches the top polygon boundary (its minimum Euclidean distance to a top hex edge is <= `ballRadius`), classify `rim-collision`;
- if the sphere is fully separated from the top opening/rim, classify `miss`.

This distinction handles both centers just inside an edge without enough ball clearance and centers just outside the polygon whose sphere still strikes the rim.

### Funnel passage

After valid top entry, inspect every trajectory segment overlapping `[z_bottom, z_top]`.

For each clipped segment, panel clearance is affine along that straight segment for each side plane, so checking both clipped segment endpoints is sufficient to find the minimum clearance for that segment/plane.

If any panel clearance becomes negative, classify `funnel-collision` and linearly interpolate the first zero-clearance location for visualization.

### Bottom crossing

A `clean-entry` requires:

1. a valid descending top entry;
2. no panel collision;
3. a descending crossing of `z_bottom`;
4. full ball-radius clearance through the bottom hex edge.

If the sphere intersects the bottom rim, classify `funnel-collision`.

This definition is intentionally conservative: a ball that physically strikes a funnel panel and might bounce into the HUB is not an optimizer-valid clean shot.

## Browser Simulation API

Refactor `src/trajectory2d.js` so one integration produces both visualization data and HUB scoring.

Add a browser-facing pure simulation function equivalent to:

```javascript
simulateShot(params, options)
```

It returns:

- canonical `samples3d`;
- projected 2-D `points`;
- `hubInteraction`;
- compatibility fields `hitTarget`, `impactPoint`, `flightTime`, `maxHeight`, `range`, `entryVelocity`, `entryAngle`.

`hitTarget` is true **only** when `hubInteraction.classification === "clean-entry"`.

The existing `simulateTrajectory2D` may remain as a compatibility wrapper over `simulateShot`, but there must no longer be a ground integration plus a second target-plane integration for each candidate.

Simulation options include an explicit `dt` so optimizer coarse passes can trade precision for speed while final/UI trajectories retain `dt=0.001`.

## Optimizer Architecture

Move optimization out of `TrajectorySimulator.jsx`.

### Pure optimizer module

Create `src/optimizer.js` containing deterministic pure search functions. It may call `simulateShot`, but has no React or Worker dependencies.

All optimizers use a deterministic lexicographic ranking:

1. classification rank: `clean-entry = 3`, `rim-collision/funnel-collision = 2`, `miss = 1`;
2. for `clean-entry`, larger positive `clearanceMargin` wins;
3. for collisions, larger (less-negative) `clearanceMargin` wins;
4. for misses, smaller `missDistance` wins;
5. exact ties prefer the candidate closest to the requested/current velocity and/or angle so results are stable.

No optimizer may return `rim-collision` or `funnel-collision` as a successful solution. If no clean entry exists in the bounded search, it reports no solution while still exposing the best near-miss for diagnostics/progress.

### Coarse-to-fine search

Angle-only:

- coarse: 5°..85°, 2° step, `dt=0.005`;
- refine around the best candidates with <=0.25° steps and `dt<=0.002`;
- final candidate validation at `dt=0.001`.

Velocity-only:

- coarse: 5..25 m/s, 1 m/s step, `dt=0.005`;
- refine around the best candidates with <=0.2 m/s steps and `dt<=0.002`;
- final candidate validation at `dt=0.001`.

Velocity + angle:

- coarse grid: velocity 5..25 m/s by 1 m/s and angle 20°..80° by 2°;
- retain only the best bounded set of candidates (maximum 12) for refinement;
- refine locally rather than resweeping the full grid;
- final result is revalidated at `dt=0.001`.

Exact refinement spacing may be adjusted during implementation if tests show a smaller deterministic search achieves equal or better clearance, but the full old 2,501-candidate high-precision sweep must not return.

## Web Worker Execution

Create `src/optimizer.worker.js` as a thin message wrapper around `src/optimizer.js`.

The React main thread:

1. creates a module Worker when an optimization begins;
2. posts the current simulation parameters plus optimization mode;
3. receives progress and the final result;
4. terminates the worker after completion or cancellation.

The worker posts progress at bounded intervals with:

- candidates evaluated;
- total or estimated total candidates;
- current best candidate;
- current best classification/clearance.

Cancellation uses `worker.terminate()` from the main thread rather than relying on the busy worker to process a cancellation message.

Only one optimization may run at a time. While active:

- all three optimizer start buttons are disabled;
- a visible **Cancel Optimization** button is enabled;
- a progress indicator is shown;
- sliders and physics toggles remain interactive;
- changing parameters does not mutate the already-running worker input; the user may cancel and restart to optimize the new settings.

Unmounting the simulator terminates an active worker.

## Physics Toggle Fix

The current `Toggle` component renders only styled `div` elements inside a label and never contains an input or invokes its `onChange` prop. That is the root cause of the reported bug.

Replace it with an accessible native checkbox:

```jsx
<label>
  <input
    type="checkbox"
    checked={checked}
    onChange={(event) => onChange(event.target.checked)}
    ...
  />
  ...
</label>
```

The visible switch remains styled, but the native input owns interaction, keyboard focus, checked state, and accessibility semantics.

Acceptance behavior:

- clicking the switch changes the toggle;
- clicking the text label changes the toggle;
- keyboard Space changes the focused toggle;
- Air Drag changes the displayed trajectory;
- Magnus Effect changes the displayed trajectory for nonzero spin;
- Show Ideal hides/shows the ideal curve;
- Show Error Envelope hides/shows envelope paths.

## 3-D Visualization

Add `src/Trajectory3DView.jsx`.

Do not add Three.js or another 3-D runtime dependency. The scene is small enough for an SVG renderer with a lightweight camera transform.

### Scene contents

Render:

- the actual top and bottom HUB hexagons from `hubGeometry.js`;
- six funnel panel faces/wireframe edges;
- the full `samples3d` trajectory polyline;
- a FUEL marker at a selectable trajectory sample;
- a collision marker when classification is `rim-collision` or `funnel-collision`;
- the top and bottom ball-center clearance hexagons as optional dashed guides;
- ground reference axes near the HUB.

The displayed geometry must be generated from the exact same constants and vertex functions used by collision classification.

### Camera interaction

Provide:

- Isometric;
- Front;
- Side;
- Top

preset buttons.

Pointer drag may orbit yaw/pitch in 3-D mode. Preset buttons are the non-drag accessible alternative.

Use an orthographic projection to avoid perspective distortion confusing clearance judgments.

A trajectory-position slider scrubs the displayed FUEL marker through the returned samples.

### UI integration

The graph panel gets a `2-D / 3-D` view selector. Preserve the current 2-D view as the default.

The 2-D HUB drawing must also be generated from the shared HUB geometry cross-section instead of the current independent `funnelRadius` / `targetRadius` schematic.

The hit status becomes classification-aware:

- `CLEAN ENTRY`
- `RIM COLLISION`
- `FUNNEL COLLISION`
- `MISS`

A collision must never render the green `TARGET HIT` state.

## Performance Requirements

The UI must not execute optimizer search loops on the main thread.

The normal single-trajectory render stays synchronous and should remain fast enough for slider updates.

The worker search is bounded:

- no unbounded loops;
- no more than the declared coarse grid plus local refinement candidates;
- progress is posted periodically rather than for every candidate;
- worker is terminated after result/cancel.

The error-envelope calculation remains four trajectories and may stay on the main thread.

## Testing

Continue using Node's built-in `node:test` for pure modules.

### HUB geometry tests

Add tests for:

- official dimensional constants/conversions;
- top/bottom regular-hex vertices;
- centerline vertical shot classifies clean;
- top edge overlap classifies rim collision;
- trajectory entering top opening but intersecting a sloped panel classifies funnel collision;
- trajectory outside by more than one ball radius classifies miss;
- a trajectory that visually used to clip a funnel side cannot return `hitTarget=true`;
- positive clean-entry clearance and negative collision clearance.

### Simulation adapter tests

Add tests for:

- only one integration path is needed for HUB classification;
- `hitTarget` is true only for `clean-entry`;
- optimizer `dt` override does not change the UI default `dt=0.001`;
- 2-D projected points correspond to x/z from the same `samples3d`.

### Optimizer tests

Add deterministic tests for:

- candidate ranking always prefers clean entry to collision/miss;
- larger clean-entry clearance wins;
- angle-only optimizer returns a clean candidate for a known solvable fixture;
- velocity-only optimizer returns a clean candidate;
- combined optimizer returns a clean candidate;
- coarse/refine candidate count is bounded well below the old exhaustive high-precision search;
- final returned candidate is revalidated at `dt=0.001`.

### Toggle regression

Because the project does not currently carry a browser DOM test framework, add a lightweight source-structure regression that verifies the shared Toggle component includes a native checkbox wired to its supplied change handler, and keep Vite build as the JSX/compiler gate.

Do not add a large UI-testing dependency only for this single regression.

### Build/CI

Required final checks:

```text
python -m unittest discover -s tests -v
npm run test:physics
npm run build
```

The existing cross-language physics parity fixtures must remain unchanged unless the underlying flight physics changes; target geometry changes alone must not require regenerating aerodynamic golden values.

## Documentation

Update README to explain:

- 3-D view and camera presets;
- clean-entry vs collision/miss classifications;
- optimizer now runs off the UI thread;
- optimizer uses official funnel-panel geometry but remains limited by uncalibrated FUEL aerodynamics;
- a real FUEL can deform/bounce, while the simulator intentionally treats it as a rigid sphere for conservative collision clearance.

## Non-Goals

Milestone 2.1 does not include:

- bounce/rebound simulation after panel contact;
- compliant/deformable FUEL contact mechanics;
- DMX light-bar/fastener collision detail;
- downstream internal HUB/exits;
- lateral aiming/yaw controls in the existing 2-D launcher UI;
- robot pose/heading transforms;
- new FUEL aerodynamic calibration;
- replacing the SVG 3-D view with a full WebGL engine.

## Acceptance Checklist

- [ ] Physics Options are operable by mouse/touch and keyboard.
- [ ] Air Drag and Magnus toggles actually change simulation parameters/results.
- [ ] The browser integrates each displayed/optimized candidate once, not separately for ground and target.
- [ ] Shared official-dimension HUB geometry drives both scoring and rendering.
- [ ] Rim/funnel clipping cannot be labeled as a hit.
- [ ] `hitTarget` means `clean-entry` only.
- [ ] Optimizer work runs in a Web Worker rather than the React main thread.
- [ ] Optimization can be cancelled and exposes progress.
- [ ] Combined optimizer uses bounded coarse-to-fine search rather than the old exhaustive high-precision grid.
- [ ] Optimizer final results are revalidated at `dt=0.001`.
- [ ] 3-D view renders trajectory, HUB frustum, FUEL marker, and collision marker from shared geometry.
- [ ] 3-D view has Isometric/Front/Side/Top presets.
- [ ] Existing 2-D view remains available and uses the same HUB geometry.
- [ ] Existing Python and JS flight-physics tests/parity remain green.
- [ ] Vite production build passes.

## References

- FIRST Robotics Competition, 2026 Season Materials / Playing Field official assets.
- FIRST Robotics Competition, 2026 REBUILT Game Manual, §5.4 HUB.
- FIRST Robotics Competition, TE-26300 HUB Team Test Element Build Instructions, Version 2.
- FIRST drawing TE-26302, HUB TOP ASSEMBLY.
- FIRST drawing GE-26329, HUB FUNNEL SIDE.
