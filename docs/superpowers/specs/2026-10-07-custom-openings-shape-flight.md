# Freeform targets, shape visualization and preliminary 6-DoF flight
2026-10-07 · Superpowers design extension

## Requirements and scope
Continue the configurable FRC game library with (1) a clickable/drag-editable freeform scoring aperture (vertical wall or horizontal hoop plane), (2) properly scaled 2-D/3-D game pieces with clearance feedback at opening passage, (3) an initial orientation-aware rigid-body flight integrator for discs and ring-shaped game pieces. Preserve spherical ball engine and saved-profile schema backward compatibility.

## Design decisions

1. **Polygon editor** (`PolygonOpeningEditor.jsx`): Browser-SVG pointer interactions to add and drag up to 64 vertices. Users may also enter precise local planar coordinates in meters and set grid snap. Geometry uses local U,V coordinates in an opening's selected vertical (lateral/height) or horizontal (forward/lateral) plane, with target position X/lateral/Z set separately. Validation rejects zero area, duplicate/coincident vertices, bad coordinates, and self-intersection. Concave simple polygons are supported. The normalized vertices are saved in version-1 local storage records. Closed-area editing is exposed through `GameSetupPanel`.

2. **Scoring and collision** (`polygonGeometry.js`, `scoringTargets.js`): A projectile must cross the opening plane in the correct direction. First find an interpolated trajectory sample at the plane. Ray casting determines if the center is inside the outline and minimum segment distance determines the nearest margin. A ball scores when center-to-edge clearance exceeds its radius. Disc and ring targets use projected **rigid cylindrical envelope support** along the nearest boundary normal as an initial orientation-aware clearance estimate. On rectangular and circular goals use boundary-specific support. A path through solid frame is a collision, farther outside is a miss. The 2026 hexagonal HUB remains a legacy ball-radius classifier; for flat pieces it is a conservative approximation.

3. **Visualization** (`gamePieceGeometry.js`, trajectory views): Common 3-D world-space wireframes represent sphere, disc, and annular ring with finite thickness. Rotate meshes with simulation quaternion and translate to true world positions; project consistently into side-plane SVG and the 3-D camera scene. Show timeline scrubber, goal-crossing jump, and edge margin in centimeters. Rendering is line-based for performance and does not yet simulate bounce/rebound.

4. **Rigid-body flight** (`rigidBodyFlight.js`): A new RK4 integrator carries 3-D position, velocity, unit quaternion, and 3-D body angular velocity (13 stored numbers, 6 physical degrees of freedom). Compute body-to-world orientation, disc normal, air-relative velocity, angle of attack, lift, drag and simplified restoring pitching torque. Apply annulus/disk inertia tensor and Euler's rotational equations, with optional damping. Model constant Cd base plus quadratic angle-dependent drag; constant Cl base plus linear angle-dependent lift; a linear pitching-moment slope, all independently user-editable and **uncalibrated**. Calibrated FUEL aerodynamics remain restricted to the ball engine. Use spin RPM as body axial spin for disc/ring, not wheel backspin. Crossing samples retain quaternion and normal.

5. **Persistence**: Old browser-local saved profiles remain readable; missing new coefficients get defaults during validation. New custom polygon targets persist with their planar vertices. Built-in presets remain immutable.

## Scientific basis and warnings

- Kamaruddin, Potts & Crowther (2018), *Aerodynamic performance of flying discs*, Aircraft Engineering and Aerospace Technology 90(2), 390–397. Wind-tunnel lift, drag, and moment coefficients plus full 6-DoF numerical trajectories; emphasizes both lift/drag ratio and pitching-moment slope. https://doi.org/10.1108/AEAT-09-2016-0143
- Hubbard & Hummel (2000), *Simulation of Frisbee Flight*. Nonlinear rigid-body flight and coefficient identification from measured marker trajectories. https://research.engineering.ucdavis.edu/biosport/sample-page/test-page-1/frisbee-flight-simulation-and-throw-biomechanics/
- Neither publication establishes applicable coefficients for FRC ULTIMATE ASCENT discs nor soft CRESCENDO NOTES; the app **does not claim validated aerodynamic predictions** for either. Drag/lift coefficients need experimental trajectory fitting per game piece.
- The model deliberately excludes foam flex/deformation, unsteady wobble, sophisticated ring aerodynamics, turbulent coefficients, landing collisions, launch-wheel transfer, and fully faithful goal capture.
- Orientation-aware clearance for a rigid cylinder is approximate for an arbitrary concave polygon near corners and for a hexagonal funnel. Goal plane crossing alone is not equivalent to a season-rule scoring determination.
- For projectiles requiring full geometry contact resolution, use a later continuous collision-detection and rigid/soft-body simulation upgrade.

## Verification
- `tests/custom_openings_rigidbody_js.test.mjs`: polygon validity, concave shape, self-crossing rejection, storage round-trip, forward/downward crossing, disc inertia/attitude, true-scale sphere/ring mesh, and orientation-dependent edge clearance.
- Existing `tests/*_js.test.mjs` maintain REBUILT ball regressions and optimizer contract.
- GitHub Actions runs `npm run test:physics`, `npm run lint`, `npm run build`, and Python unittest suite.

## Follow-up work
- Multi-plane, curved, angled, or extruded goal meshes; import SVG paths (with safe simplification); nonconvex edge-to-cylinder continuous collision; optional 3-D solid renderer.
- Experimental Cd(α), Cl(α), Cm(α), roll/yaw moment tables, measured inertial moments, frame-by-frame orientation calibration, and deformable foam-ring dynamics.
- Browser E2E accessibility and detailed click/drag interaction tests before release.
