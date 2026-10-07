# Extensible FRC Game Profiles and Scoring Design

## Goal

Make projectile integration independent of the current FRC game so a future season can supply a new game piece and target/scoring definition without duplicating the physics engine.

## Design constraints

- Preserve the current 2026 HUB behavior and public compatibility fields.
- Integrate each shot once; scoring consumes canonical 3-D samples.
- Keep physical entry/collision separate from match-dependent point values.
- Let non-spherical pieces use a conservative collision envelope separate from the aerodynamic radius.
- Keep scorer registration code-based while allowing common target geometries to be configured with JSON.
- Keep optimizer and Monte Carlo analysis compatible with any registered scorer.

## Components

### Game piece registry

`src/gameProfiles.js` stores physical definitions keyed by stable IDs. Required fields are mass, aerodynamic radius, scalar fallback drag/lift coefficients, and a display name. `collisionRadius` defaults to `radius` but can differ.

### Game profile registry

A `frc-shooting-game-v1` profile combines one piece with one scoring target. Profiles can reference a registered piece by ID or contain an inline piece. JSON profiles are validated before use.

### Scoring registry

`src/scoring.js` maps `target.kind` to a pure evaluator. Evaluators receive canonical samples, the target config, piece geometry, and optional scoring context. Built-ins are:

- `2026-hex-hub`: current detailed funnel model;
- `top-circle`: descending/ascending crossing of a horizontal circular opening;
- `plane-aperture`: arbitrary 3-D scoring plane with rectangular or circular aperture.

A future target with new collision rules can register another evaluator without editing `trajectory2d.js`.

### Normalized interaction

Every scorer returns:

- `isScore`
- `status`: `scored`, `collision`, or `miss`
- `classification`
- `points`
- `scoreRank`
- `clearanceMargin`
- `missDistance`
- optional `entrySample`, `collisionPoint`, and `geometry`

The 2026 scorer retains its legacy classification strings. `simulateShot` also exposes the generic interaction through the deprecated `hubInteraction` alias during migration.

## Historical inspiration

- RAPID REACT (2022): upper/lower HUBs and phase-dependent cargo values -> target geometry plus contextual point table.
- CRESCENDO (2024): SPEAKER aperture and amplification -> arbitrary plane aperture plus scoring variant.
- ULTIMATE ASCENT (2013): DISCS crossing goal openings -> plane aperture.
- FIRST STRONGHOLD (2016): high/low BOULDER goals -> plane aperture with spherical clearance.
- REBUILT (2026): funnel-panel clearance after top entry -> specialized scorer.

## Extension path for a new season

1. Measure/enter the new piece mass and dimensions; calibrate aerodynamics rather than copying coefficients from another piece.
2. Choose a built-in target geometry or register a new scoring method.
3. Create a versioned game profile and point table.
4. Run deterministic scorer tests at the target boundaries.
5. Run optimizer/uncertainty tests and compare against official field drawings.
