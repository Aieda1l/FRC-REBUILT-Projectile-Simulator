# Extensible Game Pieces and Scoring

The flight engine should not know which FRC game is being played. A future shooting game is described by two independent pieces of data:

1. a **game piece** containing physical properties used by the flight model and a collision envelope used by scoring; and
2. a **scoring target** that chooses a registered scoring method and provides its geometry/point rules.

`src/gameProfiles.js` owns reusable game-piece and game-profile registries. `src/scoring.js` owns scoring-method registration and geometry evaluation. `src/trajectory2d.js` integrates the projectile exactly once, then asks the selected scorer to classify that same 3-D trajectory.

## Built-in scoring methods

### `2026-hex-hub`

This is the existing detailed REBUILT HUB model. It preserves the current regular-hex funnel geometry and the `clean-entry`, `rim-collision`, `funnel-collision`, and `miss` classifications.

### `top-circle`

Use this for a horizontal circular opening such as a basket or hub. Configuration:

```json
{
  "kind": "top-circle",
  "height": 2.64,
  "centerX": 0,
  "centerY": 0,
  "openingRadius": 0.61,
  "crossingDirection": -1,
  "points": {"auto": 4, "teleop": 2, "default": 2}
}
```

The scorer finds the requested height-plane crossing, shrinks the opening by the piece collision radius, and reports clean entry, rim collision, or miss.

### `plane-aperture`

Use this for a goal opening in an arbitrary plane, including wall goals, angled openings, rectangular slots, or circular ports.

```json
{
  "kind": "plane-aperture",
  "planePoint": [0, 0, 2],
  "planeNormal": [-1, 0, 0],
  "apertureCenter": [0, 0, 2],
  "up": [0, 0, 1],
  "crossingDirection": -1,
  "shape": "rectangle",
  "width": 1.05,
  "height": 0.30,
  "points": {"default": 2, "amplified": 5}
}
```

For a circular opening, set `shape` to `circle` and provide `openingRadius` instead of `width`/`height`.

`planeNormal` controls which side of the plane is positive. `crossingDirection: -1` means the piece must move from the positive side to the negative side; `1` means the reverse; `0` accepts either crossing.

## Add a new game piece/profile without changing simulation code

Game-profile JSON uses schema `frc-shooting-game-v1`:

```json
{
  "schema": "frc-shooting-game-v1",
  "id": "frc-future-game",
  "name": "Future FRC Shooting Game",
  "gamePiece": {
    "id": "future-piece",
    "name": "Future Piece",
    "mass": 0.22,
    "radius": 0.09,
    "collisionRadius": 0.10,
    "dragCoeff": 0.47,
    "liftCoeff": 0.20
  },
  "scoring": {
    "kind": "top-circle",
    "height": 2.5,
    "openingRadius": 0.6,
    "points": {"auto": 4, "teleop": 2}
  }
}
```

Then apply it to any existing launch/environment parameters:

```js
import {applyGameProfile, parseGameProfile} from './gameProfiles.js';
import {simulateShot} from './trajectory2d.js';

const profile = parseGameProfile(profileJsonText);
const params = applyGameProfile(baseParams, profile, {phase: 'teleop'});
const result = simulateShot(params);

console.log(result.scoringInteraction.classification);
console.log(result.score);
```

`collisionRadius` is intentionally separate from the aerodynamic `radius`. That lets future non-spherical pieces use a conservative scoring envelope without pretending the same dimension is a complete aerodynamic model.

Point values are also separate from geometry. A `points` object can select values by `scoringContext.phase` (`auto`, `teleop`) or `scoringContext.variant` (for example `amplified`) without changing whether the trajectory physically entered the goal.

## Add a genuinely new scoring method

If a future game cannot be represented by a circular opening, an arbitrary plane aperture, or the 2026 funnel, register a new scorer:

```js
import {registerScoringMethod} from './scoring.js';

registerScoringMethod('double-ring', ({samples, target, piece, context}) => {
  // Evaluate the already-integrated samples. Do not integrate a second trajectory.
  // Return a normalized classification/status and optional geometry diagnostics.
  return {
    classification: 'clean-entry',
    status: 'scored',
    isScore: true,
    clearanceMargin: 0.02,
    entrySample: samples.at(-1),
  };
});
```

Scorers should return `isScore`, `status` (`scored`, `collision`, or `miss`), a human-readable `classification`, and useful geometry diagnostics such as `clearanceMargin`, `missDistance`, `entrySample`, and `collisionPoint`. The optimizer and uncertainty evaluator consume the normalized score state rather than being tied to one target shape.

## Why these abstractions fit past FRC games

Past FRC shooting games repeat a few geometry/rules patterns:

- **2022 RAPID REACT** used upper and lower funnel-shaped HUB goals, and the same physical score was worth different points in autonomous and teleop. That motivates a top-opening scorer plus point values that are evaluated from match context instead of baked into geometry.
- **2024 CRESCENDO** scored NOTES through the SPEAKER opening and could temporarily amplify SPEAKER value. That motivates a plane-aperture scorer plus a separate scoring variant/context.
- **2013 ULTIMATE ASCENT** used DISCS crossing goal openings, and **2016 FIRST STRONGHOLD** used BOULDERS shot into high/low goals. These are naturally represented as plane apertures with piece-clearance checks.
- **2026 REBUILT** needs more than a single plane check because the FUEL must clear the regular-hex funnel panels after entering. That remains a specialized registered scorer rather than forcing every game into a funnel model.

FIRST keeps manuals and field documentation for these seasons on the [Archived Game Documentation](https://www.firstinspires.org/resources/library/frc/archived-games) page. New built-in profiles should cite the applicable official manual/drawings and should clearly mark aerodynamic coefficients as measured, estimated, or uncalibrated.

## Compatibility

`simulateShot()` now returns `scoringTarget`, `scoringGeometry`, `scoringInteraction`, and `score`. The existing `hubInteraction`, `hubGeometry`, and `hitTarget` fields remain available so the current 2026 UI and callers continue to work while they migrate to the generic names.
