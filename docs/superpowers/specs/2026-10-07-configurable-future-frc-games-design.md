# Configurable FRC game pieces and scoring targets — design
Date: 2026-10-07

## Problem and scope

The existing 2026 REBUILT simulator hard-codes a 0.215 kg spherical FUEL and the 2026 hexagonal HUB in its React state and trajectory classifier. That makes it unsuitable for testing other seasons, new prototype games, or alternative slot / hoop geometry. The simulator must provide an editable, durable (browser-local) library of shooter projectile profiles and targets, and the active choices must affect simulation, optimizer, uncertainty analysis, and both plots.

This work intentionally models **one selected projectile and one selected scoring opening** at a time, not an entire FRC field or season-specific match-scoring rules. Configurable `points` is a simple successful-shot value; autonomous multipliers, scoring states, human players, bounces, and multi-target combinations are out of scope.

## FRC historical research and design decisions

Official manuals and archives:
- [2006 AIM HIGH game manual](https://firstfrc.blob.core.windows.net/frcarchive/2006/2006-game-manual.pdf): center elevated goal worth 3 points and corner goals worth 1; establishes why different openings and point values should be independently editable.
- [2012 REBOUND RUMBLE](https://www.firstinspires.org/resources/library/frc/archived-games): foam basketballs shot into hoops at several elevations; motivate horizontal circular rim target.
- [2013 ULTIMATE ASCENT manual](https://studylib.net/doc/8396371/frc-2013-game-manual): discs scored in vertical rectangular openings, including high 54 in. × 12 in. goal; motivates rectangular slot and non-spherical placeholder profile.
- [2016 STRONGHOLD / historical game overview](https://www.teamargos.org/docs/about_doc/First/FRC%20Games/): towers with high and low ball goals reinforce variation in height, geometry, and size.
- [2017 STEAMWORKS official manual](https://firstfrc.blob.core.windows.net/frc2017/Manual/HTML/2017FRCGameSeasonManual.htm): 5 in. nominal, ~74 g balls, 21.5 in. high boiler opening and 25 in. × 8.75 in. low boiler opening; motivates hoop plus rectangular low opening.
- [2020 INFINITE RECHARGE official manual](https://firstfrc.blob.core.windows.net/frc2020/Manual/HTML/2020FRCGameSeasonManual.htm): 7 in. POWER CELL, bottom rectangular port, hexagonal outer port, and a 13 in. circular inner port; motivates vertical circle and rectangle. Hexagonal *wall* openings and sequential outer→inner interference are future extensions.
- [2022 RAPID REACT official manual](https://firstfrc.blob.core.windows.net/frc2022/Manual/HTML/2022FRCGameManual.htm): 9.5 in., ~270 g CARGO and concentric 4 ft upper / ~5 ft lower horizontal hub openings at different elevations; motivates adjustable hoop size and height.
- [2024 CRESCENDO field layout](https://www.frcmanual.com/2024/arena): NOTE ring launched into SPEAKER or placed in AMP. These presets are **illustrative** rather than certified geometry because the current solver has no disc/ring orientation model.
- [2026 REBUILT manual](https://www.firstinspires.org/resources/library/frc/season-materials): 5.91 in., 0.203–0.227 kg FUEL and elevated hexagonal funnel HUB. The existing classifier and dimensions remain the default regression reference.

### Model

`gameCatalog.js` provides built-in season-inspired presets, independent editable game-piece and scoring-target profiles, schema v1 JSON persistence, validation and load fallback. Game piece fields: name, sphere/disc/ring visualization category, mass, diameter, Cd and Cl. Target fields: name, point value, 3-D location (X, lateral Y, height) and shape-dependent opening measurements. All physical units are meters/kilograms; projectile drag/lift values for historical presets are **uncalibrated placeholders**.

`scoringTargets.js` normalizes targets and classifies simulated 3-D samples:
- HUB: reuse existing `classifyHubInteraction` (descending top crossing, rim and all six funnel panels).
- Hoop: descending passage through horizontal plane; radial ball-radius clearance.
- Wall round-slot: forward (+X) crossing of a vertical circular opening.
- Wall slot: forward (+X) crossing of a rectangular opening; min edge clearance.
- All: return existing `clean-entry`, `rim-collision`, or `miss` labels, with clearance/miss distance and crossing/collision samples. Compatible with existing ranking and uncertainty analysis.
- Model ball as sphere for *all* classes; non-spherical shapes are clearly labeled approximate. A future full rigid-body model can extend this interface.

### UI and persistence

`GameSetupPanel.jsx` provides preset selection, New, Customize, Duplicate, Edit and Delete workflows for pieces and targets; validates submitted physical parameters and shows storage/quota failures. Built-in presets are immutable; saving a modification to one creates a custom profile. Browser `localStorage` keeps edits and active selections on the device, not synchronized between browsers/accounts.

The main simulator owns current selection. Changed mass/radius/coefficients flow through the existing RK flight solver; changed geometry is classified by `simulateShot` and is passed to the optimizer worker, robust uncertainty Monte Carlo, 2-D graph and 3-D wireframe graph. A points-per-shot result is displayed on a successful entry.

### Compatibility and limitations

Default legacy `targetX/targetLateralY` without `target` continues using the same 2026 HUB geometry. Keep `hubInteraction` as a compatibility field so previous optimizer code and tests work while the geometry becomes generic.

Aperture-only classifications do **not** simulate backboard collision, angle of disc/note orientation, net catches, rim bounce, motion transfer, multi-goal scoring state, or legal scoring subtleties. Preset point values are simple editable defaults and **not** complete season rule implementations. Users must measure dimensions, mass, and aerodynamic properties for engineering-grade predictions.
