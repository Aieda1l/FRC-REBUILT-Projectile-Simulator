# FUEL Calibration and Validation Guide

This guide describes a measurement-first workflow for replacing the simulator's uncalibrated FUEL baseline with a profile fitted to your robot, your launcher, and the FUEL population you actually use.

## What calibration can and cannot fix

The numerical integrator is already much more precise than typical FRC measurement error. Calibration therefore focuses on the uncertain parts of the physical model: launch state, drag, Magnus lift, and—only when directly measured—spin decay. The flight engine also includes buoyancy by default from ball volume, air density, and gravity.

A calibration profile does **not** make the simulator a CFD model. It also does not model foam deformation, favorable rim bounce, panel friction, or internal HUB interactions. The scoring model remains intentionally conservative: clean entries count; modeled rim or funnel contact does not.

Keep these error sources separate:

- **Numerical integration error** comes from the time-stepping solver. At the normal 1 ms browser step it should be small relative to shot-to-shot variation.
- **Model-form uncertainty** comes from using simplified drag/lift laws instead of full fluid dynamics.
- **Fitted-parameter uncertainty** comes from finite/noisy calibration data.
- **Launch repeatability** is the real robot's variation in exit speed, angle, spin, and robot velocity.
- **Collision-model simplification** comes from treating FUEL as a rigid sphere and rejecting contact-heavy trajectories.

## Recommended measurement setup

Use a high-frame-rate camera whenever possible. 240 fps is useful; higher frame rates are better for spin measurement. Keep the camera fixed, minimize perspective distortion, and place a known-length scale in the plane of motion.

For planar shots, place the camera approximately perpendicular to the trajectory plane. A long focal length and greater camera distance reduce perspective error. For 3-D shots, use two synchronized views or a calibrated motion-capture setup.

Mark the ball with high-contrast dots or short tape marks that remain visible for multiple frames. Do not significantly change the surface texture or mass distribution.

Record the launcher configuration with each dataset: wheel type, wheel spacing, hood position, compression, nominal motor speed, and any configuration that can change launch conditions.

## Measure exit velocity directly

Do not use motor RPM as the primary calibration input. Measure the ball center in consecutive frames after it has cleanly left the launcher, convert pixel displacement to meters using your scale, and divide by frame time.

Record either the full initial velocity vector or speed plus launch direction. For a planar dataset, speed and elevation angle are sufficient if lateral velocity is known to be negligible.

Take multiple shots at each launcher setting. The mean describes the nominal launch state; the standard deviation is useful later for Monte Carlo uncertainty.

## Measure spin directly

Track a marked feature on the ball through multiple frames and fit angular position versus time. Convert the fitted angular speed to RPM or rad/s.

For ordinary backspin shots, the compact dataset format may store backspin RPM. For more general 3-D work, store the spin vector.

Avoid inferring spin solely from trajectory curvature while simultaneously fitting lift. Doing both makes spin and lift coefficient strongly confounded.

## Capture a useful calibration envelope

Do not collect all shots at one speed and one spin. Sample the range you expect to use on the field.

A good first dataset includes:

1. several **low-spin** shots over multiple exit speeds for drag fitting;
2. several spinning shots over multiple speeds and spin rates for lift fitting;
3. repeated shots at identical commands to measure launcher variation;
4. separate labels for materially different ball wear states if wear appears important.

The profile records its calibrated Reynolds-number and spin-parameter ranges. The simulator clamps coefficient tables at those boundaries and reports clamping rather than silently extrapolating.

## Dataset fields and units

The canonical calibration representation uses SI units.

Per shot, provide:

- `id`: unique shot identifier;
- `position`: initial `[x, y, z]` in meters;
- `muzzleVelocity`: shooter-relative `[vx, vy, vz]` in m/s;
- `spin`: `[omega_x, omega_y, omega_z]` in rad/s;
- `observations`: timestamped measured positions;
- `mass`: kg;
- `diameter` or `radius`: meters.

Optional fields include `robotVelocity`, `wind`, `airDensity`, `dynamicViscosity`, ball identifier/wear state, and observed HUB result.

For each observation, use a time in seconds and a measured position in meters. Keep timestamps relative to launch.

## Fit a profile

Install the offline calibration dependencies first (these are intentionally separate from the Vercel runtime dependencies):

```bash
pip install -r requirements-calibration.txt
```

Then run:

```bash
python -m scripts.calibrate_fuel shots.json \
  --output fuel-profile.json \
  --validation-fraction 0.2 \
  --seed 2026 \
  --drag-max-spin-parameter 0.05
```

The calibration workflow is staged:

1. split the dataset into training and held-out validation shots;
2. within the training set, select drag shots with dimensionless spin parameter $S \leq 0.05$ by default;
3. fit drag only from that low-spin subset;
4. hold drag fixed and fit lift from the remaining spinning training shots;
5. fit spin decay only if spin-versus-time measurements are available.

The threshold is controlled by `--drag-max-spin-parameter`. It is a **data-selection rule**, not an aerodynamic coefficient, and it can be changed explicitly when your measurement setup justifies a different low-spin boundary. The tool reports how many training shots went to each subset.

If no training shot is at or below the configured threshold, calibration stops instead of allowing Magnus-affected trajectories to bias the drag fit. Collect low-spin shots when possible; otherwise, raise `--drag-max-spin-parameter` explicitly and understand that the separation between drag and lift becomes weaker. Held-out validation shots are never borrowed to satisfy the drag fit.

The fitter uses bounded least-squares optimization and regularizes multi-knot tables. Lift tables may fit signed coefficients. It rejects a requested table complexity when the dataset cannot support it.

The engine can evaluate supplied two-dimensional $C_d(Re,S)$ tables, but this calibration tool intentionally does **not** fit them automatically. Separating spin-dependent drag from Magnus lift using trajectory observations alone is an identifiability problem that needs a purpose-built experiment rather than more optimizer parameters.

## Validate on held-out shots

Do not judge a profile only by its training residual. Keep a validation set of shots that were not used for fitting.

The profile can report:

- RMS 3-D position error;
- RMS vertical and downrange error;
- HUB-plane crossing error where applicable;
- entry-angle error where measured;
- predicted-versus-observed clean-entry counts where labels exist;
- calibrated Reynolds/spin ranges;
- fraction of validation samples that required coefficient clamping.

A low training error with a much larger held-out error is a warning for overfitting or an incomplete model.

## Load the profile in the browser

Open **Advanced Physics**, choose the calibration-profile mode, and load the generated JSON profile. The browser validates the schema and aerodynamic tables before applying anything.

The panel shows the profile name, validation summary when present, calibrated domain, and a warning when the simulated flight leaves that domain. Clearing the profile restores the legacy baseline model.

The profile contains data only. The browser does not execute code from imported profiles.

## Configure uncertainty for robust optimization

Calibration describes the mean flight model. Robust optimization also needs real shot variation.

Measure repeated shots and enter realistic uncertainty for:

- exit speed;
- elevation/azimuth;
- spin;
- ball mass when relevant;
- drag/lift multipliers when your validation data supports them;
- robot velocity;
- wind if field airflow matters.

The Monte Carlo evaluator is seeded for reproducibility. Robust optimization ranks candidates primarily by clean-entry probability and low-percentile clearance, not just nominal centerline clearance.

Avoid assigning arbitrary huge uncertainty "to be safe." Use distributions derived from measurements whenever possible.

## Recalibrate when the system changes

A profile should be treated as configuration-specific. Recalibrate or at least revalidate after meaningful changes such as:

- different wheel material or diameter;
- changed compression or hood geometry;
- substantially different FUEL wear;
- launcher service that changes exit conditions;
- operating well outside the original speed/spin range.

Do not extrapolate a profile far beyond its recorded Reynolds or spin-parameter domain. A clamp warning is a signal to collect more data, not a guarantee that the boundary coefficient remains valid.

If a profile was fitted with a simulator version that did not include buoyancy, revalidate it with the current engine. Older fitted drag or lift coefficients may have partially absorbed the previously missing upward force.
