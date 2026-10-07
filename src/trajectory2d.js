import {integrateTrajectory, launchState} from './physics3d.js';
import {scoreTrajectory} from './scoring.js';
import {summarizeCalibrationDomain} from './calibration.js';

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

function projectSamples(samples) {
  const projected = [];
  let lastBucket = -1;

  samples.forEach((sample, index) => {
    const bucket = Math.floor(sample.time * 200 + 1e-12);
    const isBoundary = index === 0 || index === samples.length - 1;
    if (isBoundary || bucket > lastBucket) {
      const state = sample.state;
      projected.push({
        t: sample.time,
        x: state[0],
        y: state[2],
        vx: state[3],
        vy: state[5],
        speed: Math.hypot(state[3], state[4], state[5]),
      });
      lastBucket = bucket;
    }
  });
  return projected;
}

export function simulateShot(params, options = {}) {
  const {
    launchX,
    launchY,
    velocity,
    angleDeg,
    azimuthDeg = 0,
    spinRPM,
    mass,
    radius,
    collisionRadius = radius,
    dragCoeff,
    liftCoeff,
    airDensity,
    dynamicViscosity,
    dragModel,
    liftModel,
    spinDecayTimeConstant,
    calibrationProfile = null,
    gravity,
    enableDrag,
    enableMagnus,
    enableBuoyancy = true,
    targetX = 0,
    targetLateralY = 0,
    scoringTarget = null,
    scoringContext = {},
    robotVelocity = [0, 0, 0],
    wind = [0, 0, 0],
  } = params;

  const angle = angleDeg * DEG_TO_RAD;
  const azimuth = azimuthDeg * DEG_TO_RAD;
  const spin = spinRPM * 2 * Math.PI / 60;
  const horizontalSpeed = velocity * Math.cos(angle);
  const initial = launchState(
    [launchX, 0, launchY],
    [
      horizontalSpeed * Math.cos(azimuth),
      horizontalSpeed * Math.sin(azimuth),
      velocity * Math.sin(angle),
    ],
    [
      spin * Math.sin(azimuth),
      -spin * Math.cos(azimuth),
      0,
    ],
    robotVelocity,
  );
  const flightParams = {
    mass,
    radius,
    dragCoefficient: dragCoeff,
    liftCoefficient: liftCoeff,
    airDensity,
    gravity,
    wind,
    dragModel,
    liftModel,
    ...(spinDecayTimeConstant === undefined ? {} : {spinDecayTimeConstant}),
    ...(dynamicViscosity === undefined ? {} : {dynamicViscosity}),
    enableDrag,
    enableMagnus,
    enableBuoyancy,
  };

  const samples3d = integrateTrajectory(initial, flightParams, {
    method: options.method ?? 'rk4',
    dt: options.dt ?? 0.001,
    maxTime: options.maxTime ?? 5,
    terminalHeight: 0,
    terminalDirection: -1,
  });

  const calibrationDiagnostics = calibrationProfile
    ? summarizeCalibrationDomain(samples3d, flightParams, calibrationProfile)
    : null;

  const resolvedScoringTarget = scoringTarget ?? {
    kind: '2026-hex-hub',
    centerX: targetX,
    centerY: targetLateralY,
  };
  const scoringInteraction = scoreTrajectory(
    samples3d,
    resolvedScoringTarget,
    {radius, collisionRadius},
    scoringContext,
  );
  const hitTarget = scoringInteraction.isScore;

  const final = samples3d.at(-1);
  const points = projectSamples(samples3d);
  const maxHeight = Math.max(...samples3d.map((sample) => sample.state[2]));

  let impactSample = final;
  if (scoringInteraction.collisionPoint) {
    impactSample = scoringInteraction.collisionPoint;
  } else if (scoringInteraction.entrySample) {
    impactSample = scoringInteraction.entrySample;
  } else if (scoringInteraction.topCrossing) {
    impactSample = scoringInteraction.topCrossing;
  }

  const impactPoint = {
    x: impactSample.state[0],
    y: impactSample.state[2],
  };

  let entryVelocity = null;
  let entryAngle = null;
  const entrySample = scoringInteraction.entrySample ?? scoringInteraction.topCrossing;
  if (hitTarget && entrySample) {
    const state = entrySample.state;
    entryVelocity = Math.hypot(state[3], state[4], state[5]);
    entryAngle = Math.atan2(
      state[5],
      Math.hypot(state[3], state[4]),
    ) * RAD_TO_DEG;
  }

  const hubGeometry = resolvedScoringTarget.kind === '2026-hex-hub'
    ? scoringInteraction.geometry
    : null;

  return {
    samples3d,
    points,
    calibrationDiagnostics,
    scoringTarget: resolvedScoringTarget,
    scoringGeometry: scoringInteraction.geometry,
    scoringInteraction,
    // Deprecated compatibility aliases. Existing 2026 UI/optimizer consumers can
    // migrate independently without changing current behavior.
    hubGeometry,
    hubInteraction: scoringInteraction,
    hitTarget,
    score: scoringInteraction.points,
    impactPoint,
    flightTime: final.time,
    maxHeight,
    range: final.state[0] - launchX,
    entryVelocity,
    entryAngle,
  };
}

export function simulateTrajectory2D(params, options = {}) {
  return simulateShot(params, options);
}
