import {integrateTrajectory, launchState} from './physics3d.js';

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

export function simulateTrajectory2D(params) {
  const {
    launchX,
    launchY,
    velocity,
    angleDeg,
    spinRPM,
    mass,
    radius,
    dragCoeff,
    liftCoeff,
    airDensity,
    gravity,
    enableDrag,
    enableMagnus,
    targetX,
    targetY,
    targetRadius,
  } = params;

  const angle = angleDeg * DEG_TO_RAD;
  const spin = spinRPM * 2 * Math.PI / 60;
  const initial = launchState(
    [launchX, 0, launchY],
    [velocity * Math.cos(angle), 0, velocity * Math.sin(angle)],
    [0, -spin, 0],
  );
  const flightParams = {
    mass,
    radius,
    dragCoefficient: dragCoeff,
    liftCoefficient: liftCoeff,
    airDensity,
    gravity,
    enableDrag,
    enableMagnus,
  };
  const options = {method: 'rk4', dt: 0.001, maxTime: 5};

  const groundSamples = integrateTrajectory(initial, flightParams, {
    ...options,
    terminalHeight: 0,
    terminalDirection: -1,
  });

  let hitTarget = false;
  let targetCrossing = null;
  if (
    Number.isFinite(targetX)
    && Number.isFinite(targetY)
    && Number.isFinite(targetRadius)
  ) {
    const targetSamples = integrateTrajectory(initial, flightParams, {
      ...options,
      terminalHeight: targetY,
      terminalDirection: -1,
    });
    const crossing = targetSamples.at(-1);
    if (
      crossing
      && Math.abs(crossing.state[2] - targetY) <= 1e-9
      && crossing.state[5] < 0
    ) {
      const horizontalError = Math.hypot(
        crossing.state[0] - targetX,
        crossing.state[1],
      );
      if (horizontalError <= targetRadius) {
        hitTarget = true;
        targetCrossing = crossing;
      }
    }
  }

  const final = groundSamples.at(-1);
  const points = projectSamples(groundSamples);
  const maxHeight = Math.max(...groundSamples.map((sample) => sample.state[2]));

  let impactPoint = {x: final.state[0], y: Math.max(0, final.state[2])};
  let entryVelocity = null;
  let entryAngle = null;
  if (targetCrossing) {
    impactPoint = {x: targetCrossing.state[0], y: targetY};
    entryVelocity = Math.hypot(
      targetCrossing.state[3],
      targetCrossing.state[4],
      targetCrossing.state[5],
    );
    entryAngle = Math.atan2(
      targetCrossing.state[5],
      Math.hypot(targetCrossing.state[3], targetCrossing.state[4]),
    ) * RAD_TO_DEG;
  }

  return {
    points,
    hitTarget,
    impactPoint,
    flightTime: final.time,
    maxHeight,
    range: final.state[0] - launchX,
    entryVelocity,
    entryAngle,
  };
}
