const EPS = 1e-12;

export const DEFAULT_FLIGHT_PARAMETERS = Object.freeze({
  mass: 0.215,
  radius: 0.075,
  dragCoefficient: 0.47,
  liftCoefficient: 0.25,
  airDensity: 1.204,
  gravity: 9.81,
  wind: [0, 0, 0],
  enableDrag: true,
  enableMagnus: true,
  spinDecayTimeConstant: null,
});

export class IntegrationError extends Error {}

function finiteNumber(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

function vector3(values, name) {
  if (!Array.isArray(values) || values.length !== 3 || !values.every(Number.isFinite)) {
    throw new RangeError(`${name} must contain exactly three finite values`);
  }
  return values.map(Number);
}

function state9(values) {
  if (!Array.isArray(values) || values.length !== 9 || !values.every(Number.isFinite)) {
    throw new RangeError('state must contain exactly nine finite values');
  }
  return values.map(Number);
}

function normalizeParams(input = {}) {
  const p = {
    ...DEFAULT_FLIGHT_PARAMETERS,
    ...input,
    wind: input.wind ? vector3(input.wind, 'wind') : [...DEFAULT_FLIGHT_PARAMETERS.wind],
  };
  finiteNumber(p.mass, 'mass');
  finiteNumber(p.radius, 'radius');
  finiteNumber(p.dragCoefficient, 'dragCoefficient');
  finiteNumber(p.liftCoefficient, 'liftCoefficient');
  finiteNumber(p.airDensity, 'airDensity');
  finiteNumber(p.gravity, 'gravity');
  if (p.mass <= 0 || p.radius <= 0) throw new RangeError('mass and radius must be positive');
  if (p.dragCoefficient < 0 || p.liftCoefficient < 0 || p.airDensity < 0 || p.gravity < 0) {
    throw new RangeError('aerodynamic coefficients, air density, and gravity must be non-negative');
  }
  if (p.spinDecayTimeConstant !== null) {
    finiteNumber(p.spinDecayTimeConstant, 'spinDecayTimeConstant');
    if (p.spinDecayTimeConstant <= 0) throw new RangeError('spinDecayTimeConstant must be positive');
  }
  return p;
}

function norm(v) {
  return Math.sqrt(v.reduce((sum, value) => sum + value * value, 0));
}

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function addScaled(base, terms, dt) {
  return base.map((value, i) => (
    value + dt * terms.reduce((sum, [coefficient, k]) => sum + coefficient * k[i], 0)
  ));
}

export function launchState(position, muzzleVelocity, spin, robotVelocity = [0, 0, 0]) {
  const p = vector3(position, 'position');
  const muzzle = vector3(muzzleVelocity, 'muzzleVelocity');
  const omega = vector3(spin, 'spin');
  const robot = vector3(robotVelocity, 'robotVelocity');
  return [
    ...p,
    muzzle[0] + robot[0],
    muzzle[1] + robot[1],
    muzzle[2] + robot[2],
    ...omega,
  ];
}

export function derivatives(state, params = {}) {
  const y = state9(state);
  const p = normalizeParams(params);
  const velocity = y.slice(3, 6);
  const omega = y.slice(6, 9);
  const relativeVelocity = velocity.map((value, i) => value - p.wind[i]);
  const speed = norm(relativeVelocity);
  const acceleration = [0, 0, -p.gravity];

  if (speed > EPS) {
    const uHat = relativeVelocity.map((value) => value / speed);
    const area = Math.PI * p.radius * p.radius;
    const dynamicArea = 0.5 * p.airDensity * area * speed * speed;

    if (p.enableDrag && p.dragCoefficient > 0) {
      for (let i = 0; i < 3; i += 1) {
        acceleration[i] += -dynamicArea * p.dragCoefficient * uHat[i] / p.mass;
      }
    }

    if (p.enableMagnus && p.liftCoefficient > 0) {
      const projection = dot(omega, uHat);
      const omegaPerp = omega.map((value, i) => value - projection * uHat[i]);
      const omegaPerpMag = norm(omegaPerp);
      if (omegaPerpMag > EPS) {
        const spinParameter = p.radius * omegaPerpMag / speed;
        const effectiveCl = p.liftCoefficient * Math.min(spinParameter / 0.5, 1);
        const lift = cross(omegaPerp, uHat);
        const liftNorm = norm(lift);
        if (liftNorm > EPS) {
          for (let i = 0; i < 3; i += 1) {
            acceleration[i] += dynamicArea * effectiveCl * (lift[i] / liftNorm) / p.mass;
          }
        }
      }
    }
  }

  const spinDerivative = p.spinDecayTimeConstant === null
    ? [0, 0, 0]
    : omega.map((value) => -value / p.spinDecayTimeConstant);

  return [...velocity, ...acceleration, ...spinDerivative];
}

export function rk4Step(state, params = {}, dt) {
  finiteNumber(dt, 'dt');
  if (dt <= 0) throw new RangeError('dt must be positive');
  const y = state9(state);
  const k1 = derivatives(y, params);
  const k2 = derivatives(addScaled(y, [[0.5, k1]], dt), params);
  const k3 = derivatives(addScaled(y, [[0.5, k2]], dt), params);
  const k4 = derivatives(addScaled(y, [[1, k3]], dt), params);
  return y.map((value, i) => (
    value + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
  ));
}

function rk45Step(state, params, dt) {
  const y = state9(state);
  const k1 = derivatives(y, params);
  const k2 = derivatives(addScaled(y, [[1 / 5, k1]], dt), params);
  const k3 = derivatives(addScaled(y, [[3 / 40, k1], [9 / 40, k2]], dt), params);
  const k4 = derivatives(addScaled(y, [[44 / 45, k1], [-56 / 15, k2], [32 / 9, k3]], dt), params);
  const k5 = derivatives(addScaled(y, [
    [19372 / 6561, k1], [-25360 / 2187, k2], [64448 / 6561, k3], [-212 / 729, k4],
  ], dt), params);
  const k6 = derivatives(addScaled(y, [
    [9017 / 3168, k1], [-355 / 33, k2], [46732 / 5247, k3], [49 / 176, k4], [-5103 / 18656, k5],
  ], dt), params);
  const y5 = addScaled(y, [
    [35 / 384, k1], [500 / 1113, k3], [125 / 192, k4], [-2187 / 6784, k5], [11 / 84, k6],
  ], dt);
  const k7 = derivatives(y5, params);
  const y4 = addScaled(y, [
    [5179 / 57600, k1], [7571 / 16695, k3], [393 / 640, k4],
    [-92097 / 339200, k5], [187 / 2100, k6], [1 / 40, k7],
  ], dt);
  return [y5, y5.map((value, i) => value - y4[i])];
}

function crossesHeight(a, b, height, direction) {
  const za = a[2] - height;
  const zb = b[2] - height;
  if (direction === -1) return za > 0 && zb <= 0;
  if (direction === 1) return za < 0 && zb >= 0;
  return (za > 0 && zb <= 0) || (za < 0 && zb >= 0);
}

function interpolateCrossing(t0, y0, t1, y1, height) {
  const dz = y1[2] - y0[2];
  const alpha = Math.max(0, Math.min(1, Math.abs(dz) <= EPS ? 0 : (height - y0[2]) / dz));
  const state = y0.map((value, i) => value + alpha * (y1[i] - value));
  state[2] = height;
  return {time: t0 + alpha * (t1 - t0), state};
}

export function integrateTrajectory(initialState, params = {}, options = {}) {
  const p = normalizeParams(params);
  let y = state9(initialState);
  const method = options.method ?? 'rk4';
  const dt = options.dt ?? 0.001;
  const maxTime = options.maxTime ?? 5;
  const rtol = options.rtol ?? 1e-6;
  const atol = options.atol ?? 1e-9;
  const minStep = options.minStep ?? 1e-5;
  const maxStep = options.maxStep ?? 0.05;
  const terminalHeight = options.terminalHeight === undefined ? 0 : options.terminalHeight;
  const terminalDirection = options.terminalDirection ?? -1;

  if (!['rk4', 'rk45'].includes(method)) throw new RangeError("method must be 'rk4' or 'rk45'");
  for (const [value, name] of [[dt, 'dt'], [maxTime, 'maxTime'], [rtol, 'rtol'], [atol, 'atol'], [minStep, 'minStep'], [maxStep, 'maxStep']]) {
    finiteNumber(value, name);
  }
  if (dt <= 0 || rtol <= 0 || atol <= 0 || minStep <= 0 || maxStep <= 0 || maxTime < 0 || minStep > maxStep) {
    throw new RangeError('invalid solver step or tolerance');
  }
  if (![-1, 0, 1].includes(terminalDirection)) throw new RangeError('terminalDirection must be -1, 0, or 1');
  if (terminalHeight !== null) finiteNumber(terminalHeight, 'terminalHeight');

  let t = 0;
  const samples = [{time: t, state: [...y]}];

  if (method === 'rk4') {
    while (t < maxTime - 1e-15) {
      const step = Math.min(dt, maxTime - t);
      const next = rk4Step(y, p, step);
      const nextTime = t + step;
      if (terminalHeight !== null && crossesHeight(y, next, terminalHeight, terminalDirection)) {
        samples.push(interpolateCrossing(t, y, nextTime, next, terminalHeight));
        break;
      }
      samples.push({time: nextTime, state: [...next]});
      y = next;
      t = nextTime;
    }
    return samples;
  }

  let step = Math.min(Math.max(dt, minStep), maxStep);
  while (t < maxTime - 1e-15) {
    const remaining = maxTime - t;
    const trialStep = Math.min(step, remaining);
    const [next, error] = rk45Step(y, p, trialStep);
    const scale = y.map((value, i) => atol + rtol * Math.max(Math.abs(value), Math.abs(next[i])));
    const errorNorm = Math.sqrt(
      error.reduce((sum, value, i) => sum + (value / scale[i]) ** 2, 0) / error.length,
    );

    if (errorNorm <= 1) {
      const nextTime = t + trialStep;
      if (terminalHeight !== null && crossesHeight(y, next, terminalHeight, terminalDirection)) {
        samples.push(interpolateCrossing(t, y, nextTime, next, terminalHeight));
        break;
      }
      samples.push({time: nextTime, state: [...next]});
      y = next;
      t = nextTime;
    }

    const factor = errorNorm === 0 ? 5 : Math.max(0.2, Math.min(5, 0.9 * errorNorm ** -0.2));
    if (errorNorm > 1 && trialStep <= minStep * (1 + 1e-12)) {
      throw new IntegrationError('RK45 tolerance cannot be met at minStep');
    }
    step = Math.min(maxStep, Math.max(minStep, trialStep * factor));
  }
  return samples;
}
