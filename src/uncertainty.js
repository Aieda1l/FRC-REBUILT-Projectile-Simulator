import {simulateShot} from './trajectory2d.js';

const CLASSIFICATIONS = ['clean-entry', 'rim-collision', 'funnel-collision', 'miss'];
const UINT32 = 0x100000000;

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return Number(value);
}

function distributionValue(definition, rng, name) {
  if (definition === undefined) return undefined;
  try {
    return sampleDistribution(definition, rng);
  } catch (error) {
    if (error instanceof RangeError) throw new RangeError(`${name}: ${error.message}`);
    throw error;
  }
}

export function createSeededRng(seed = 0) {
  let state = Number(seed) >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / UINT32;
  };
}

function unitSample(rng) {
  const value = finite(rng(), 'rng result');
  if (value < 0 || value >= 1) throw new RangeError('rng result must be in [0, 1)');
  return value;
}

export function sampleDistribution(definition, rng) {
  if (!definition || typeof definition !== 'object' || Array.isArray(definition)) {
    throw new RangeError('distribution must be an object');
  }
  if (typeof rng !== 'function') throw new RangeError('rng must be a function');
  if (definition.kind === 'fixed') return finite(definition.value, 'value');
  if (definition.kind === 'uniform') {
    const min = finite(definition.min, 'min');
    const max = finite(definition.max, 'max');
    if (max < min) throw new RangeError('max must be greater than or equal to min');
    return min + (max - min) * unitSample(rng);
  }
  if (definition.kind === 'normal') {
    const mean = finite(definition.mean ?? 0, 'mean');
    const sigma = finite(definition.sigma, 'sigma');
    if (sigma < 0) throw new RangeError('sigma must be non-negative');
    const min = definition.min === undefined ? -Infinity : finite(definition.min, 'min');
    const max = definition.max === undefined ? Infinity : finite(definition.max, 'max');
    if (max < min) throw new RangeError('max must be greater than or equal to min');
    if (sigma === 0) return Math.max(min, Math.min(max, mean));
    let candidate = mean;
    for (let attempt = 0; attempt < 64; attempt += 1) {
      const u1 = Math.max(unitSample(rng), Number.EPSILON);
      const u2 = unitSample(rng);
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      candidate = mean + sigma * z;
      if (candidate >= min && candidate <= max) return candidate;
    }
    return Math.max(min, Math.min(max, candidate));
  }
  throw new RangeError(`unknown distribution kind: ${definition.kind}`);
}

function scaleDragModel(model, factor) {
  if (!model) return model;
  if (model.kind === 'constant') return {...model, coefficient: model.coefficient * factor};
  if (model.kind === 'table1d') {
    return {...model, coefficients: model.coefficients.map((value) => value * factor)};
  }
  throw new RangeError(`cannot scale drag model kind: ${model.kind}`);
}

function scaleLiftModel(model, factor) {
  if (!model) return model;
  if (model.kind === 'legacy-spin-cap') {
    return {...model, maxCoefficient: model.maxCoefficient * factor};
  }
  if (model.kind === 'table1d') {
    return {...model, coefficients: model.coefficients.map((value) => value * factor)};
  }
  if (model.kind === 'table2d') {
    return {
      ...model,
      coefficients: model.coefficients.map((row) => row.map((value) => value * factor)),
    };
  }
  throw new RangeError(`cannot scale lift model kind: ${model.kind}`);
}

function perturbedVector(baseVector, definitions, rng, name) {
  const base = Array.isArray(baseVector) ? [...baseVector] : [0, 0, 0];
  if (definitions === undefined) return base;
  if (!Array.isArray(definitions) || definitions.length !== 3) {
    throw new RangeError(`${name} uncertainty must contain exactly three distributions`);
  }
  return base.map((value, index) => (
    finite(value, `${name}[${index}]`)
    + distributionValue(definitions[index], rng, `${name}[${index}]`)
  ));
}

export function sampleShotParams(baseParams, uncertainty = {}, rng = Math.random) {
  if (!baseParams || typeof baseParams !== 'object') throw new RangeError('baseParams must be an object');
  if (!uncertainty || typeof uncertainty !== 'object') throw new RangeError('uncertainty must be an object');

  const out = {
    ...baseParams,
    robotVelocity: [...(baseParams.robotVelocity ?? [0, 0, 0])],
    wind: [...(baseParams.wind ?? [0, 0, 0])],
  };
  for (const key of ['velocity', 'angleDeg', 'spinRPM', 'mass']) {
    const delta = distributionValue(uncertainty[key], rng, key);
    if (delta !== undefined) out[key] = finite(out[key], key) + delta;
  }
  if (!(out.velocity > 0)) throw new RangeError('sampled velocity must be positive');
  if (!(out.mass > 0)) throw new RangeError('sampled mass must be positive');

  const dragMultiplier = distributionValue(uncertainty.dragMultiplier, rng, 'dragMultiplier');
  if (dragMultiplier !== undefined) {
    if (dragMultiplier < 0) throw new RangeError('dragMultiplier must be non-negative');
    if (out.dragModel) out.dragModel = scaleDragModel(out.dragModel, dragMultiplier);
    if (out.dragCoeff !== undefined) out.dragCoeff *= dragMultiplier;
  }
  const liftMultiplier = distributionValue(uncertainty.liftMultiplier, rng, 'liftMultiplier');
  if (liftMultiplier !== undefined) {
    if (liftMultiplier < 0) throw new RangeError('liftMultiplier must be non-negative');
    if (out.liftModel) out.liftModel = scaleLiftModel(out.liftModel, liftMultiplier);
    if (out.liftCoeff !== undefined) out.liftCoeff *= liftMultiplier;
  }

  out.robotVelocity = perturbedVector(out.robotVelocity, uncertainty.robotVelocity, rng, 'robotVelocity');
  out.wind = perturbedVector(out.wind, uncertainty.wind, rng, 'wind');
  return out;
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * fraction;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sorted[low];
  const t = position - low;
  return sorted[low] + t * (sorted[high] - sorted[low]);
}

function summary(values) {
  const finiteValues = values.filter(Number.isFinite);
  return {
    p10: percentile(finiteValues, 0.10),
    median: percentile(finiteValues, 0.50),
    p90: percentile(finiteValues, 0.90),
  };
}

export function evaluateShotUncertainty(
  baseParams,
  uncertainty = {},
  {sampleCount = 256, seed = 2026, dt = 0.002} = {},
) {
  if (!Number.isInteger(sampleCount) || sampleCount < 1 || sampleCount > 10000) {
    throw new RangeError('sampleCount must be an integer from 1 to 10000');
  }
  if (!Number.isFinite(dt) || dt <= 0) throw new RangeError('dt must be finite and positive');

  const rng = createSeededRng(seed);
  const counts = Object.fromEntries(CLASSIFICATIONS.map((key) => [key, 0]));
  const clearances = [];
  const entryVelocities = [];
  const entryAngles = [];

  for (let index = 0; index < sampleCount; index += 1) {
    const sampled = sampleShotParams(baseParams, uncertainty, rng);
    const result = simulateShot(sampled, {dt});
    const classification = result.hubInteraction.classification;
    counts[classification] += 1;
    if (Number.isFinite(result.hubInteraction.clearanceMargin)) {
      clearances.push(result.hubInteraction.clearanceMargin);
    }
    if (Number.isFinite(result.entryVelocity)) entryVelocities.push(result.entryVelocity);
    if (Number.isFinite(result.entryAngle)) entryAngles.push(result.entryAngle);
  }

  return {
    sampleCount,
    seed: Number(seed) >>> 0,
    counts,
    probabilities: Object.fromEntries(
      CLASSIFICATIONS.map((key) => [key, counts[key] / sampleCount]),
    ),
    clearance: summary(clearances),
    entryVelocity: summary(entryVelocities),
    entryAngle: summary(entryAngles),
  };
}
