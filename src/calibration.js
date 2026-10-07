import {
  normalizeDragModel,
  normalizeLiftModel,
} from './aerodynamics.js';
import {aerodynamicDiagnostics} from './physics3d.js';

export const CALIBRATION_SCHEMA = 'frc-projectile-calibration-v1';

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return Number(value);
}

function positive(value, name) {
  const out = finite(value, name);
  if (out <= 0) throw new RangeError(`${name} must be positive`);
  return out;
}

function range2(value, name) {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new RangeError(`${name} must contain [min, max]`);
  }
  const low = finite(value[0], `${name}[0]`);
  const high = finite(value[1], `${name}[1]`);
  if (low < 0 || high < low) throw new RangeError(`${name} must be non-negative and ordered`);
  return [low, high];
}

export function parseCalibrationProfile(value) {
  let source = value;
  if (typeof source === 'string') {
    try {
      source = JSON.parse(source);
    } catch (error) {
      throw new RangeError(`invalid calibration JSON: ${error.message}`);
    }
  }
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    throw new RangeError('calibration profile must be an object');
  }
  if (source.schema !== CALIBRATION_SCHEMA) {
    throw new RangeError(`unsupported calibration schema: ${source.schema}`);
  }
  if (typeof source.name !== 'string' || !source.name.trim()) {
    throw new RangeError('calibration profile name must be non-empty');
  }
  if (!source.dragModel || !source.liftModel) {
    throw new RangeError('calibration profile requires dragModel and liftModel');
  }

  const environment = {
    ...(source.environment ?? {}),
    dynamicViscosity: positive(
      source.environment?.dynamicViscosity ?? 1.81e-5,
      'environment.dynamicViscosity',
    ),
  };
  const gamePiece = {
    ...(source.gamePiece ?? {}),
    diameter: positive(source.gamePiece?.diameter, 'gamePiece.diameter'),
    massReference: positive(source.gamePiece?.massReference, 'gamePiece.massReference'),
  };
  const decay = source.spinDecayTimeConstant;
  if (decay !== null && decay !== undefined) positive(decay, 'spinDecayTimeConstant');

  return {
    schema: CALIBRATION_SCHEMA,
    name: source.name.trim(),
    createdAt: source.createdAt ?? null,
    gamePiece,
    environment,
    dragModel: normalizeDragModel(source.dragModel, 0),
    liftModel: normalizeLiftModel(source.liftModel, 0),
    spinDecayTimeConstant: decay ?? null,
    domain: {
      reynolds: range2(source.domain?.reynolds, 'domain.reynolds'),
      spinParameter: range2(source.domain?.spinParameter, 'domain.spinParameter'),
    },
    validation: source.validation && typeof source.validation === 'object'
      ? {...source.validation}
      : {},
  };
}

export function applyCalibrationProfile(baseParams, profile) {
  if (profile === null || profile === undefined) {
    const {
      dragModel,
      liftModel,
      dynamicViscosity,
      spinDecayTimeConstant,
      calibrationProfile,
      ...baseline
    } = baseParams;
    void dragModel;
    void liftModel;
    void dynamicViscosity;
    void spinDecayTimeConstant;
    void calibrationProfile;
    return baseline;
  }
  const normalized = parseCalibrationProfile(profile);
  return {
    ...baseParams,
    dragModel: normalized.dragModel,
    liftModel: normalized.liftModel,
    dynamicViscosity: normalized.environment.dynamicViscosity,
    spinDecayTimeConstant: normalized.spinDecayTimeConstant,
    calibrationProfile: normalized,
  };
}

export function summarizeCalibrationDomain(samples, flightParams, profile) {
  const normalized = parseCalibrationProfile(profile);
  if (!Array.isArray(samples)) throw new RangeError('samples must be an array');
  let clampedSamples = 0;
  let minRe = Infinity;
  let maxRe = -Infinity;
  let minSpin = Infinity;
  let maxSpin = -Infinity;

  for (const sample of samples) {
    const diagnostics = aerodynamicDiagnostics(sample.state, flightParams);
    minRe = Math.min(minRe, diagnostics.reynolds);
    maxRe = Math.max(maxRe, diagnostics.reynolds);
    minSpin = Math.min(minSpin, diagnostics.spinParameter);
    maxSpin = Math.max(maxSpin, diagnostics.spinParameter);
    const outside = (
      diagnostics.reynolds < normalized.domain.reynolds[0]
      || diagnostics.reynolds > normalized.domain.reynolds[1]
      || diagnostics.spinParameter < normalized.domain.spinParameter[0]
      || diagnostics.spinParameter > normalized.domain.spinParameter[1]
      || diagnostics.dragClamped
      || diagnostics.liftClamped
    );
    if (outside) clampedSamples += 1;
  }

  const sampleCount = samples.length;
  return {
    sampleCount,
    clampedSamples,
    clampedFraction: sampleCount === 0 ? 0 : clampedSamples / sampleCount,
    reynoldsRange: sampleCount === 0 ? null : [minRe, maxRe],
    spinParameterRange: sampleCount === 0 ? null : [minSpin, maxSpin],
  };
}
