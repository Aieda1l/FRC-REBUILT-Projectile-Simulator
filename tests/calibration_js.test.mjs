import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applyCalibrationProfile,
  parseCalibrationProfile,
  summarizeCalibrationDomain,
} from '../src/calibration.js';
import {simulateShot} from '../src/trajectory2d.js';

const PROFILE = {
  schema: 'frc-projectile-calibration-v1',
  name: 'Test FUEL profile',
  gamePiece: {diameter: 0.15, massReference: 0.215},
  environment: {dynamicViscosity: 1.81e-5},
  dragModel: {kind: 'table1d', reynolds: [50000, 200000], coefficients: [0.5, 0.3]},
  liftModel: {kind: 'table1d', spinParameters: [0, 1], coefficients: [0, 0.3]},
  spinDecayTimeConstant: null,
  domain: {reynolds: [50000, 200000], spinParameter: [0, 1]},
  validation: {rms3d: 0.04},
};

const BASE = {
  launchX: 0,
  launchY: 1,
  velocity: 10,
  angleDeg: 30,
  spinRPM: 500,
  mass: 0.215,
  radius: 0.075,
  dragCoeff: 0.47,
  liftCoeff: 0.25,
  airDensity: 1.204,
  gravity: 9.81,
  enableDrag: true,
  enableMagnus: true,
  targetX: 3,
};

test('calibration profile v1 parses and normalizes model data', () => {
  const profile = parseCalibrationProfile(PROFILE);
  assert.equal(profile.schema, 'frc-projectile-calibration-v1');
  assert.equal(profile.dragModel.kind, 'table1d');
  assert.equal(profile.liftModel.kind, 'table1d');
  assert.deepEqual(profile.domain.reynolds, [50000, 200000]);
});

test('schema v1 accepts 2-D drag and signed lift tables', () => {
  const profile = parseCalibrationProfile({
    ...PROFILE,
    dragModel: {
      kind: 'table2d',
      reynolds: [50000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0.5, 0.45], [0.35, 0.3]],
    },
    liftModel: {
      kind: 'table1d',
      spinParameters: [0, 1],
      coefficients: [-0.1, 0.3],
    },
  });
  assert.equal(profile.dragModel.kind, 'table2d');
  assert.deepEqual(profile.liftModel.coefficients, [-0.1, 0.3]);
});

test('unknown schema and reversed domains are rejected', () => {
  assert.throws(() => parseCalibrationProfile({...PROFILE, schema: 'future-v2'}), RangeError);
  assert.throws(() => parseCalibrationProfile({
    ...PROFILE,
    domain: {...PROFILE.domain, reynolds: [200000, 50000]},
  }), RangeError);
});

test('applying and clearing a profile preserves unrelated shot parameters', () => {
  const profile = parseCalibrationProfile(PROFILE);
  const applied = applyCalibrationProfile(BASE, profile);
  assert.equal(applied.velocity, BASE.velocity);
  assert.equal(applied.dragModel.kind, 'table1d');
  assert.equal(applied.spinDecayTimeConstant, null);

  const cleared = applyCalibrationProfile(BASE, null);
  assert.deepEqual(cleared, BASE);
  const original = simulateShot(BASE, {maxTime: 0.1});
  const reverted = simulateShot(cleared, {maxTime: 0.1});
  assert.deepEqual(reverted.samples3d.at(-1).state, original.samples3d.at(-1).state);
});

test('domain summary reports calibrated-domain clamping risk', () => {
  const profile = parseCalibrationProfile(PROFILE);
  const summary = summarizeCalibrationDomain(
    [{time: 0, state: [0, 0, 1, 1, 0, 0, 0, -10, 0]}],
    {
      radius: 0.075,
      dragCoefficient: 0.47,
      liftCoefficient: 0.25,
      dragModel: profile.dragModel,
      liftModel: profile.liftModel,
      dynamicViscosity: profile.environment.dynamicViscosity,
    },
    profile,
  );
  assert.equal(summary.sampleCount, 1);
  assert.equal(summary.clampedSamples, 1);
  assert.equal(summary.clampedFraction, 1);
  assert.ok(summary.reynoldsRange[1] < profile.domain.reynolds[0]);
});
