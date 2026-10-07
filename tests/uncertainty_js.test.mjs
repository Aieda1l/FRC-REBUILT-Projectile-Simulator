import test from 'node:test';
import assert from 'node:assert/strict';

import {HUB_DIMENSIONS} from '../src/hubGeometry.js';
import {simulateShot} from '../src/trajectory2d.js';
import {createSeededRng, evaluateShotUncertainty, sampleDistribution, sampleShotParams} from '../src/uncertainty.js';

function baseParams(overrides = {}) {
  return {
    launchX: -3, launchY: 0.5, velocity: 8, angleDeg: 75, spinRPM: 0,
    mass: 0.215, radius: 0.075, dragCoeff: 0.47, liftCoeff: 0.25,
    airDensity: 1.204, gravity: 9.81, enableDrag: false, enableMagnus: false,
    targetX: 0, targetLateralY: 0, robotVelocity: [0, 0, 0], wind: [0, 0, 0],
    ...overrides,
  };
}
function cleanVacuumParams() {
  const launchX = -3, launchY = 0.5, xCross = -0.30, vx = 2;
  const t = (xCross - launchX) / vx;
  const vz = (HUB_DIMENSIONS.topZ - launchY + 0.5 * 9.81 * t * t) / t;
  return baseParams({velocity: Math.hypot(vx, vz), angleDeg: Math.atan2(vz, vx) * 180 / Math.PI});
}
test('seeded RNG is stable across languages', () => {
  const rng = createSeededRng(1);
  const expected = [0.23645552527159452, 0.3692706737201661, 0.5042420323006809, 0.7048832636792213];
  for (const value of expected) assert.ok(Math.abs(rng() - value) < 1e-15);
});
test('distribution sampling validates fixed uniform and normal definitions', () => {
  assert.equal(sampleDistribution({kind: 'fixed', value: 2}, () => 0.5), 2);
  assert.equal(sampleDistribution({kind: 'uniform', min: 2, max: 6}, () => 0.25), 3);
  assert.equal(sampleDistribution({kind: 'normal', mean: 4, sigma: 0}, () => 0.5), 4);
  assert.throws(() => sampleDistribution({kind: 'normal', mean: 0, sigma: -1}, Math.random), RangeError);
  assert.throws(() => sampleDistribution({kind: 'uniform', min: 5, max: 2}, Math.random), RangeError);
});
test('sampleShotParams applies additive perturbations and aerodynamic multipliers', () => {
  const sampled = sampleShotParams(baseParams(), {
    velocity: {kind: 'fixed', value: 1}, angleDeg: {kind: 'fixed', value: -2},
    spinRPM: {kind: 'fixed', value: 50}, mass: {kind: 'fixed', value: 0.01},
    dragMultiplier: {kind: 'fixed', value: 1.1}, liftMultiplier: {kind: 'fixed', value: 0.8},
    robotVelocity: [{kind: 'fixed', value: 0.2}, {kind: 'fixed', value: -0.1}, {kind: 'fixed', value: 0}],
    wind: [{kind: 'fixed', value: 0.5}, {kind: 'fixed', value: 0}, {kind: 'fixed', value: 0}],
  }, createSeededRng(7));
  assert.equal(sampled.velocity, 9);
  assert.equal(sampled.angleDeg, 73);
  assert.equal(sampled.spinRPM, 50);
  assert.equal(sampled.mass, 0.225);
  assert.equal(sampled.dragCoeff, 0.47 * 1.1);
  assert.equal(sampled.liftCoeff, 0.25 * 0.8);
  assert.deepEqual(sampled.robotVelocity, [0.2, -0.1, 0]);
  assert.deepEqual(sampled.wind, [0.5, 0, 0]);
});

test('uncertainty scaling supports 2-D drag and preserves signed lift and buoyancy toggle', () => {
  const sampled = sampleShotParams(baseParams({
    enableBuoyancy: false,
    dragModel: {
      kind: 'table2d',
      reynolds: [50000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0.2, 0.4], [0.3, 0.5]],
    },
    liftModel: {
      kind: 'table1d',
      spinParameters: [0, 1],
      coefficients: [-0.2, 0.1],
    },
  }), {
    dragMultiplier: {kind: 'fixed', value: 2},
    liftMultiplier: {kind: 'fixed', value: 0.5},
  }, createSeededRng(9));

  assert.deepEqual(sampled.dragModel.coefficients, [[0.4, 0.8], [0.6, 1.0]]);
  assert.deepEqual(sampled.liftModel.coefficients, [-0.1, 0.05]);
  assert.equal(sampled.enableBuoyancy, false);
});

test('fixed zero uncertainty reproduces deterministic shot and probabilities sum to one', () => {
  const params = cleanVacuumParams();
  const deterministic = simulateShot(params, {dt: 0.002});
  const robust = evaluateShotUncertainty(params, {
    velocity: {kind: 'fixed', value: 0}, angleDeg: {kind: 'fixed', value: 0}, spinRPM: {kind: 'fixed', value: 0},
  }, {sampleCount: 8, seed: 2026, dt: 0.002});
  assert.equal(deterministic.hubInteraction.classification, 'clean-entry');
  assert.equal(robust.counts['clean-entry'], 8);
  assert.equal(robust.probabilities['clean-entry'], 1);
  assert.ok(Math.abs(Object.values(robust.probabilities).reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Number.isFinite(robust.clearance.p10));
  assert.ok(Number.isFinite(robust.entryVelocity.median));
  assert.ok(Number.isFinite(robust.entryAngle.median));
});
test('generic scoring targets expose score probability without fixed classification keys', () => {
  const params = cleanVacuumParams();
  params.scoringTarget = {
    kind: 'top-circle',
    height: HUB_DIMENSIONS.topZ,
    centerX: -0.30,
    centerY: 0,
    openingRadius: 0.5,
    points: 2,
  };
  const robust = evaluateShotUncertainty(params, {
    velocity: {kind: 'fixed', value: 0},
    angleDeg: {kind: 'fixed', value: 0},
    spinRPM: {kind: 'fixed', value: 0},
  }, {sampleCount: 8, seed: 2026, dt: 0.002});
  assert.equal(robust.scoreCount, 8);
  assert.equal(robust.scoreProbability, 1);
  assert.equal(robust.counts['clean-entry'], 8);
  assert.equal(robust.counts['funnel-collision'], undefined);
});

test('uncertainty evaluator rejects pathological sample settings', () => {
  const params = cleanVacuumParams();
  assert.throws(() => evaluateShotUncertainty(params, {}, {sampleCount: 0}), RangeError);
  assert.throws(() => evaluateShotUncertainty(params, {}, {sampleCount: 10001}), RangeError);
});
