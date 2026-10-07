import test from 'node:test';
import assert from 'node:assert/strict';

import {HUB_DIMENSIONS} from '../src/hubGeometry.js';
import {
  optimizeAngle,
  optimizeBoth,
  optimizeVelocity,
  optimizeRobust,
  rankCandidate,
  rankRobustCandidate,
} from '../src/optimizer.js';

const candidate = (classification, clearanceMargin, missDistance, velocity, angle) => ({
  velocity,
  angle,
  result: {hubInteraction: {classification, clearanceMargin, missDistance}},
});

test('clean entry always outranks collision and miss', () => {
  const clean = candidate('clean-entry', 0.01, 0, 10, 50);
  const collision = candidate('funnel-collision', -0.001, 0, 10, 50);
  const miss = candidate('miss', -0.2, 0.001, 10, 50);
  assert.ok(rankCandidate(clean, collision, {velocity: 10, angle: 50}) < 0);
  assert.ok(rankCandidate(clean, miss, {velocity: 10, angle: 50}) < 0);
});

test('larger positive clearance wins between clean entries', () => {
  const a = candidate('clean-entry', 0.03, 0, 10, 50);
  const b = candidate('clean-entry', 0.01, 0, 10, 50);
  assert.ok(rankCandidate(a, b, {velocity: 10, angle: 50}) < 0);
});

test('less-negative clearance wins between collisions', () => {
  const a = candidate('rim-collision', -0.002, 0, 10, 50);
  const b = candidate('funnel-collision', -0.02, 0, 10, 50);
  assert.ok(rankCandidate(a, b, {velocity: 10, angle: 50}) < 0);
});

test('smaller miss distance wins between misses', () => {
  const a = candidate('miss', -0.2, 0.01, 10, 50);
  const b = candidate('miss', -0.2, 0.10, 10, 50);
  assert.ok(rankCandidate(a, b, {velocity: 10, angle: 50}) < 0);
});

test('exact ties prefer parameters closest to the reference', () => {
  const a = candidate('miss', -0.2, 0.01, 10.1, 50.1);
  const b = candidate('miss', -0.2, 0.01, 14, 60);
  assert.ok(rankCandidate(a, b, {velocity: 10, angle: 50}) < 0);
});

function baseParams(overrides = {}) {
  return {
    launchX: -3,
    launchY: 0.5,
    velocity: 8,
    angleDeg: 75,
    spinRPM: 0,
    mass: 0.215,
    radius: 0.075,
    dragCoeff: 0.47,
    liftCoeff: 0.25,
    airDensity: 1.204,
    gravity: 9.81,
    enableDrag: false,
    enableMagnus: false,
    targetX: 0,
    ...overrides,
  };
}

function makeVacuumCleanParams() {
  const launchX = -3;
  const launchY = 0.5;
  const xCross = -0.30;
  const vx = 2;
  const t = (xCross - launchX) / vx;
  const vz = (HUB_DIMENSIONS.topZ - launchY + 0.5 * 9.81 * t * t) / t;
  return baseParams({
    launchX,
    launchY,
    velocity: Math.hypot(vx, vz),
    angleDeg: Math.atan2(vz, vx) * 180 / Math.PI,
  });
}

function assertFinalFineStep(solution) {
  assert.ok(solution);
  assert.equal(solution.result.hubInteraction.classification, 'clean-entry');
  const times = solution.result.samples3d.map((sample) => sample.time);
  for (let i = 1; i < times.length; i += 1) {
    assert.ok(times[i] - times[i - 1] <= 0.001 + 1e-12);
  }
}

test('angle optimizer returns a clean entry and revalidates at fine dt', () => {
  const params = makeVacuumCleanParams();
  const result = optimizeAngle(params);
  assertFinalFineStep(result.solution);
});

test('velocity optimizer returns a clean entry and revalidates at fine dt', () => {
  const params = makeVacuumCleanParams();
  const result = optimizeVelocity(params);
  assertFinalFineStep(result.solution);
});

test('combined optimizer is bounded and returns a fine-validated clean entry', () => {
  const params = makeVacuumCleanParams();
  const result = optimizeBoth(params);
  assert.ok(result.evaluatedCandidates < 1100);
  assertFinalFineStep(result.solution);
});

test('no-solution optimizer reports near miss without promoting it to success', () => {
  const params = baseParams({
    launchX: -30,
    gravity: 100,
    velocity: 5,
    angleDeg: 20,
  });
  const result = optimizeBoth(params);
  assert.equal(result.solution, null);
  assert.ok(result.bestNearMiss);
  assert.notEqual(result.bestNearMiss.result.hubInteraction.classification, 'clean-entry');
});

const robustCandidate = (probability, p10, velocity, angle, clearance = 0.02) => ({
  velocity,
  angle,
  result: {hubInteraction: {classification: 'clean-entry', clearanceMargin: clearance, missDistance: 0}},
  robust: {probabilities: {'clean-entry': probability}, clearance: {p10}},
});

test('robust ranking prefers clean-entry probability then low-percentile clearance', () => {
  const reference = {velocity: 10, angle: 50};
  const likely = robustCandidate(0.9, 0.001, 10, 50);
  const roomy = robustCandidate(0.8, 0.05, 10, 50);
  assert.ok(rankRobustCandidate(likely, roomy, reference) < 0);

  const safer = robustCandidate(0.9, 0.02, 10, 50);
  assert.ok(rankRobustCandidate(safer, likely, reference) < 0);
});

test('robust ranking falls back to deterministic reference-distance tie break', () => {
  const reference = {velocity: 10, angle: 50};
  const close = robustCandidate(0.9, 0.02, 10.1, 50.1);
  const far = robustCandidate(0.9, 0.02, 14, 60);
  assert.ok(rankRobustCandidate(close, far, reference) < 0);
});

test('robust optimizer revalidates winner at fine dt and full sample count', () => {
  const result = optimizeRobust(makeVacuumCleanParams(), {
    mode: 'angle',
    uncertainty: {
      velocity: {kind: 'fixed', value: 0},
      angleDeg: {kind: 'fixed', value: 0},
    },
    coarseSamples: 4,
    finalSamples: 8,
    seed: 99,
  });
  assert.ok(result.solution);
  assert.equal(result.solution.robust.sampleCount, 8);
  assert.equal(result.solution.robust.probabilities['clean-entry'], 1);
  const times = result.solution.result.samples3d.map((sample) => sample.time);
  for (let i = 1; i < times.length; i += 1) {
    assert.ok(times[i] - times[i - 1] <= 0.001 + 1e-12);
  }
  assert.ok(result.monteCarloEvaluations <= 12);
});
