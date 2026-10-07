import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_DYNAMIC_VISCOSITY,
  evaluateDragModel,
  evaluateLiftModel,
  normalizeDragModel,
  normalizeLiftModel,
  reynoldsNumber,
  spinParameter,
} from '../src/aerodynamics.js';

test('Reynolds number uses rho v D over dynamic viscosity', () => {
  const actual = reynoldsNumber({
    airDensity: 1.204,
    speed: 12,
    diameter: 0.15,
    dynamicViscosity: DEFAULT_DYNAMIC_VISCOSITY,
  });
  assert.ok(Math.abs(actual - (1.204 * 12 * 0.15 / 1.81e-5)) < 1e-9);
});

test('spin parameter is finite and zero without airflow', () => {
  assert.equal(spinParameter({radius: 0.075, perpendicularSpin: 200, speed: 0}), 0);
  assert.equal(spinParameter({radius: 0.075, perpendicularSpin: 200, speed: 1e-13}), 0);
  assert.ok(Math.abs(spinParameter({radius: 0.075, perpendicularSpin: 80, speed: 12}) - 0.5) < 1e-12);
});

test('constant and 1-D drag models evaluate and clamp', () => {
  assert.deepEqual(
    evaluateDragModel(normalizeDragModel({kind: 'constant', coefficient: 0.47}, 0.1), 120000),
    {coefficient: 0.47, clamped: false},
  );
  const model = normalizeDragModel(
    {kind: 'table1d', reynolds: [100000, 200000], coefficients: [0.5, 0.3]},
    0.47,
  );
  assert.deepEqual(evaluateDragModel(model, 150000), {coefficient: 0.4, clamped: false});
  assert.deepEqual(evaluateDragModel(model, 50000), {coefficient: 0.5, clamped: true});
  assert.deepEqual(evaluateDragModel(model, 150000, 0.75), {coefficient: 0.4, clamped: false});
});

test('2-D drag models interpolate over Reynolds number and spin', () => {
  const model = normalizeDragModel({
    kind: 'table2d',
    reynolds: [100000, 200000],
    spinParameters: [0, 1],
    coefficients: [[0.5, 0.4], [0.3, 0.2]],
  }, 0.47);

  assert.deepEqual(evaluateDragModel(model, 150000, 0.5), {coefficient: 0.35, clamped: false});
  assert.deepEqual(evaluateDragModel(model, 50000, 0.5), {coefficient: 0.45, clamped: true});
  const highSpin = evaluateDragModel(model, 150000, 2);
  assert.equal(highSpin.clamped, true);
  assert.ok(Math.abs(highSpin.coefficient - 0.3) < 1e-12);
  assert.deepEqual(evaluateDragModel(model, 150000), {coefficient: 0.4, clamped: false});
});

test('legacy, 1-D, and 2-D lift models interpolate correctly', () => {
  const legacy = normalizeLiftModel({kind: 'legacy-spin-cap', maxCoefficient: 0.25}, 0.1);
  assert.deepEqual(evaluateLiftModel(legacy, 120000, 0.25), {coefficient: 0.125, clamped: false});
  assert.deepEqual(evaluateLiftModel(legacy, 120000, 0.75), {coefficient: 0.25, clamped: false});

  const oneD = normalizeLiftModel(
    {kind: 'table1d', spinParameters: [0, 1], coefficients: [0, 0.4]},
    0.25,
  );
  assert.deepEqual(evaluateLiftModel(oneD, 120000, 0.5), {coefficient: 0.2, clamped: false});
  assert.deepEqual(evaluateLiftModel(oneD, 120000, 2), {coefficient: 0.4, clamped: true});

  const twoD = normalizeLiftModel({
    kind: 'table2d',
    reynolds: [100000, 200000],
    spinParameters: [0, 1],
    coefficients: [[0, 0.2], [0.2, 0.6]],
  }, 0.25);
  assert.deepEqual(evaluateLiftModel(twoD, 150000, 0.5), {coefficient: 0.25, clamped: false});

  const signed = normalizeLiftModel(
    {kind: 'table1d', spinParameters: [0, 1], coefficients: [-0.2, 0.2]},
    0.25,
  );
  assert.deepEqual(evaluateLiftModel(signed, 120000, 0.25), {coefficient: -0.1, clamped: false});
});

test('invalid tables are rejected', () => {
  const bad = [
    () => normalizeDragModel({kind: 'table1d', reynolds: [100000], coefficients: [0.4]}, 0.47),
    () => normalizeDragModel({kind: 'table1d', reynolds: [200000, 100000], coefficients: [0.3, 0.4]}, 0.47),
    () => normalizeDragModel({kind: 'table1d', reynolds: [100000, 100000], coefficients: [0.3, 0.4]}, 0.47),
    () => normalizeLiftModel({
      kind: 'table2d',
      reynolds: [100000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0, 0.1]],
    }, 0.25),
    () => normalizeDragModel({
      kind: 'table2d',
      reynolds: [100000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0.5, -0.1], [0.3, 0.2]],
    }, 0.47),
    () => normalizeLiftModel({kind: 'legacy-spin-cap', maxCoefficient: -0.1}, 0.25),
  ];
  for (const fn of bad) assert.throws(fn, RangeError);
});
