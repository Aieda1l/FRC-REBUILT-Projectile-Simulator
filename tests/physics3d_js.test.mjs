import test from 'node:test';
import assert from 'node:assert/strict';

import { simulateShot, simulateTrajectory2D } from '../src/trajectory2d.js';
import { HUB_DIMENSIONS } from '../src/hubGeometry.js';

import {
  IntegrationError,
  derivatives,
  integrateTrajectory,
  launchState,
  rk4Step,
} from '../src/physics3d.js';

function assertArrayClose(actual, expected, tolerance = 1e-12) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= tolerance,
      `index ${index}: ${value} != ${expected[index]}`,
    );
  });
}

test('launch adds robot field velocity', () => {
  assertArrayClose(
    launchState([0, 0, 1], [10, 0, 5], [0, -100, 0], [1, 2, 0]),
    [0, 0, 1, 11, 2, 5, 0, -100, 0],
  );
});

test('gravity-only RK4 matches analytic solution', () => {
  const out = integrateTrajectory(
    launchState([0, 0, 2], [10, 3, 5], [0, 0, 0]),
    {enableDrag: false, enableMagnus: false, enableBuoyancy: false},
    {method: 'rk4', dt: 0.01, maxTime: 0.4, terminalHeight: null},
  );
  assertArrayClose(
    out.at(-1).state.slice(0, 3),
    [4, 1.2, 2 + 5 * 0.4 - 0.5 * 9.81 * 0.4 ** 2],
    1e-9,
  );
});

test('backspin lifts and sidespin deflects', () => {
  assert.ok(
    derivatives(
      launchState([0, 0, 2], [10, 0, 0], [0, -100, 0]),
      {enableDrag: false, enableBuoyancy: false},
    )[5] > -9.81,
  );
  assert.ok(
    derivatives(
      launchState([0, 0, 2], [10, 0, 0], [0, 0, 100]),
      {enableDrag: false, enableBuoyancy: false},
    )[4] > 0,
  );
});

test('parallel spin gives zero Magnus lift', () => {
  assert.ok(Math.abs(
    derivatives(
      launchState([0, 0, 2], [10, 0, 0], [100, 0, 0]),
      {enableDrag: false, enableBuoyancy: false},
    )[5] + 9.81,
  ) < 1e-12);
});

test('matching wind removes aerodynamic force', () => {
  assertArrayClose(
    derivatives(
      launchState([0, 0, 2], [10, 0, 0], [0, 0, 0]),
      {wind: [10, 0, 0], enableBuoyancy: false},
    ).slice(3, 6),
    [0, 0, -9.81],
  );
});


test('buoyancy reduces effective downward acceleration', () => {
  const params = {
    mass: 0.215,
    radius: 0.075,
    airDensity: 1.204,
    gravity: 9.81,
    enableDrag: false,
    enableMagnus: false,
  };
  const state = launchState([0, 0, 2], [0, 0, 0], [0, 0, 0]);
  const volume = (4 / 3) * Math.PI * params.radius ** 3;
  const expected = -params.gravity
    + params.airDensity * volume * params.gravity / params.mass;
  assert.ok(Math.abs(derivatives(state, params)[5] - expected) < 1e-12);
  assert.equal(derivatives(state, {...params, enableBuoyancy: false})[5], -params.gravity);
  assert.equal(derivatives(state, {...params, airDensity: 0})[5], -params.gravity);
  assert.equal(derivatives(state, {...params, gravity: 0})[5], 0);
});

test('signed lift reverses Magnus acceleration', () => {
  const state = launchState([0, 0, 2], [10, 0, 0], [0, -100, 0]);
  const common = {
    enableDrag: false,
    enableBuoyancy: false,
    liftModel: {kind: 'table1d', spinParameters: [0, 1], coefficients: [0.2, 0.2]},
  };
  const positive = derivatives(state, common)[5] + 9.81;
  const negative = derivatives(state, {
    ...common,
    liftModel: {kind: 'table1d', spinParameters: [0, 1], coefficients: [-0.2, -0.2]},
  })[5] + 9.81;
  assert.ok(positive > 0);
  assert.ok(negative < 0);
  assert.ok(Math.abs(positive + negative) < 1e-12);
});

test('2-D drag uses current spin parameter', () => {
  const common = {
    enableMagnus: false,
    enableBuoyancy: false,
    dragModel: {
      kind: 'table2d',
      reynolds: [50000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0.2, 0.8], [0.2, 0.8]],
    },
  };
  const unspun = derivatives(
    launchState([0, 0, 2], [10, 0, 0], [0, 0, 0]),
    common,
  )[3];
  const spun = derivatives(
    launchState([0, 0, 2], [10, 0, 0], [0, -100, 0]),
    common,
  )[3];
  assert.ok(spun < unspun);
});

test('zero relative airflow has no drag or Magnus with 2-D drag model', () => {
  const params = {
    wind: [10, 0, 0],
    enableBuoyancy: false,
    dragModel: {
      kind: 'table2d',
      reynolds: [50000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0.2, 0.8], [0.2, 0.8]],
    },
  };
  const state = launchState([0, 0, 2], [10, 0, 0], [0, -100, 0]);
  assertArrayClose(derivatives(state, params).slice(3, 6), [0, 0, -9.81]);
  assert.equal(aerodynamicDiagnostics(state, params).dragClamped, true);
});

test('RK45 reaches maxTime exactly', () => {
  const out = integrateTrajectory(
    launchState([0, 0, 2], [3, 0, 4], [0, 0, 0]),
    {},
    {method: 'rk45', dt: 0.03, maxTime: 0.2, terminalHeight: null},
  );
  assert.ok(Math.abs(out.at(-1).time - 0.2) < 1e-12);
});

test('RK45 throws if minimum step cannot meet tolerance', () => {
  assert.throws(() => integrateTrajectory(
    launchState([0, 0, 2], [40, 15, 25], [0, -800, 300]),
    {wind: [3, -2, 0]},
    {
      method: 'rk45',
      dt: 0.2,
      minStep: 0.2,
      maxStep: 0.2,
      maxTime: 0.2,
      rtol: 1e-16,
      atol: 1e-16,
      terminalHeight: null,
    },
  ), IntegrationError);
});

test('invalid vectors and solver options throw', () => {
  assert.throws(() => launchState([0, 0], [1, 0, 0], [0, 0, 0]), RangeError);
  assert.throws(() => integrateTrajectory(
    launchState([0, 0, 1], [1, 0, 0], [0, 0, 0]),
    {},
    {method: 'euler'},
  ), RangeError);
});


test('Python golden fixtures match JavaScript', async () => {
  const { readFile } = await import('node:fs/promises');
  const text = await readFile(new URL('./fixtures/physics3d_golden.json', import.meta.url), 'utf8');
  const fixture = JSON.parse(text);
  assert.equal(fixture.schema, 'physics3d-golden-v1');

  for (const item of fixture.cases) {
    if (item.operation === 'derivatives') {
      assertArrayClose(derivatives(item.state, item.params), item.expected, 1e-10);
    } else if (item.operation === 'launch') {
      assertArrayClose(
        launchState(item.position, item.muzzleVelocity, item.spin, item.robotVelocity),
        item.expected,
        1e-10,
      );
    } else if (item.operation === 'rk4Step') {
      assertArrayClose(rk4Step(item.state, item.params, item.dt), item.expected, 1e-10);
    } else if (item.operation === 'trajectory') {
      const samples = integrateTrajectory(item.initialState, item.params, item.options);
      assert.equal(samples.length, item.expectedSamples.length);
      samples.forEach((sample, index) => {
        assert.ok(Math.abs(sample.time - item.expectedSamples[index].time) <= 1e-10);
        assertArrayClose(sample.state, item.expectedSamples[index].state, 1e-8);
      });
    } else if (item.operation === 'trajectoryFinal') {
      const samples = integrateTrajectory(item.initialState, item.params, item.options);
      assertArrayClose(samples.at(-1).state, item.expectedFinal, 2e-6);
    } else {
      assert.fail(`unknown fixture operation: ${item.operation}`);
    }
  }
});


function baseParams(overrides = {}) {
  return {
    launchX: -3,
    launchY: 0.8,
    velocity: 12,
    angleDeg: 45,
    spinRPM: 0,
    mass: 0.215,
    radius: 0.075,
    dragCoeff: 0.47,
    liftCoeff: 0.25,
    airDensity: 1.204,
    gravity: 9.81,
    enableDrag: true,
    enableMagnus: true,
    targetX: 0,
    targetY: 1.828,
    targetRadius: 0.454,
    ...overrides,
  };
}


function vacuumShotThroughTopAt(xCross) {
  const launchX = -3;
  const launchY = 0.5;
  const vx = 2;
  const t = (xCross - launchX) / vx;
  const vz = (HUB_DIMENSIONS.topZ - launchY + 0.5 * 9.81 * t * t) / t;
  return baseParams({
    launchX,
    launchY,
    velocity: Math.hypot(vx, vz),
    angleDeg: Math.atan2(vz, vx) * 180 / Math.PI,
    enableDrag: false,
    enableMagnus: false,
    airDensity: 0,
    targetX: 0,
  });
}

function crossingX(verticalVelocity, launchHeight, targetHeight, horizontalVelocity, descending) {
  const a = 0.5 * 9.81;
  const b = -verticalVelocity;
  const c = targetHeight - launchHeight;
  const disc = b * b - 4 * a * c;
  const t1 = (-b - Math.sqrt(disc)) / (2 * a);
  const t2 = (-b + Math.sqrt(disc)) / (2 * a);
  return horizontalVelocity * (descending ? Math.max(t1, t2) : Math.min(t1, t2));
}

test('legacy UI projection keeps x/z as x/y', () => {
  const result = simulateTrajectory2D(baseParams({
    launchX: 0,
    launchY: 1,
    velocity: 10,
    angleDeg: 30,
    enableDrag: false,
    enableMagnus: false,
  }));
  assert.ok(Math.abs(result.points[0].x) < 1e-12);
  assert.ok(Math.abs(result.points[0].y - 1) < 1e-12);
});

test('positive UI RPM produces upward Magnus effect', () => {
  const spun = simulateTrajectory2D(baseParams({spinRPM: 2000, enableMagnus: true}));
  const unspun = simulateTrajectory2D(baseParams({spinRPM: 0, enableMagnus: true}));
  assert.ok(spun.maxHeight > unspun.maxHeight);
});

test('ascending target-height crossing is ignored', () => {
  const vx = 2;
  const vz = 5;
  const result = simulateTrajectory2D(baseParams({
    launchX: 0,
    launchY: 0.5,
    velocity: Math.hypot(vx, vz),
    angleDeg: Math.atan2(vz, vx) * 180 / Math.PI,
    enableDrag: false,
    enableMagnus: false,
    targetX: crossingX(vz, 0.5, 1.0, vx, false),
    targetY: 1.0,
    targetRadius: 0.05,
  }));
  assert.equal(result.hitTarget, false);
});

test('simulateShot returns canonical 3-D samples and x/z projection from the same flight', () => {
  const result = simulateShot(vacuumShotThroughTopAt(-0.30), {dt: 0.002});
  assert.ok(result.samples3d.length > 2);
  assert.ok(result.points.length > 2);
  assert.equal(result.points[0].x, result.samples3d[0].state[0]);
  assert.equal(result.points[0].y, result.samples3d[0].state[2]);
  assert.ok(result.hubInteraction);
});

test('hitTarget is true only for clean entry', () => {
  const clean = simulateShot(vacuumShotThroughTopAt(-0.30));
  assert.equal(clean.hubInteraction.classification, 'clean-entry');
  assert.equal(clean.hitTarget, true);

  const collision = simulateShot(vacuumShotThroughTopAt(0.42));
  assert.equal(collision.hubInteraction.classification, 'funnel-collision');
  assert.equal(collision.hitTarget, false);
});

test('optimizer dt override uses fewer samples without changing the UI default', () => {
  const params = vacuumShotThroughTopAt(-0.30);
  const fine = simulateShot(params, {dt: 0.001});
  const coarse = simulateShot(params, {dt: 0.005});
  const uiDefault = simulateTrajectory2D(params);
  assert.ok(coarse.samples3d.length < fine.samples3d.length);
  assert.equal(uiDefault.samples3d.length, fine.samples3d.length);
  assert.ok(Math.abs(uiDefault.samples3d[1].time - 0.001) < 1e-12);
});



import { aerodynamicDiagnostics } from '../src/physics3d.js';

test('advanced aerodynamic models override scalar fallbacks', () => {
  const state = launchState([0, 0, 1], [10, 0, 0], [0, -100, 0]);
  const diagnostics = aerodynamicDiagnostics(state, {
    dragCoefficient: 0.99,
    liftCoefficient: 0.99,
    dragModel: {kind: 'constant', coefficient: 0.12},
    liftModel: {
      kind: 'table2d',
      reynolds: [50000, 200000],
      spinParameters: [0, 1],
      coefficients: [[0, 0.1], [0, 0.3]],
    },
  });
  assert.equal(diagnostics.dragCoefficient, 0.12);
  assert.ok(diagnostics.liftCoefficient > 0 && diagnostics.liftCoefficient < 0.99);
  assert.equal(diagnostics.dragClamped, false);
  assert.equal(diagnostics.liftClamped, false);
});

test('aerodynamic diagnostics stay finite at zero relative airflow', () => {
  const state = launchState([0, 0, 1], [10, 0, 0], [0, -100, 0]);
  const diagnostics = aerodynamicDiagnostics(state, {wind: [10, 0, 0]});
  assert.deepEqual(diagnostics, {
    reynolds: 0,
    spinParameter: 0,
    dragCoefficient: 0.47,
    liftCoefficient: 0,
    dragClamped: false,
    liftClamped: false,
  });
});


test('simulateShot adds robot velocity once and forwards lateral motion', () => {
  const result = simulateShot(baseParams({
    launchX: 0,
    launchY: 1,
    velocity: 10,
    angleDeg: 0,
    enableDrag: false,
    enableMagnus: false,
    robotVelocity: [2, 1, 0],
    wind: [0, 0, 0],
  }), {maxTime: 0.1});
  assertArrayClose(result.samples3d[0].state.slice(3, 6), [12, 1, 0]);
  assert.ok(result.samples3d.at(-1).state[1] > 0);
});


test('simulateShot rotates shooter-relative muzzle direction and backspin with azimuth', () => {
  const result = simulateShot({
    launchX: 0,
    launchY: 1,
    velocity: 10,
    angleDeg: 0,
    azimuthDeg: 90,
    spinRPM: 60,
    mass: 0.215,
    radius: 0.075,
    dragCoeff: 0.47,
    liftCoeff: 0.25,
    airDensity: 1.204,
    gravity: 0,
    enableDrag: false,
    enableMagnus: false,
    targetX: 0,
    robotVelocity: [0, 0, 0],
    wind: [0, 0, 0],
  }, {dt: 0.01, maxTime: 0.01});
  const initial = result.samples3d[0].state;
  assert.ok(Math.abs(initial[3]) < 1e-12);
  assert.ok(Math.abs(initial[4] - 10) < 1e-12);
  assert.ok(Math.abs(initial[6] - 2 * Math.PI) < 1e-12);
  assert.ok(Math.abs(initial[7]) < 1e-12);
});
