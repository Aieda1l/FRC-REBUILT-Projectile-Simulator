import test from 'node:test';
import assert from 'node:assert/strict';

import {
  findPlaneCrossing,
  listScoringMethods,
  registerScoringMethod,
  scoreTrajectory,
} from '../src/scoring.js';
import {
  DEFAULT_GAME_PROFILE,
  GAME_PROFILE_SCHEMA,
  applyGameProfile,
  getGamePiece,
  parseGameProfile,
  registerGamePiece,
  registerGameProfile,
} from '../src/gameProfiles.js';

function sample(time, x, y, z, vx = 0, vy = 0, vz = -1) {
  return {time, state: [x, y, z, vx, vy, vz, 0, 0, 0]};
}

const sphere = {radius: 0.1, collisionRadius: 0.1};

test('built-in scoring methods cover funnel, top opening, and plane aperture families', () => {
  assert.deepEqual(listScoringMethods(), ['2026-hex-hub', 'plane-aperture', 'top-circle']);
});

test('top-circle scorer distinguishes clean entry, rim collision, and miss', () => {
  const target = {
    kind: 'top-circle',
    height: 2,
    centerX: 0,
    centerY: 0,
    openingRadius: 0.5,
    points: {auto: 4, teleop: 2, default: 2},
  };
  const clean = scoreTrajectory(
    [sample(0, 0, 0, 3), sample(1, 0, 0, 1)],
    target,
    sphere,
    {phase: 'auto'},
  );
  assert.equal(clean.classification, 'clean-entry');
  assert.equal(clean.isScore, true);
  assert.equal(clean.points, 4);
  assert.ok(clean.clearanceMargin > 0);

  const rim = scoreTrajectory(
    [sample(0, 0.55, 0, 3), sample(1, 0.55, 0, 1)],
    target,
    sphere,
  );
  assert.equal(rim.classification, 'rim-collision');
  assert.equal(rim.status, 'collision');
  assert.equal(rim.isScore, false);

  const miss = scoreTrajectory(
    [sample(0, 0.8, 0, 3), sample(1, 0.8, 0, 1)],
    target,
    sphere,
  );
  assert.equal(miss.classification, 'miss');
  assert.ok(miss.missDistance > 0);
});

test('plane-aperture scorer handles wall goals with a projectile clearance envelope', () => {
  const target = {
    kind: 'plane-aperture',
    planePoint: [0, 0, 1],
    planeNormal: [-1, 0, 0],
    apertureCenter: [0, 0, 1],
    up: [0, 0, 1],
    crossingDirection: -1,
    shape: 'rectangle',
    width: 1.0,
    height: 0.5,
    points: 3,
  };
  const clean = scoreTrajectory(
    [sample(0, -1, 0, 1, 2, 0, 0), sample(1, 1, 0, 1, 2, 0, 0)],
    target,
    {radius: 0.05},
  );
  assert.equal(clean.isScore, true);
  assert.equal(clean.points, 3);

  const frame = scoreTrajectory(
    [sample(0, -1, 0, 1.27, 2, 0, 0), sample(1, 1, 0, 1.27, 2, 0, 0)],
    target,
    {radius: 0.05},
  );
  assert.equal(frame.classification, 'rim-collision');
  assert.equal(frame.collisionType, 'frame');
  assert.equal(frame.status, 'collision');
});

test('plane crossing direction can reject the wrong traversal', () => {
  const samples = [
    sample(0, -1, 0, 1, 2, 0, 0),
    sample(1, 1, 0, 1, 2, 0, 0),
  ];
  assert.ok(findPlaneCrossing(samples, [0, 0, 1], [-1, 0, 0], -1));
  assert.equal(findPlaneCrossing(samples, [0, 0, 1], [-1, 0, 0], 1), null);
});

test('custom scoring methods can be registered without changing the simulator core', () => {
  registerScoringMethod('test-always-score', () => ({
    classification: 'clean-entry',
    status: 'scored',
    isScore: true,
    clearanceMargin: 1,
  }));
  const interaction = scoreTrajectory(
    [sample(0, 0, 0, 1), sample(1, 1, 0, 0)],
    {kind: 'test-always-score', points: {special: 7, default: 1}},
    sphere,
    {variant: 'special'},
  );
  assert.equal(interaction.isScore, true);
  assert.equal(interaction.points, 7);
});

test('game profiles register reusable pieces and apply physics plus scoring config', () => {
  const defaultParams = applyGameProfile({velocity: 12}, DEFAULT_GAME_PROFILE);
  assert.equal(defaultParams.mass, 0.215);
  assert.equal(defaultParams.radius, 0.075);
  assert.equal(defaultParams.scoringTarget.kind, '2026-hex-hub');

  registerGamePiece({
    id: 'test-piece',
    name: 'Test Piece',
    mass: 0.2,
    radius: 0.08,
    collisionRadius: 0.09,
    dragCoeff: 0.4,
    liftCoeff: 0.1,
  });
  registerGameProfile({
    schema: GAME_PROFILE_SCHEMA,
    id: 'test-game',
    name: 'Test Game',
    gamePiece: 'test-piece',
    scoring: {
      kind: 'top-circle',
      height: 2.5,
      openingRadius: 0.6,
      points: {auto: 4, teleop: 2},
    },
  });
  const params = applyGameProfile({velocity: 10}, 'test-game', {phase: 'teleop'});
  assert.equal(params.mass, 0.2);
  assert.equal(params.collisionRadius, 0.09);
  assert.equal(params.scoringTarget.kind, 'top-circle');
  assert.equal(params.scoringContext.phase, 'teleop');
  assert.equal(getGamePiece('test-piece').name, 'Test Piece');
});

test('JSON game profiles validate scorer kinds and schema', () => {
  const parsed = parseGameProfile(JSON.stringify({
    schema: GAME_PROFILE_SCHEMA,
    id: 'json-game',
    name: 'JSON Game',
    gamePiece: {
      id: 'json-piece',
      name: 'JSON Piece',
      mass: 0.25,
      radius: 0.1,
      dragCoeff: 0.47,
      liftCoeff: 0.2,
    },
    scoring: {kind: 'top-circle', height: 2, openingRadius: 0.5},
  }));
  assert.equal(parsed.id, 'json-game');
  assert.throws(() => parseGameProfile('{not json'), RangeError);
  assert.throws(() => parseGameProfile(JSON.stringify({
    schema: GAME_PROFILE_SCHEMA,
    id: 'bad',
    name: 'Bad',
    gamePiece: 'fuel-2026',
    scoring: {kind: 'unknown-method'},
  })), RangeError);
});
