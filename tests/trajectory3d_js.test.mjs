import test from 'node:test';
import assert from 'node:assert/strict';
import {createHubGeometry} from '../src/hubGeometry.js';
import {CAMERA_PRESETS, projectScene} from '../src/trajectory3dProjection.js';

const hub = createHubGeometry();
const representative = [
  ...hub.topVertices,
  ...hub.bottomVertices,
  [-3, 0, 0.5],
  [-1.5, 0.2, 2.5],
  [0, -0.2, 1.5],
];

test('every camera preset projects representative scene to finite coordinates', () => {
  for (const camera of Object.values(CAMERA_PRESETS)) {
    const result = projectScene(representative, camera, {width: 640, height: 420, padding: 30});
    assert.ok(result.scale > 0 && Number.isFinite(result.scale));
    assert.ok(result.points.every((point) => (
      Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.depth)
    )));
  }
});

test('top view separates positive and negative lateral y', () => {
  const result = projectScene([[0, -1, 1], [0, 1, 1]], CAMERA_PRESETS.top, {
    width: 400, height: 400, padding: 20,
  });
  assert.notEqual(result.points[0].y, result.points[1].y);
});

test('side view collapses lateral y for points sharing x and z', () => {
  const result = projectScene([[1, -2, 3], [1, 2, 3]], CAMERA_PRESETS.front, {
    width: 400, height: 400, padding: 20,
  });
  assert.ok(Math.abs(result.points[0].x - result.points[1].x) < 1e-9);
  assert.ok(Math.abs(result.points[0].y - result.points[1].y) < 1e-9);
});

test('degenerate near-flat scene still produces finite projection', () => {
  const result = projectScene(
    [[1, 0, 1], [1 + 1e-15, 0, 1.2], [1, 0, 1.4]],
    CAMERA_PRESETS.side,
    {width: 320, height: 240, padding: 20},
  );
  assert.ok(result.points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)));
  assert.ok(result.scale > 0 && Number.isFinite(result.scale));
});
