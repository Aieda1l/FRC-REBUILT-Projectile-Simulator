import test from 'node:test';
import assert from 'node:assert/strict';
import {createHubGeometry} from '../src/hubGeometry.js';
import {CAMERA_PRESETS, projectScene} from '../src/trajectory3dProjection.js';
const finitePoint = (p) => Number.isFinite(p.x) && Number.isFinite(p.y);
function representativePoints() {
  const hub = createHubGeometry();
  return [...hub.topVertices, ...hub.bottomVertices, [-3,0,0.5], [-1.5,0.1,2.5], [0,0,hub.topZ]];
}
test('every camera preset projects representative scene to finite coordinates', () => {
  for (const camera of Object.values(CAMERA_PRESETS)) {
    const projected = projectScene(representativePoints(), camera, {width:600,height:400,padding:30});
    assert.equal(projected.points.length, representativePoints().length);
    assert.ok(projected.points.every(finitePoint));
    assert.ok(Number.isFinite(projected.scale) && projected.scale > 0);
  }
});
test('top view separates positive and negative lateral y', () => {
  const p = projectScene([[0,-1,0],[0,1,0]], CAMERA_PRESETS.top, {width:300,height:300,padding:20});
  assert.notEqual(p.points[0].y, p.points[1].y);
});
test('side view collapses lateral y for points sharing x and z', () => {
  const p = projectScene([[1,-2,3],[1,2,3]], CAMERA_PRESETS.side, {width:300,height:300,padding:20});
  assert.ok(Math.abs(p.points[0].x-p.points[1].x)<1e-12 && Math.abs(p.points[0].y-p.points[1].y)<1e-12);
});
test('degenerate near-flat scene still produces finite projection', () => {
  const p = projectScene([[1,0,1],[1+1e-14,0,2],[1+2e-14,0,3]], CAMERA_PRESETS.isometric, {width:320,height:240,padding:20});
  assert.ok(p.points.every(finitePoint));
  assert.ok(Number.isFinite(p.scale) && p.scale > 0);
});
