import test from 'node:test';
import assert from 'node:assert/strict';

import {
  HUB_DIMENSIONS,
  classifyHubInteraction,
  createHubGeometry,
  hexVertices,
} from '../src/hubGeometry.js';

const state = (x, y, z, vz = -1) => [x, y, z, 0, 0, vz, 0, 0, 0];
const sample = (time, x, y, z, vz = -1) => ({time, state: state(x, y, z, vz)});

function verticalSamplesAt(x, y = 0) {
  const g = createHubGeometry();
  return [
    sample(0, x, y, g.topZ + 0.10),
    sample(0.4, x, y, g.bottomZ - 0.10),
  ];
}

function centerlineSamples() {
  return verticalSamplesAt(0, 0);
}

test('HUB dimensions match official drawings', () => {
  assert.ok(Math.abs(HUB_DIMENSIONS.topAcrossFlats - 41.727 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.topZ - 72 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.bottomSide - 18.92 * 0.0254) < 1e-12);
  assert.ok(Math.abs(HUB_DIMENSIONS.panelHeight - 17.90 * 0.0254) < 1e-12);
});

test('regular hex vertices satisfy all half spaces and have a flat at negative x', () => {
  const g = createHubGeometry();
  assert.equal(g.topVertices.length, 6);
  assert.equal(g.bottomVertices.length, 6);
  for (const [x, y] of g.topVertices) {
    for (const [nx, ny] of g.normals) {
      assert.ok(nx * (x - g.centerX) + ny * (y - g.centerY) <= g.topApothem + 1e-12);
    }
  }
  const left = g.topVertices.filter(([x]) => Math.abs(x - (g.centerX - g.topApothem)) < 1e-12);
  assert.equal(left.length, 2);
});

test('hexVertices returns six points on the requested z plane', () => {
  const vertices = hexVertices(0.5, 1.2, 0.1, -0.2);
  assert.equal(vertices.length, 6);
  assert.ok(vertices.every(([, , z]) => z === 1.2));
});

test('centerline descending passage is clean entry', () => {
  const result = classifyHubInteraction(centerlineSamples(), createHubGeometry(), 0.075);
  assert.equal(result.classification, 'clean-entry');
  assert.ok(result.clearanceMargin > 0);
  assert.ok(result.topCrossing);
  assert.ok(result.bottomCrossing);
});

test('top edge overlap is rim collision', () => {
  const g = createHubGeometry();
  const result = classifyHubInteraction(verticalSamplesAt(g.topApothem - 0.03), g, 0.075);
  assert.equal(result.classification, 'rim-collision');
  assert.ok(result.clearanceMargin < 0);
});

test('clean top entry that reaches shrinking side is funnel collision', () => {
  const result = classifyHubInteraction(verticalSamplesAt(0.42), createHubGeometry(), 0.075);
  assert.equal(result.classification, 'funnel-collision');
  assert.ok(result.collisionPoint);
  assert.ok(result.clearanceMargin < 0);
});

test('sphere fully outside top opening is miss', () => {
  const g = createHubGeometry();
  const result = classifyHubInteraction(verticalSamplesAt(g.topApothem + 0.20), g, 0.075);
  assert.equal(result.classification, 'miss');
  assert.ok(result.missDistance > 0);
});

test('hex-corner grazing is rim collision rather than miss', () => {
  const g = createHubGeometry();
  const vertex = g.topVertices[1];
  const radial = Math.hypot(vertex[0] - g.centerX, vertex[1] - g.centerY);
  const x = vertex[0] + 0.04 * (vertex[0] - g.centerX) / radial;
  const y = vertex[1] + 0.04 * (vertex[1] - g.centerY) / radial;
  const result = classifyHubInteraction(verticalSamplesAt(x, y), g, 0.075);
  assert.equal(result.classification, 'rim-collision');
});

test('trajectory beginning below the top plane cannot be clean entry', () => {
  const g = createHubGeometry();
  const samples = [
    sample(0, 0, 0, g.topZ - 0.05),
    sample(0.2, 0, 0, g.bottomZ - 0.10),
  ];
  const result = classifyHubInteraction(samples, g, 0.075);
  assert.notEqual(result.classification, 'clean-entry');
});

test('bottom-rim overlap after valid top entry is funnel collision', () => {
  const g = createHubGeometry();
  const xTop = 0.30;
  const xBottom = g.bottomApothem - 0.03;
  const samples = [
    sample(0, xTop, 0, g.topZ + 0.10),
    sample(0.2, xTop, 0, g.topZ),
    sample(0.4, xBottom, 0, g.bottomZ),
    sample(0.5, xBottom, 0, g.bottomZ - 0.10),
  ];
  const result = classifyHubInteraction(samples, g, 0.075);
  assert.equal(result.classification, 'funnel-collision');
  assert.ok(result.clearanceMargin < 0);
});
