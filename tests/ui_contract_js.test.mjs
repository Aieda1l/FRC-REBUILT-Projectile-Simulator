import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('Toggle uses a controlled native checkbox wired to onChange', async () => {
  const source = await readFile(new URL('../src/Toggle.jsx', import.meta.url), 'utf8');
  assert.match(source, /type=["']checkbox["']/);
  assert.match(source, /checked=\{checked\}/);
  assert.match(source, /onChange=\{[^}]*onChange/);
  assert.match(source, /event\.target\.checked/);
});

test('Physics Options still render through Toggle', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  for (const label of [
    'Air Drag',
    'Magnus Effect (Backspin)',
    'Show Ideal (No Drag)',
    'Show Error Envelope',
  ]) {
    assert.match(source, new RegExp(`<Toggle[^>]*label=["']${label.replace(/[()]/g, '\\$&')}["']`));
  }
});


test('trajectory2d uses one integration call site for flight and HUB scoring', async () => {
  const source = await readFile(new URL('../src/trajectory2d.js', import.meta.url), 'utf8');
  const callCount = source.split('integrateTrajectory(').length - 1;
  assert.equal(callCount, 1);
});

test('TrajectorySimulator delegates optimization to worker client with cancel UI', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /createOptimizerClient/);
  assert.match(source, /Cancel Optimization/);
  assert.match(source, /optimizerProgress/);
  assert.doesNotMatch(source, /const findOptimalAngle\s*=/);
  assert.doesNotMatch(source, /const findOptimalVelocity\s*=/);
  assert.doesNotMatch(source, /const findOptimalBoth\s*=/);
});


test('TrajectorySimulator uses shared HUB geometry for 2-D and 3-D views', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /HUB_DIMENSIONS/);
  assert.match(source, /createHubGeometry/);
  assert.match(source, /Trajectory3DView/);
  assert.match(source, /2-D/);
  assert.match(source, /3-D/);
  assert.doesNotMatch(source, /0\.529/);
  assert.doesNotMatch(source, /0\.454/);
  for (const label of ['CLEAN ENTRY', 'RIM COLLISION', 'FUNNEL COLLISION', 'MISS']) {
    assert.match(source, new RegExp(label));
  }
});


test('worker and progress UI share totalCandidates field', async () => {
  const worker = await readFile(new URL('../src/optimizer.worker.js', import.meta.url), 'utf8');
  const ui = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(worker, /totalCandidates/);
  assert.match(ui, /optimizerProgress\.totalCandidates/);
});
