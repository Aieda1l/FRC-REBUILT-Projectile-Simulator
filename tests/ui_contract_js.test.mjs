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

test('TrajectorySimulator offers shared-geometry 2-D and 3-D target views', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /hubGeometry/);
  assert.match(source, /Trajectory3DView/);
  assert.match(source, />2-D</);
  assert.match(source, />3-D</);
  assert.doesNotMatch(source, /0\.529/);
  assert.doesNotMatch(source, /0\.454/);
  for (const label of ['CLEAN ENTRY', 'RIM COLLISION', 'FUNNEL COLLISION', 'MISS']) assert.match(source, new RegExp(label));
});
test('Trajectory3DView exposes camera presets and a trajectory scrubber', async () => {
  const source = await readFile(new URL('../src/Trajectory3DView.jsx', import.meta.url), 'utf8');
  assert.match(source, /CAMERA_PRESETS/);
  for (const label of ['Isometric','Front','Side','Top']) assert.match(source, new RegExp(label));
  assert.match(source, /type=["']range["']/);
});

test('Toggle knob visibly follows checked state without relying on nested peer selector', async () => {
  const source = await readFile(new URL('../src/Toggle.jsx', import.meta.url), 'utf8');
  assert.match(source, /checked\s*\?\s*['"]translate-x-5['"]\s*:\s*['"]translate-x-0\.5['"]/);
});


test('3-D view receives ideal and envelope samples from existing toggles', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /idealSamples={showIdeal/);
  assert.match(source, /envelopeSamples={showEnvelope/);
  assert.ok(source.includes('idealResult?.samples3d'));
  assert.ok(source.includes('envelopeResults.map((entry) => entry.samples3d)'));
});

test('3-D renderer has distinct ideal and envelope styles', async () => {
  const source = await readFile(new URL('../src/Trajectory3DView.jsx', import.meta.url), 'utf8');
  assert.match(source, /idealSamples/);
  assert.match(source, /envelopeSamples/);
  assert.match(source, /strokeDasharray/);
  assert.match(source, /#22d3ee/);
  assert.match(source, /#f59e0b/);
});

test('Advanced Physics panel exposes motion wind profile and uncertainty controls', async () => {
  const source = await readFile(new URL('../src/AdvancedPhysicsPanel.jsx', import.meta.url), 'utf8');
  for (const label of [
    'Advanced Physics',
    'Robot Forward Velocity',
    'Robot Lateral Velocity',
    'Wind Forward',
    'Wind Lateral',
    'Calibration Profile',
    'Robust Analysis',
    'Velocity σ',
    'Angle σ',
    'Spin σ',
  ]) assert.match(source, new RegExp(label));
  assert.match(source, /validation\.rms3d/);
  assert.match(source, /clampedFraction/);
});

test('TrajectorySimulator validates profile JSON without replacing a valid profile on error', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /parseCalibrationProfile/);
  assert.match(source, /setProfileError/);
  assert.match(source, /setCalibrationProfile/);
  assert.match(source, /applyCalibrationProfile/);
  assert.match(source, /calibrationProfile/);
});

test('flywheel estimator requires explicit Apply Estimate action for measured launch state', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /handleApplyEstimate/);
  assert.match(source, /setVelocity\(/);
  assert.match(source, /setSpinRPM\(/);
  assert.match(source, /Apply Estimate/);
  assert.doesNotMatch(source, /Apply Estimated Backspin/);
});

test('robust optimization is routed through worker only when robust analysis is enabled', async () => {
  const source = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(source, /robustEnabled/);
  assert.match(source, /optimizerMode/);
  assert.match(source, /optimizerOptions/);
  assert.match(source, /optimizerClient\.start\(optimizerMode, params, optimizerOptions\)/);
  assert.match(source, /finalSamples/);
  assert.match(source, /uncertainty/);
});


test('robust uncertainty UI exposes forward and lateral robot velocity sigma', async () => {
  const panel = await readFile(new URL('../src/AdvancedPhysicsPanel.jsx', import.meta.url), 'utf8');
  const simulator = await readFile(new URL('../src/TrajectorySimulator.jsx', import.meta.url), 'utf8');
  assert.match(panel, /Robot Forward σ/);
  assert.match(panel, /Robot Lateral σ/);
  assert.match(panel, /setVectorSigma/);
  assert.match(simulator, /robotVelocity:\s*\[/);
  assert.match(simulator, /kind:\s*['"]normal['"][^\n]*mean:\s*0[^\n]*sigma:/);
});
