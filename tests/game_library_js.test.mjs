import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GAME_PIECES, SCORING_TARGETS, DEFAULT_PIECE_ID, DEFAULT_TARGET_ID,
  STORAGE_KEY, emptyLibrary, parseLibrary, saveLibrary,
  resolveLibrarySelection, upsertLibraryItem, validatePiece, validateTarget,
} from '../src/gameCatalog.js';
import {
  createTargetGeometry, classifyTargetInteraction, targetWireframes, targetSideProfile,
} from '../src/scoringTargets.js';
import {simulateShot} from '../src/trajectory2d.js';

function samples(from, to) {
  return [
    {time: 0, state: [...from, 2, 0, -2, 0, 0, 0]},
    {time: 1, state: [...to, 2, 0, -2, 0, 0, 0]},
  ];
}
const opening = {id: 'custom-a', name: 'Slot', kind: 'slot', x: 0, lateralY: 0, z: 2, width: 1, height: 1, points: 3};

test('default REBUILT presets remain available and correctly selected', () => {
  const selection = resolveLibrarySelection(emptyLibrary());
  assert.equal(selection.piece.id, DEFAULT_PIECE_ID);
  assert.equal(selection.target.id, DEFAULT_TARGET_ID);
  assert.equal(GAME_PIECES[0].mass, 0.215);
  assert.equal(SCORING_TARGETS[0].kind, 'hub');
});

test('custom piece and scoring method persist through storage and reload', () => {
  const data = new Map();
  const storage = {
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
  };
  const piece = validatePiece({
    id: 'custom-ball', name: 'My new ball', shape: 'sphere',
    mass: 0.32, diameter: 0.19, dragCoeff: 0.55, liftCoeff: 0.2,
  });
  const target = validateTarget({...opening, id: 'custom-slot', width: 0.8});
  let library = upsertLibraryItem(emptyLibrary(), 'pieces', piece);
  library = upsertLibraryItem(library, 'targets', target);
  saveLibrary(library, storage);
  assert.equal(JSON.parse(data.get(STORAGE_KEY)).version, 1);
  const reloaded = parseLibrary(storage.getItem(STORAGE_KEY));
  assert.deepEqual(resolveLibrarySelection(reloaded), {piece, target});
  assert.equal(reloaded.targets[0].points, 3);
});

test('custom entries can be edited and deleted without mutating presets', () => {
  const piece = {...GAME_PIECES[0], id: 'custom-editor', name: 'Edited', mass: 0.3};
  let library = upsertLibraryItem(emptyLibrary(), 'pieces', piece);
  library = upsertLibraryItem(library, 'pieces', {...piece, mass: 0.4});
  assert.equal(library.pieces.length, 1);
  assert.equal(library.pieces[0].mass, 0.4);
  assert.throws(() => upsertLibraryItem(library, 'pieces', {...piece, id: DEFAULT_PIECE_ID}));
  library = parseLibrary({version: 1, ...library, pieces: [], pieceId: 'custom-editor'});
  assert.equal(library.pieceId, DEFAULT_PIECE_ID);
  assert.equal(GAME_PIECES[0].mass, 0.215);
});

test('corrupt, stale, and dangerous saved data are safely rejected', () => {
  assert.deepEqual(parseLibrary('not json'), emptyLibrary());
  assert.deepEqual(parseLibrary({version: 17}), emptyLibrary());
  const bad = parseLibrary({
    version: 1, pieceId: 'custom-fake', targetId: 'custom-foo',
    pieces: [{id: 'custom-fake', name: 'Invalid', shape: 'sphere', mass: -1}],
    targets: [{...opening, id: 'custom-foo', height: NaN}],
  });
  assert.equal(bad.pieces.length, 0);
  assert.equal(bad.targets.length, 0);
  assert.equal(bad.pieceId, DEFAULT_PIECE_ID);
  assert.equal(bad.targetId, DEFAULT_TARGET_ID);
  assert.throws(() => validatePiece({...GAME_PIECES[0], dragCoeff: Infinity}));
  assert.throws(() => validateTarget({...opening, points: 2.5}));
  assert.throws(() => validateTarget({...SCORING_TARGETS[0], panelHeight: 99}));
});

test('rectangular holes score only when center clears all edges and moves forward', () => {
  const geometry = createTargetGeometry(opening);
  const scored = classifyTargetInteraction(samples([-1, 0, 2], [1, 0, 2]), geometry, 0.1);
  assert.equal(scored.classification, 'clean-entry');
  assert.equal(scored.topCrossing.state[0], 0);
  assert.equal(scored.clearanceMargin, 0.4);
  const frame = classifyTargetInteraction(samples([-1, 0.47, 2], [1, 0.47, 2]), geometry, 0.1);
  assert.equal(frame.classification, 'rim-collision');
  const missed = classifyTargetInteraction(samples([-1, 2, 2], [1, 2, 2]), geometry, 0.1);
  assert.equal(missed.classification, 'miss');
  const wrongWay = classifyTargetInteraction(samples([1, 0, 2], [-1, 0, 2]), geometry, 0.1);
  assert.equal(wrongWay.classification, 'miss');
  const high = classifyTargetInteraction(samples([-1, 0, 3], [1, 0, 3]), geometry, 0.1);
  assert.equal(high.classification, 'miss');
});

test('round wall holes and horizontal hoops use different crossing planes', () => {
  const round = createTargetGeometry({...opening, kind: 'round-slot', diameter: 0.8});
  const hoop = createTargetGeometry({...opening, kind: 'hoop', diameter: 0.8});
  assert.equal(classifyTargetInteraction(samples([-1, 0, 2], [1, 0, 2]), round, 0.1).classification, 'clean-entry');
  assert.equal(classifyTargetInteraction(samples([-1, 0, 2], [1, 0, 2]), hoop, 0.1).classification, 'miss');
  assert.equal(classifyTargetInteraction(samples([0, 0, 3], [0, 0, 1]), hoop, 0.1).classification, 'clean-entry');
  assert.equal(classifyTargetInteraction(samples([0.35, 0, 3], [0.35, 0, 1]), hoop, 0.1).classification, 'rim-collision');
  assert.equal(classifyTargetInteraction(samples([1, 0, 3], [1, 0, 1]), hoop, 0.1).classification, 'miss');
});

test('custom hub respects its dimensions while original hub stays unchanged', () => {
  const baseline = createTargetGeometry();
  const custom = createTargetGeometry({...SCORING_TARGETS[0], topAcrossFlats: 2, z: 3.5, panelHeight: 1});
  assert.equal(baseline.topZ, 1.8288);
  assert.equal(custom.topZ, 3.5);
  assert.equal(custom.topApothem, 1);
  assert.ok(targetWireframes(custom, 0.1).frames.length === 2);
  assert.ok(targetWireframes(createTargetGeometry(opening), 0.1).frames.length === 1);
  assert.ok(targetSideProfile(createTargetGeometry(opening), 0.1).polygon.length === 2);
});

test('selected wall opening changes real shot hit classification', () => {
  const base = {
    launchX: -2, launchY: 1, velocity: 10, angleDeg: 30, azimuthDeg: 0,
    spinRPM: 0, mass: 0.2, radius: 0.1, dragCoeff: 0.47, liftCoeff: 0.25,
    airDensity: 1.204, gravity: 9.81, enableDrag: false, enableMagnus: false,
  };
  const big = simulateShot({...base, target: {...opening, width: 3, height: 3}});
  const narrow = simulateShot({...base, target: {...opening, width: 0.15, height: 3}});
  assert.equal(big.hitTarget, true);
  assert.equal(big.hubInteraction.classification, 'clean-entry');
  assert.equal(narrow.hitTarget, false);
  assert.equal(narrow.hubInteraction.classification, 'rim-collision');
  assert.equal(big.hubGeometry.kind, 'slot');
});
