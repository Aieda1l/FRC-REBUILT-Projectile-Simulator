import test from 'node:test';
import assert from 'node:assert/strict';

import {
  GAME_PROFILE_STORAGE_KEY,
  deleteGameProfile,
  loadSavedGameProfiles,
  saveGameProfile,
} from '../src/gameProfileStorage.js';
import {GAME_PROFILE_SCHEMA} from '../src/gameProfiles.js';

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
  };
}

function profile(id, name = id) {
  return {
    schema: GAME_PROFILE_SCHEMA,
    id,
    name,
    gamePiece: {
      id: `${id}-piece`,
      name: `${name} Piece`,
      mass: 0.2,
      radius: 0.08,
      collisionRadius: 0.08,
      dragCoeff: 0.47,
      liftCoeff: 0.2,
    },
    scoring: {
      kind: 'top-circle',
      height: 2.1,
      centerX: 0,
      centerY: 0,
      openingRadius: 0.5,
      points: {default: 2},
    },
  };
}

test('saved game profiles persist, replace by id, and remain normalized', () => {
  const storage = memoryStorage();
  let saved = saveGameProfile(profile('alpha', 'Alpha'), storage);
  assert.equal(saved.length, 1);
  saved = saveGameProfile(profile('alpha', 'Alpha Updated'), storage);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].name, 'Alpha Updated');

  const loaded = loadSavedGameProfiles(storage);
  assert.deepEqual(loaded.errors, []);
  assert.equal(loaded.profiles[0].gamePiece.mass, 0.2);
});

test('saved game profiles can be deleted without affecting other profiles', () => {
  const storage = memoryStorage();
  saveGameProfile(profile('alpha'), storage);
  saveGameProfile(profile('beta'), storage);
  const remaining = deleteGameProfile('alpha', storage);
  assert.deepEqual(remaining.map((entry) => entry.id), ['beta']);
});

test('corrupt storage is reported without breaking profile loading', () => {
  const storage = memoryStorage({[GAME_PROFILE_STORAGE_KEY]: '{bad json'});
  const loaded = loadSavedGameProfiles(storage);
  assert.deepEqual(loaded.profiles, []);
  assert.equal(loaded.errors.length, 1);
});
