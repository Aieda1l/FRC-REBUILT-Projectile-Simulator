import {normalizeGameProfile} from './gameProfiles.js';

export const GAME_PROFILE_STORAGE_KEY = 'frc-projectile-simulator.game-profiles.v1';

function requireStorage(storage) {
  if (!storage || typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function') {
    throw new RangeError('profile storage is unavailable');
  }
  return storage;
}

export function loadSavedGameProfiles(storage) {
  const source = requireStorage(storage);
  const raw = source.getItem(GAME_PROFILE_STORAGE_KEY);
  if (!raw) return {profiles: [], errors: []};

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    return {profiles: [], errors: [`saved game profiles are invalid JSON: ${error.message}`]};
  }
  if (!Array.isArray(parsed)) {
    return {profiles: [], errors: ['saved game profiles must be an array']};
  }

  const profiles = [];
  const errors = [];
  const seen = new Set();
  parsed.forEach((entry, index) => {
    try {
      const profile = normalizeGameProfile(entry);
      if (seen.has(profile.id)) {
        errors.push(`saved game profile ${profile.id} is duplicated`);
        return;
      }
      seen.add(profile.id);
      profiles.push(profile);
    } catch (error) {
      errors.push(`saved game profile ${index + 1}: ${error.message}`);
    }
  });
  return {profiles, errors};
}

export function saveGameProfile(profile, storage) {
  const source = requireStorage(storage);
  const normalized = normalizeGameProfile(profile);
  const current = loadSavedGameProfiles(source).profiles;
  const next = current.filter((entry) => entry.id !== normalized.id);
  next.push(normalized);
  next.sort((a, b) => a.name.localeCompare(b.name));
  source.setItem(GAME_PROFILE_STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function deleteGameProfile(profileId, storage) {
  const source = requireStorage(storage);
  const current = loadSavedGameProfiles(source).profiles;
  const next = current.filter((entry) => entry.id !== profileId);
  source.setItem(GAME_PROFILE_STORAGE_KEY, JSON.stringify(next));
  return next;
}
