import {hasScoringMethod} from './scoring.js';

export const GAME_PROFILE_SCHEMA = 'frc-shooting-game-v1';

const gamePieces = new Map();
const gameProfiles = new Map();

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function finite(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new RangeError(`${name} must be finite`);
  return number;
}

function positive(value, name) {
  const number = finite(value, name);
  if (number <= 0) throw new RangeError(`${name} must be positive`);
  return number;
}

function nonNegative(value, name) {
  const number = finite(value, name);
  if (number < 0) throw new RangeError(`${name} must be non-negative`);
  return number;
}

function requiredString(value, name) {
  if (typeof value !== 'string' || !value.trim()) throw new RangeError(`${name} must be a non-empty string`);
  return value.trim();
}

export function normalizeGamePiece(definition) {
  if (!definition || typeof definition !== 'object' || Array.isArray(definition)) {
    throw new RangeError('game piece must be an object');
  }
  const piece = {
    ...clone(definition),
    id: requiredString(definition.id, 'gamePiece.id'),
    name: requiredString(definition.name, 'gamePiece.name'),
    mass: positive(definition.mass, 'gamePiece.mass'),
    radius: positive(definition.radius, 'gamePiece.radius'),
    dragCoeff: nonNegative(definition.dragCoeff, 'gamePiece.dragCoeff'),
    liftCoeff: nonNegative(definition.liftCoeff, 'gamePiece.liftCoeff'),
  };
  piece.collisionRadius = positive(
    definition.collisionRadius ?? piece.radius,
    'gamePiece.collisionRadius',
  );
  if (definition.momentOfInertia !== undefined) {
    piece.momentOfInertia = positive(definition.momentOfInertia, 'gamePiece.momentOfInertia');
  }
  if (definition.spinDecayTimeConstant !== undefined && definition.spinDecayTimeConstant !== null) {
    piece.spinDecayTimeConstant = positive(
      definition.spinDecayTimeConstant,
      'gamePiece.spinDecayTimeConstant',
    );
  }
  return piece;
}

export function registerGamePiece(definition, {replace = false} = {}) {
  const piece = normalizeGamePiece(definition);
  if (gamePieces.has(piece.id) && !replace) {
    throw new RangeError(`game piece already registered: ${piece.id}`);
  }
  gamePieces.set(piece.id, piece);
  return clone(piece);
}

export function getGamePiece(id) {
  const piece = gamePieces.get(id);
  if (!piece) throw new RangeError(`unknown game piece: ${id}`);
  return clone(piece);
}

export function listGamePieces() {
  return [...gamePieces.values()].map(clone);
}

function normalizeScoring(scoring) {
  if (!scoring || typeof scoring !== 'object' || Array.isArray(scoring)) {
    throw new RangeError('gameProfile.scoring must be an object');
  }
  const kind = requiredString(scoring.kind, 'gameProfile.scoring.kind');
  if (!hasScoringMethod(kind)) throw new RangeError(`unknown scoring method: ${kind}`);
  return {...clone(scoring), kind};
}

export function normalizeGameProfile(definition) {
  if (!definition || typeof definition !== 'object' || Array.isArray(definition)) {
    throw new RangeError('game profile must be an object');
  }
  if (definition.schema !== GAME_PROFILE_SCHEMA) {
    throw new RangeError(`game profile schema must be ${GAME_PROFILE_SCHEMA}`);
  }
  const pieceDefinition = definition.gamePiece;
  let gamePiece;
  if (typeof pieceDefinition === 'string') {
    if (!gamePieces.has(pieceDefinition)) throw new RangeError(`unknown game piece: ${pieceDefinition}`);
    gamePiece = pieceDefinition;
  } else {
    gamePiece = normalizeGamePiece(pieceDefinition);
  }
  return {
    ...clone(definition),
    schema: GAME_PROFILE_SCHEMA,
    id: requiredString(definition.id, 'gameProfile.id'),
    name: requiredString(definition.name, 'gameProfile.name'),
    gamePiece,
    scoring: normalizeScoring(definition.scoring),
  };
}

export function registerGameProfile(definition, {replace = false} = {}) {
  const profile = normalizeGameProfile(definition);
  if (gameProfiles.has(profile.id) && !replace) {
    throw new RangeError(`game profile already registered: ${profile.id}`);
  }
  gameProfiles.set(profile.id, profile);
  return clone(profile);
}

export function getGameProfile(id) {
  const profile = gameProfiles.get(id);
  if (!profile) throw new RangeError(`unknown game profile: ${id}`);
  return clone(profile);
}

export function listGameProfiles() {
  return [...gameProfiles.values()].map(clone);
}

export function parseGameProfile(text) {
  if (typeof text !== 'string') throw new RangeError('game profile JSON must be a string');
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new RangeError(`invalid game profile JSON: ${error.message}`);
  }
  return normalizeGameProfile(parsed);
}

function resolvePiece(profile) {
  return typeof profile.gamePiece === 'string'
    ? getGamePiece(profile.gamePiece)
    : normalizeGamePiece(profile.gamePiece);
}

export function applyGameProfile(baseParams, profileOrId, scoringContext = {}) {
  if (!baseParams || typeof baseParams !== 'object' || Array.isArray(baseParams)) {
    throw new RangeError('baseParams must be an object');
  }
  const profile = typeof profileOrId === 'string'
    ? getGameProfile(profileOrId)
    : normalizeGameProfile(profileOrId);
  const piece = resolvePiece(profile);
  return {
    ...baseParams,
    mass: piece.mass,
    radius: piece.radius,
    collisionRadius: piece.collisionRadius,
    dragCoeff: piece.dragCoeff,
    liftCoeff: piece.liftCoeff,
    ...(piece.spinDecayTimeConstant === undefined
      ? {}
      : {spinDecayTimeConstant: piece.spinDecayTimeConstant}),
    scoringTarget: clone(profile.scoring),
    scoringContext: {
      ...(baseParams.scoringContext ?? {}),
      ...scoringContext,
    },
    gameProfileId: profile.id,
    gamePieceId: piece.id,
  };
}

export const FUEL_2026 = Object.freeze({
  id: 'fuel-2026',
  name: 'FUEL (2026)',
  mass: 0.215,
  radius: 0.075,
  collisionRadius: 0.075,
  dragCoeff: 0.47,
  liftCoeff: 0.25,
  momentOfInertia: 0.000484,
  aerodynamicsStatus: 'uncalibrated sphere-like baseline',
});

export const DEFAULT_GAME_PROFILE = Object.freeze({
  schema: GAME_PROFILE_SCHEMA,
  id: 'frc-2026-rebuilt',
  name: '2026 REBUILT',
  season: 2026,
  gamePiece: FUEL_2026.id,
  scoring: {
    kind: '2026-hex-hub',
    centerX: 0,
    centerY: 0,
    points: {default: 1},
  },
});

registerGamePiece(FUEL_2026);
registerGameProfile(DEFAULT_GAME_PROFILE);
