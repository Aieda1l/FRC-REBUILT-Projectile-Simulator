import {classifyHubInteraction, createHubGeometry} from './hubGeometry.js';

const EPS = 1e-12;
const scoringMethods = new Map();

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

function vector3(value, name) {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new RangeError(`${name} must contain exactly three values`);
  }
  return value.map((component, index) => finite(component, `${name}[${index}]`));
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v, amount) {
  return [v[0] * amount, v[1] * amount, v[2] * amount];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v, name) {
  const magnitude = Math.hypot(...v);
  if (!Number.isFinite(magnitude) || magnitude <= EPS) {
    throw new RangeError(`${name} must have non-zero magnitude`);
  }
  return v.map((component) => component / magnitude);
}

export function interpolateSample(a, b, alpha) {
  const clamped = Math.max(0, Math.min(1, finite(alpha, 'alpha')));
  return {
    time: a.time + clamped * (b.time - a.time),
    state: a.state.map((value, index) => value + clamped * (b.state[index] - value)),
  };
}

export function findPlaneCrossing(
  samples,
  planePoint,
  planeNormal,
  crossingDirection = 0,
) {
  if (!Array.isArray(samples) || samples.length < 2) return null;
  const point = vector3(planePoint, 'planePoint');
  const normal = normalize(vector3(planeNormal, 'planeNormal'), 'planeNormal');
  if (![ -1, 0, 1 ].includes(crossingDirection)) {
    throw new RangeError('crossingDirection must be -1, 0, or 1');
  }

  for (let index = 0; index < samples.length - 1; index += 1) {
    const a = samples[index];
    const b = samples[index + 1];
    const da = dot(subtract(a.state.slice(0, 3), point), normal);
    const db = dot(subtract(b.state.slice(0, 3), point), normal);
    const descending = da > EPS && db <= EPS;
    const ascending = da < -EPS && db >= -EPS;
    const crosses = crossingDirection === -1
      ? descending
      : crossingDirection === 1
        ? ascending
        : descending || ascending;
    if (!crosses) continue;

    const denominator = da - db;
    if (Math.abs(denominator) <= EPS) continue;
    return interpolateSample(a, b, da / denominator);
  }
  return null;
}

export function resolveScorePoints(points, context = {}) {
  if (points === undefined) return 1;
  if (Number.isFinite(points)) return Number(points);
  if (!points || typeof points !== 'object' || Array.isArray(points)) {
    throw new RangeError('points must be a number or an object');
  }

  const keys = [context.variant, context.phase, 'default'].filter(Boolean);
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(points, key)) {
      return finite(points[key], `points.${key}`);
    }
  }
  return 1;
}

function normalizeResult(result, target, context) {
  if (!result || typeof result !== 'object') {
    throw new RangeError(`scoring method ${target.kind} returned no result`);
  }
  if (typeof result.classification !== 'string' || !result.classification) {
    throw new RangeError(`scoring method ${target.kind} must return a classification`);
  }
  const isScore = Boolean(result.isScore);
  const status = result.status ?? (isScore ? 'scored' : 'miss');
  const scoreRank = Number.isFinite(result.scoreRank)
    ? Number(result.scoreRank)
    : isScore
      ? 3
      : status === 'collision'
        ? 2
        : 1;
  const normalized = {
    method: target.kind,
    status,
    isScore,
    points: isScore ? resolveScorePoints(target.points, context) : 0,
    scoreRank,
    clearanceMargin: -Infinity,
    missDistance: isScore ? 0 : Infinity,
    entrySample: null,
    collisionPoint: null,
    geometry: null,
    ...result,
  };
  normalized.method = target.kind;
  normalized.isScore = isScore;
  normalized.status = status;
  normalized.scoreRank = scoreRank;
  normalized.points = isScore ? resolveScorePoints(target.points, context) : 0;
  return normalized;
}

function projectileRadius(piece) {
  if (!piece || typeof piece !== 'object') throw new RangeError('piece must be an object');
  return positive(piece.collisionRadius ?? piece.radius, 'piece.collisionRadius');
}

function evaluate2026Hub({samples, target, piece}) {
  const radius = projectileRadius(piece);
  const geometry = createHubGeometry({
    centerX: finite(target.centerX ?? 0, 'target.centerX'),
    centerY: finite(target.centerY ?? 0, 'target.centerY'),
  });
  const interaction = classifyHubInteraction(samples, geometry, radius);
  const isScore = interaction.classification === 'clean-entry';
  const collision = interaction.classification === 'rim-collision'
    || interaction.classification === 'funnel-collision';
  return {
    ...interaction,
    isScore,
    status: isScore ? 'scored' : collision ? 'collision' : 'miss',
    scoreRank: isScore ? 3 : collision ? 2 : 1,
    entrySample: interaction.topCrossing,
    geometry,
  };
}

function evaluateTopCircle({samples, target, piece}) {
  const radius = projectileRadius(piece);
  const height = finite(target.height, 'target.height');
  const centerX = finite(target.centerX ?? 0, 'target.centerX');
  const centerY = finite(target.centerY ?? 0, 'target.centerY');
  const openingRadius = positive(target.openingRadius, 'target.openingRadius');
  const crossingDirection = target.crossingDirection ?? -1;
  const crossing = findPlaneCrossing(
    samples,
    [0, 0, height],
    [0, 0, 1],
    crossingDirection,
  );
  const geometry = {kind: 'top-circle', centerX, centerY, height, openingRadius};
  if (!crossing) {
    return {classification: 'miss', isScore: false, status: 'miss', geometry};
  }

  const distance = Math.hypot(crossing.state[0] - centerX, crossing.state[1] - centerY);
  const clearanceMargin = openingRadius - radius - distance;
  if (clearanceMargin >= -EPS) {
    return {
      classification: 'clean-entry',
      isScore: true,
      status: 'scored',
      scoreRank: 3,
      clearanceMargin: Math.max(0, clearanceMargin),
      missDistance: 0,
      entrySample: crossing,
      topCrossing: crossing,
      geometry,
    };
  }
  if (distance <= openingRadius + radius + EPS) {
    return {
      classification: 'rim-collision',
      isScore: false,
      status: 'collision',
      scoreRank: 2,
      clearanceMargin,
      missDistance: 0,
      entrySample: crossing,
      topCrossing: crossing,
      collisionPoint: crossing,
      geometry,
    };
  }
  return {
    classification: 'miss',
    isScore: false,
    status: 'miss',
    scoreRank: 1,
    clearanceMargin,
    missDistance: Math.max(0, distance - openingRadius - radius),
    entrySample: crossing,
    topCrossing: crossing,
    geometry,
  };
}

function apertureBasis(target) {
  const planePoint = vector3(target.planePoint, 'target.planePoint');
  const normal = normalize(vector3(target.planeNormal, 'target.planeNormal'), 'target.planeNormal');
  const requestedUp = vector3(target.up ?? [0, 0, 1], 'target.up');
  const projectedUp = subtract(requestedUp, scale(normal, dot(requestedUp, normal)));
  const up = normalize(projectedUp, 'target.up projected into scoring plane');
  const right = normalize(cross(up, normal), 'target aperture right axis');
  const center = vector3(target.apertureCenter ?? planePoint, 'target.apertureCenter');
  return {planePoint, normal, up, right, center};
}

function rectangleAperture(crossing, target, basis, radius) {
  const width = positive(target.width, 'target.width');
  const height = positive(target.height, 'target.height');
  const offset = subtract(crossing.state.slice(0, 3), basis.center);
  const horizontal = Math.abs(dot(offset, basis.right));
  const vertical = Math.abs(dot(offset, basis.up));
  const horizontalClearance = width / 2 - radius - horizontal;
  const verticalClearance = height / 2 - radius - vertical;
  const clearanceMargin = Math.min(horizontalClearance, verticalClearance);
  const touchesFrame = horizontal <= width / 2 + radius + EPS
    && vertical <= height / 2 + radius + EPS;
  const missDistance = Math.hypot(
    Math.max(0, horizontal - width / 2 - radius),
    Math.max(0, vertical - height / 2 - radius),
  );
  return {clearanceMargin, touchesFrame, missDistance, width, height};
}

function circleAperture(crossing, target, basis, radius) {
  const openingRadius = positive(target.openingRadius, 'target.openingRadius');
  const offset = subtract(crossing.state.slice(0, 3), basis.center);
  const horizontal = dot(offset, basis.right);
  const vertical = dot(offset, basis.up);
  const distance = Math.hypot(horizontal, vertical);
  return {
    clearanceMargin: openingRadius - radius - distance,
    touchesFrame: distance <= openingRadius + radius + EPS,
    missDistance: Math.max(0, distance - openingRadius - radius),
    openingRadius,
  };
}

function evaluatePlaneAperture({samples, target, piece}) {
  const radius = projectileRadius(piece);
  const basis = apertureBasis(target);
  const crossingDirection = target.crossingDirection ?? -1;
  const crossing = findPlaneCrossing(
    samples,
    basis.planePoint,
    basis.normal,
    crossingDirection,
  );
  const shape = target.shape ?? 'rectangle';
  if (!['rectangle', 'circle'].includes(shape)) {
    throw new RangeError('target.shape must be rectangle or circle');
  }
  const geometry = {
    kind: 'plane-aperture',
    shape,
    planePoint: basis.planePoint,
    planeNormal: basis.normal,
    apertureCenter: basis.center,
    up: basis.up,
    right: basis.right,
  };
  if (!crossing) {
    return {classification: 'miss', isScore: false, status: 'miss', geometry};
  }

  const aperture = shape === 'circle'
    ? circleAperture(crossing, target, basis, radius)
    : rectangleAperture(crossing, target, basis, radius);
  Object.assign(geometry, shape === 'circle'
    ? {openingRadius: aperture.openingRadius}
    : {width: aperture.width, height: aperture.height});

  if (aperture.clearanceMargin >= -EPS) {
    return {
      classification: 'clean-entry',
      isScore: true,
      status: 'scored',
      scoreRank: 3,
      clearanceMargin: Math.max(0, aperture.clearanceMargin),
      missDistance: 0,
      entrySample: crossing,
      geometry,
    };
  }
  if (aperture.touchesFrame) {
    return {
      classification: 'rim-collision',
      collisionType: 'frame',
      isScore: false,
      status: 'collision',
      scoreRank: 2,
      clearanceMargin: aperture.clearanceMargin,
      missDistance: 0,
      entrySample: crossing,
      collisionPoint: crossing,
      geometry,
    };
  }
  return {
    classification: 'miss',
    isScore: false,
    status: 'miss',
    scoreRank: 1,
    clearanceMargin: aperture.clearanceMargin,
    missDistance: aperture.missDistance,
    entrySample: crossing,
    geometry,
  };
}

export function registerScoringMethod(kind, evaluator, {replace = false} = {}) {
  if (typeof kind !== 'string' || !kind.trim()) throw new RangeError('scoring method kind must be a non-empty string');
  if (typeof evaluator !== 'function') throw new RangeError('scoring method evaluator must be a function');
  const key = kind.trim();
  if (scoringMethods.has(key) && !replace) {
    throw new RangeError(`scoring method already registered: ${key}`);
  }
  scoringMethods.set(key, evaluator);
  return key;
}

export function hasScoringMethod(kind) {
  return scoringMethods.has(kind);
}

export function listScoringMethods() {
  return [...scoringMethods.keys()].sort();
}

export function scoreTrajectory(samples, target, piece, context = {}) {
  if (!target || typeof target !== 'object' || Array.isArray(target)) {
    throw new RangeError('target must be an object');
  }
  if (typeof target.kind !== 'string' || !target.kind) {
    throw new RangeError('target.kind must be a non-empty string');
  }
  const evaluator = scoringMethods.get(target.kind);
  if (!evaluator) throw new RangeError(`unknown scoring method: ${target.kind}`);
  return normalizeResult(evaluator({samples, target, piece, context}), target, context);
}

registerScoringMethod('2026-hex-hub', evaluate2026Hub);
registerScoringMethod('top-circle', evaluateTopCircle);
registerScoringMethod('plane-aperture', evaluatePlaneAperture);
