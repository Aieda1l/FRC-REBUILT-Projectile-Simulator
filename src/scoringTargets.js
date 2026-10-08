import {classifyHubInteraction, createHubGeometry, hexVertices} from './hubGeometry.js';
import {SCORING_TARGETS, validateTarget} from './gameCatalog.js';

const EPS = 1e-10;

export function createTargetGeometry(input = SCORING_TARGETS[0]) {
  const target = validateTarget(input);
  if (target.kind === 'hub') {
    return {
      kind: 'hub',
      ...createHubGeometry({
        centerX: target.x,
        centerY: target.lateralY,
        topZ: target.z,
        topAcrossFlats: target.topAcrossFlats,
        bottomSide: target.bottomSide,
        panelHeight: target.panelHeight,
      }),
    };
  }
  return {...target};
}

function interpolate(a, b, fraction) {
  const t = Math.max(0, Math.min(1, fraction));
  return {
    time: a.time + t * (b.time - a.time),
    state: a.state.map((n, i) => n + t * (b.state[i] - n)),
  };
}

function crossing(samples, axis, plane, direction) {
  if (!Array.isArray(samples) || samples.length < 2) return null;
  for (let i = 0; i < samples.length - 1; i += 1) {
    const a = samples[i], b = samples[i + 1];
    const before = direction * (a.state[axis] - plane);
    const after = direction * (b.state[axis] - plane);
    if (before > EPS || after < -EPS || b.state[axis + 3] * direction <= EPS) continue;
    if (Math.abs(b.state[axis] - a.state[axis]) <= EPS) continue;
    const hit = interpolate(a, b, (plane - a.state[axis]) / (b.state[axis] - a.state[axis]));
    hit.state[axis] = plane;
    return hit;
  }
  return null;
}

function outsideOrRim(sample, signedClearance, outsideDistance, ballRadius) {
  if (signedClearance >= -EPS) {
    return {
      classification: 'clean-entry',
      topCrossing: sample,
      bottomCrossing: null,
      collisionPoint: null,
      clearanceMargin: Math.max(0, signedClearance),
      missDistance: 0,
    };
  }
  // Distances beyond the outer frame do not represent rim / frame contact.
  const isContact = outsideDistance <= ballRadius + EPS;
  return {
    classification: isContact ? 'rim-collision' : 'miss',
    topCrossing: sample,
    bottomCrossing: null,
    collisionPoint: isContact ? sample : null,
    clearanceMargin: signedClearance,
    missDistance: isContact ? 0 : outsideDistance - ballRadius,
  };
}

export function classifyTargetInteraction(samples, geometry, ballRadius) {
  if (!Number.isFinite(ballRadius) || ballRadius <= 0) {
    throw new RangeError('ballRadius must be positive');
  }
  if (geometry.kind === 'hub') return classifyHubInteraction(samples, geometry, ballRadius);
  const horizontal = geometry.kind === 'hoop';
  const sample = horizontal
    ? crossing(samples, 2, geometry.z, -1)
    : crossing(samples, 0, geometry.x, 1);
  if (!sample) {
    return {
      classification: 'miss', topCrossing: null, bottomCrossing: null,
      collisionPoint: null, clearanceMargin: -Infinity, missDistance: Infinity,
    };
  }
  const lateralOffset = sample.state[1] - geometry.lateralY;
  if (geometry.kind === 'hoop') {
    const radialDistance = Math.hypot(sample.state[0] - geometry.x, lateralOffset);
    const apertureRadius = geometry.diameter / 2;
    return outsideOrRim(sample, apertureRadius - radialDistance - ballRadius,
      Math.abs(radialDistance - apertureRadius), ballRadius);
  }
  const heightOffset = sample.state[2] - geometry.z;
  if (geometry.kind === 'round-slot') {
    const radialDistance = Math.hypot(lateralOffset, heightOffset);
    const apertureRadius = geometry.diameter / 2;
    return outsideOrRim(sample, apertureRadius - radialDistance - ballRadius,
      Math.abs(radialDistance - apertureRadius), ballRadius);
  }
  if (geometry.kind === 'slot') {
    const dx = Math.abs(lateralOffset) - geometry.width / 2;
    const dz = Math.abs(heightOffset) - geometry.height / 2;
    const signedClearance = Math.min(-dx, -dz) - ballRadius;
    const outsideDistance = Math.hypot(Math.max(0, dx), Math.max(0, dz));
    // A ball inside but too near a rectangular edge is a frame collision.
    return outsideOrRim(sample, signedClearance, outsideDistance, ballRadius);
  }
  throw new RangeError('Unsupported scoring target kind: ' + geometry.kind);
}

function circleVertices(cx, cy, cz, radius, horizontal, segments = 48) {
  return Array.from({length: segments}, (_, i) => {
    const angle = i * 2 * Math.PI / segments;
    return horizontal
      ? [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle), cz]
      : [cx, cy + radius * Math.cos(angle), cz + radius * Math.sin(angle)];
  });
}

function rectangleVertices(geometry, width, height) {
  const y = geometry.lateralY, z = geometry.z, x = geometry.x;
  return [
    [x, y - width / 2, z - height / 2],
    [x, y + width / 2, z - height / 2],
    [x, y + width / 2, z + height / 2],
    [x, y - width / 2, z + height / 2],
  ];
}

export function targetWireframes(geometry, radius) {
  if (geometry.kind === 'hub') {
    return {
      frames: [geometry.topVertices, geometry.bottomVertices],
      clearance: [
        hexVertices(Math.max(0.0001, geometry.topApothem - radius), geometry.topZ, geometry.centerX, geometry.centerY),
        hexVertices(Math.max(0.0001, geometry.bottomApothem - radius), geometry.bottomZ, geometry.centerX, geometry.centerY),
      ],
      connectors: geometry.topVertices.map((vertex, i) => [vertex, geometry.bottomVertices[i]]),
    };
  }
  if (geometry.kind === 'slot') {
    return {
      frames: [rectangleVertices(geometry, geometry.width, geometry.height)],
      clearance: geometry.width > 2 * radius && geometry.height > 2 * radius
        ? [rectangleVertices(geometry, geometry.width - 2 * radius, geometry.height - 2 * radius)]
        : [],
      connectors: [],
    };
  }
  const r = geometry.diameter / 2;
  const horizontal = geometry.kind === 'hoop';
  return {
    frames: [circleVertices(geometry.x, geometry.lateralY, geometry.z, r, horizontal)],
    clearance: r > radius
      ? [circleVertices(geometry.x, geometry.lateralY, geometry.z, r - radius, horizontal)]
      : [],
    connectors: [],
  };
}

// Orthographic side view: y is suppressed; for vertical targets show their height
// at the goal plane, for horizontal targets show the diameter at their rim height.
export function targetSideProfile(geometry, radius) {
  if (geometry.kind === 'hub') {
    return {
      polygon: [
        [geometry.centerX - geometry.topApothem, geometry.topZ],
        [geometry.centerX - geometry.bottomApothem, geometry.bottomZ],
        [geometry.centerX + geometry.bottomApothem, geometry.bottomZ],
        [geometry.centerX + geometry.topApothem, geometry.topZ],
      ],
      clearances: [
        [[geometry.centerX - Math.max(0, geometry.topApothem - radius), geometry.topZ],
          [geometry.centerX + Math.max(0, geometry.topApothem - radius), geometry.topZ]],
        [[geometry.centerX - Math.max(0, geometry.bottomApothem - radius), geometry.bottomZ],
          [geometry.centerX + Math.max(0, geometry.bottomApothem - radius), geometry.bottomZ]],
      ],
      labelPoint: [geometry.centerX, geometry.topZ],
    };
  }
  if (geometry.kind === 'hoop') {
    return {
      polygon: [
        [geometry.x - geometry.diameter / 2, geometry.z],
        [geometry.x + geometry.diameter / 2, geometry.z],
      ],
      clearances: [[
        [geometry.x - Math.max(0, geometry.diameter / 2 - radius), geometry.z],
        [geometry.x + Math.max(0, geometry.diameter / 2 - radius), geometry.z],
      ]],
      labelPoint: [geometry.x, geometry.z],
    };
  }
  const height = geometry.kind === 'slot' ? geometry.height : geometry.diameter;
  return {
    polygon: [[geometry.x, geometry.z - height / 2], [geometry.x, geometry.z + height / 2]],
    clearances: [[
      [geometry.x, geometry.z - Math.max(0, height / 2 - radius)],
      [geometry.x, geometry.z + Math.max(0, height / 2 - radius)],
    ]],
    labelPoint: [geometry.x, geometry.z],
  };
}
