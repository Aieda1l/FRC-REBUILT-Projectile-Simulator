export const INCH_TO_METER = 0.0254;

export const HUB_DIMENSIONS = Object.freeze({
  topAcrossFlats: 41.727 * INCH_TO_METER,
  topZ: 72.0 * INCH_TO_METER,
  bottomSide: 18.92 * INCH_TO_METER,
  panelHeight: 17.90 * INCH_TO_METER,
});

const EPS = 1e-12;
const SQRT3 = Math.sqrt(3);

export function hexVertices(apothem, z, centerX = 0, centerY = 0) {
  const circumradius = 2 * apothem / SQRT3;
  return Array.from({length: 6}, (_, index) => {
    const angle = Math.PI / 6 + index * Math.PI / 3;
    return [
      centerX + circumradius * Math.cos(angle),
      centerY + circumradius * Math.sin(angle),
      z,
    ];
  });
}

export function createHubGeometry({
  centerX = 0,
  centerY = 0,
  topZ = HUB_DIMENSIONS.topZ,
  topAcrossFlats = HUB_DIMENSIONS.topAcrossFlats,
  bottomSide = HUB_DIMENSIONS.bottomSide,
  panelHeight = HUB_DIMENSIONS.panelHeight,
} = {}) {
  const topApothem = topAcrossFlats / 2;
  const bottomAcrossFlats = SQRT3 * bottomSide;
  const bottomApothem = bottomAcrossFlats / 2;
  const bottomZ = topZ - panelHeight;
  const slope = (topApothem - bottomApothem) / (topZ - bottomZ);
  const normals = Array.from({length: 6}, (_, index) => {
    const angle = index * Math.PI / 3;
    return [Math.cos(angle), Math.sin(angle)];
  });

  return {
    centerX,
    centerY,
    topZ,
    bottomZ,
    topApothem,
    bottomApothem,
    bottomAcrossFlats,
    slope,
    normals,
    topVertices: hexVertices(topApothem, topZ, centerX, centerY),
    bottomVertices: hexVertices(bottomApothem, bottomZ, centerX, centerY),
  };
}

function interpolateSample(a, b, alpha) {
  const clamped = Math.max(0, Math.min(1, alpha));
  return {
    time: a.time + clamped * (b.time - a.time),
    state: a.state.map((value, index) => (
      value + clamped * (b.state[index] - value)
    )),
  };
}

function descendingCrossing(samples, height, afterTime = -Infinity) {
  for (let index = 0; index < samples.length - 1; index += 1) {
    const a = samples[index];
    const b = samples[index + 1];
    if (b.time < afterTime - EPS) continue;
    const za = a.state[2] - height;
    const zb = b.state[2] - height;
    if (za > EPS && zb <= EPS && b.state[5] < 0) {
      const alpha = za / (za - zb);
      const crossing = interpolateSample(a, b, alpha);
      crossing.state[2] = height;
      if (crossing.time + EPS >= afterTime) return crossing;
    }
  }
  return null;
}

function rawHexClearance(x, y, apothem, geometry) {
  const localX = x - geometry.centerX;
  const localY = y - geometry.centerY;
  let clearance = Infinity;
  for (const [nx, ny] of geometry.normals) {
    clearance = Math.min(clearance, apothem - (nx * localX + ny * localY));
  }
  return clearance;
}

function pointSegmentDistance2D(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const denom = dx * dx + dy * dy;
  if (denom <= EPS) return Math.hypot(px - ax, py - ay);
  const projection = ((px - ax) * dx + (py - ay) * dy) / denom;
  const t = Math.max(0, Math.min(1, projection));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function minBoundaryDistance(x, y, vertices) {
  let distance = Infinity;
  for (let index = 0; index < vertices.length; index += 1) {
    const a = vertices[index];
    const b = vertices[(index + 1) % vertices.length];
    distance = Math.min(
      distance,
      pointSegmentDistance2D(x, y, a[0], a[1], b[0], b[1]),
    );
  }
  return distance;
}

function panelApothemAt(z, geometry) {
  return geometry.bottomApothem + geometry.slope * (z - geometry.bottomZ);
}

function panelClearance(state, normal, geometry, ballRadius) {
  const [x, y, z] = state;
  const [nx, ny] = normal;
  const localX = x - geometry.centerX;
  const localY = y - geometry.centerY;
  const signedNumerator = panelApothemAt(z, geometry) - (nx * localX + ny * localY);
  return signedNumerator / Math.sqrt(1 + geometry.slope ** 2) - ballRadius;
}

function failure(classification, {
  topCrossing = null,
  bottomCrossing = null,
  collisionPoint = null,
  clearanceMargin = -Infinity,
  missDistance = 0,
} = {}) {
  return {
    classification,
    topCrossing,
    bottomCrossing,
    collisionPoint,
    clearanceMargin,
    missDistance,
  };
}

export function classifyHubInteraction(samples, geometry = createHubGeometry(), ballRadius) {
  if (!Array.isArray(samples) || samples.length < 2) {
    return failure('miss', {missDistance: Infinity});
  }
  if (!Number.isFinite(ballRadius) || ballRadius <= 0) {
    throw new RangeError('ballRadius must be finite and positive');
  }

  const topCrossing = descendingCrossing(samples, geometry.topZ);
  if (!topCrossing) {
    return failure('miss', {missDistance: Infinity});
  }

  const [topX, topY] = topCrossing.state;
  const rawTopClearance = rawHexClearance(
    topX,
    topY,
    geometry.topApothem,
    geometry,
  );
  const topEdgeClearance = rawTopClearance - ballRadius;

  if (topEdgeClearance < -EPS) {
    const boundaryDistance = minBoundaryDistance(
      topX,
      topY,
      geometry.topVertices,
    );
    if (boundaryDistance <= ballRadius + EPS) {
      return failure('rim-collision', {
        topCrossing,
        collisionPoint: topCrossing,
        clearanceMargin: topEdgeClearance,
      });
    }
    return failure('miss', {
      topCrossing,
      clearanceMargin: topEdgeClearance,
      missDistance: Math.max(EPS, boundaryDistance - ballRadius),
    });
  }

  const bottomCrossing = descendingCrossing(
    samples,
    geometry.bottomZ,
    topCrossing.time + EPS,
  );

  if (!bottomCrossing) {
    return failure('miss', {
      topCrossing,
      clearanceMargin: topEdgeClearance,
      missDistance: Infinity,
    });
  }

  const path = [topCrossing];
  for (const current of samples) {
    if (
      current.time > topCrossing.time + EPS
      && current.time < bottomCrossing.time - EPS
    ) {
      path.push(current);
    }
  }
  path.push(bottomCrossing);

  let minimumClearance = topEdgeClearance;
  let firstCollision = null;

  for (let index = 0; index < path.length - 1; index += 1) {
    const a = path[index];
    const b = path[index + 1];
    for (const normal of geometry.normals) {
      const c0 = panelClearance(a.state, normal, geometry, ballRadius);
      const c1 = panelClearance(b.state, normal, geometry, ballRadius);
      minimumClearance = Math.min(minimumClearance, c0, c1);

      if (!firstCollision && c0 < -EPS) {
        firstCollision = a;
      } else if (!firstCollision && c0 >= -EPS && c1 < -EPS) {
        const alpha = c0 / (c0 - c1);
        firstCollision = interpolateSample(a, b, alpha);
      }
    }
    if (firstCollision) break;
  }

  if (firstCollision) {
    return failure('funnel-collision', {
      topCrossing,
      bottomCrossing,
      collisionPoint: firstCollision,
      clearanceMargin: Math.min(minimumClearance, -EPS),
    });
  }

  const [bottomX, bottomY] = bottomCrossing.state;
  const bottomEdgeClearance = rawHexClearance(
    bottomX,
    bottomY,
    geometry.bottomApothem,
    geometry,
  ) - ballRadius;
  minimumClearance = Math.min(minimumClearance, bottomEdgeClearance);

  if (bottomEdgeClearance < -EPS) {
    return failure('funnel-collision', {
      topCrossing,
      bottomCrossing,
      collisionPoint: bottomCrossing,
      clearanceMargin: bottomEdgeClearance,
    });
  }

  return {
    classification: 'clean-entry',
    topCrossing,
    bottomCrossing,
    collisionPoint: null,
    clearanceMargin: minimumClearance,
    missDistance: 0,
  };
}
