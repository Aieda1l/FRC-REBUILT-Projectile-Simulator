// Pure polygon geometry for arbitrary FRC scoring openings (local plane coordinates, meters).
const EPS = 1e-9;
const finitePoint = (p) => Array.isArray(p) && p.length === 2 && p.every((n) => Number.isFinite(n) && Math.abs(n) <= 20);

export function polygonArea(points) {
  let twice = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return twice / 2;
}

function orient(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function segmentsIntersect(a, b, c, d) {
  const x = orient(a, b, c), y = orient(a, b, d);
  const u = orient(c, d, a), v = orient(c, d, b);
  if (Math.abs(x) <= EPS && onSegment(a, b, c)) return true;
  if (Math.abs(y) <= EPS && onSegment(a, b, d)) return true;
  if (Math.abs(u) <= EPS && onSegment(c, d, a)) return true;
  if (Math.abs(v) <= EPS && onSegment(c, d, b)) return true;
  return x * y < 0 && u * v < 0;
}

function onSegment(a, b, p) {
  return p[0] >= Math.min(a[0], b[0]) - EPS
    && p[0] <= Math.max(a[0], b[0]) + EPS
    && p[1] >= Math.min(a[1], b[1]) - EPS
    && p[1] <= Math.max(a[1], b[1]) + EPS;
}

export function validatePolygonVertices(vertices) {
  if (!Array.isArray(vertices) || vertices.length < 3 || vertices.length > 64
    || !vertices.every(finitePoint)) {
    throw new RangeError('Custom opening needs 3–64 finite vertices in the ±20 m drawing plane');
  }
  const points = vertices.map((p) => [...p]);
  if (Math.abs(polygonArea(points)) < 1e-5) {
    throw new RangeError('Opening must have a nonzero area');
  }
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-4) {
      throw new RangeError('Opening has duplicate or nearly coincident vertices');
    }
    for (let j = i + 1; j < points.length; j++) {
      if (j === i + 1 || (i === 0 && j === points.length - 1)) continue;
      if (segmentsIntersect(a, b, points[j], points[(j + 1) % points.length])) {
        throw new RangeError('Opening outline crosses itself');
      }
    }
  }
  return points;
}

export function pointSegmentDistance(point, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const length2 = dx * dx + dy * dy;
  const t = length2 <= EPS ? 0 : Math.max(0, Math.min(1,
    ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2,
  ));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

// Returns whether the center is inside, signed clearance to nearest boundary,
// and the nearest boundary's inward normal (used for orientation-aware clearance).
export function polygonClearance(point, vertices) {
  let inside = false;
  let nearest = Infinity, nearestNormal = [0, 0];
  const winding = Math.sign(polygonArea(vertices)) || 1;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i], b = vertices[(i + 1) % vertices.length];
    if ((a[1] > point[1]) !== (b[1] > point[1])
      && point[0] < a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])) {
      inside = !inside;
    }
    const distance = pointSegmentDistance(point, a, b);
    if (distance < nearest) {
      nearest = distance;
      const dx = b[0] - a[0], dy = b[1] - a[1], norm = Math.hypot(dx, dy);
      nearestNormal = [-dy * winding / norm, dx * winding / norm];
    }
  }
  return {inside: inside || nearest <= EPS, distance: nearest,
    margin: (inside || nearest <= EPS ? 1 : -1) * nearest,
    normal: nearestNormal};
}
