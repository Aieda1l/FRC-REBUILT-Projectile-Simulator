const DEG = Math.PI / 180;
const EPS = 1e-9;
export const CAMERA_PRESETS = Object.freeze({
  isometric: Object.freeze({name: 'isometric', yaw: -45 * DEG, pitch: -35.264 * DEG}),
  front: Object.freeze({name: 'front', yaw: 90 * DEG, pitch: 0}),
  side: Object.freeze({name: 'side', yaw: 0, pitch: 0}),
  top: Object.freeze({name: 'top', yaw: 0, pitch: -90 * DEG}),
});
function rotatePoint([x, y, z], camera) {
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  return {x: x1, y: -(sp * y1 + cp * z), depth: cp * y1 - sp * z};
}
export function projectScene(points, camera, {width = 600, height = 400, padding = 30} = {}) {
  if (!Array.isArray(points) || points.length === 0) {
    return {points: [], scale: 1, bounds: {minX: 0, maxX: 0, minY: 0, maxY: 0}};
  }
  const raw = points.map((point) => rotatePoint(point, camera));
  const xs = raw.map((p) => p.x), ys = raw.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const extentX = Math.max(EPS, maxX - minX), extentY = Math.max(EPS, maxY - minY);
  const scale = Math.min(
    Math.max(EPS, width - 2 * padding) / extentX,
    Math.max(EPS, height - 2 * padding) / extentY,
  );
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  return {
    points: raw.map((p) => ({x: width / 2 + (p.x - cx) * scale, y: height / 2 + (p.y - cy) * scale, depth: p.depth})),
    scale,
    bounds: {minX, maxX, minY, maxY},
  };
}


export function projectTrajectoryGroups(
  {actual = [], ideal = [], envelopes = [], context = []},
  camera,
  options = {},
) {
  const world = [
    ...context,
    ...actual,
    ...ideal,
    ...envelopes.flat(),
  ];
  const projected = projectScene(world, camera, options);
  let offset = 0;
  const take = (count) => {
    const chunk = projected.points.slice(offset, offset + count);
    offset += count;
    return chunk;
  };

  const projectedContext = take(context.length);
  const projectedActual = take(actual.length);
  const projectedIdeal = take(ideal.length);
  const projectedEnvelopes = envelopes.map((trajectory) => take(trajectory.length));

  return {
    scale: projected.scale,
    context: projectedContext,
    actual: projectedActual,
    ideal: projectedIdeal,
    envelopes: projectedEnvelopes,
  };
}
