export const CAMERA_PRESETS = Object.freeze({
  isometric: Object.freeze({yaw: Math.PI / 4, pitch: Math.PI / 5}),
  front: Object.freeze({yaw: 0, pitch: 0}),
  side: Object.freeze({yaw: Math.PI / 2, pitch: 0}),
  top: Object.freeze({yaw: 0, pitch: Math.PI / 2}),
});

const EPS = 1e-9;

function rotatePoint([x, y, z], {yaw = 0, pitch = 0}) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const yawX = cy * x - sy * y;
  const yawY = sy * x + cy * y;
  return {
    horizontal: yawX,
    vertical: cp * z - sp * yawY,
    depth: sp * z + cp * yawY,
  };
}

export function projectScene(points, camera, {width = 640, height = 420, padding = 30} = {}) {
  const rotated = points.map((point) => rotatePoint(point, camera));
  if (!rotated.length) return {points: [], scale: 1};
  const xs = rotated.map((point) => point.horizontal);
  const ys = rotated.map((point) => point.vertical);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const extentX = Math.max(maxX - minX, EPS);
  const extentY = Math.max(maxY - minY, EPS);
  const usableWidth = Math.max(width - 2 * padding, 1);
  const usableHeight = Math.max(height - 2 * padding, 1);
  const scale = Math.min(usableWidth / extentX, usableHeight / extentY);
  const contentWidth = extentX * scale;
  const contentHeight = extentY * scale;
  const offsetX = (width - contentWidth) / 2;
  const offsetY = (height - contentHeight) / 2;

  return {
    scale,
    points: rotated.map((point) => ({
      x: offsetX + (point.horizontal - minX) * scale,
      y: height - offsetY - (point.vertical - minY) * scale,
      depth: point.depth,
    })),
  };
}
