import React, {useMemo, useRef, useState} from 'react';
import {hexVertices} from './hubGeometry.js';
import {CAMERA_PRESETS, projectScene} from './trajectory3dProjection.js';

const WIDTH = 640;
const HEIGHT = 420;

function pathFrom(points, close = false) {
  if (!points.length) return '';
  return points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ')
    + (close ? ' Z' : '');
}

export default function Trajectory3DView({samples, hubGeometry, interaction, ballRadius}) {
  const [preset, setPreset] = useState('isometric');
  const [camera, setCamera] = useState({...CAMERA_PRESETS.isometric});
  const [sampleIndex, setSampleIndex] = useState(0);
  const dragRef = useRef(null);

  const safeIndex = Math.min(Math.max(sampleIndex, 0), Math.max(samples.length - 1, 0));
  const topGuide = hexVertices(
    Math.max(hubGeometry.topApothem - ballRadius, 1e-6),
    hubGeometry.topZ,
    hubGeometry.centerX,
    hubGeometry.centerY,
  );
  const bottomGuide = hexVertices(
    Math.max(hubGeometry.bottomApothem - ballRadius, 1e-6),
    hubGeometry.bottomZ,
    hubGeometry.centerX,
    hubGeometry.centerY,
  );

  const scene = useMemo(() => {
    const trajectory = samples.map((sample) => sample.state.slice(0, 3));
    const axes = [
      [hubGeometry.centerX, hubGeometry.centerY, 0],
      [hubGeometry.centerX + 0.55, hubGeometry.centerY, 0],
      [hubGeometry.centerX, hubGeometry.centerY + 0.55, 0],
      [hubGeometry.centerX, hubGeometry.centerY, 0.55],
    ];
    const collision = interaction?.collisionPoint?.state?.slice(0, 3) ?? [];
    const all = [
      ...hubGeometry.topVertices,
      ...hubGeometry.bottomVertices,
      ...topGuide,
      ...bottomGuide,
      ...trajectory,
      ...axes,
      ...(collision.length ? [collision] : []),
    ];
    const projection = projectScene(all, camera, {width: WIDTH, height: HEIGHT, padding: 34});
    let offset = 0;
    const take = (count) => {
      const value = projection.points.slice(offset, offset + count);
      offset += count;
      return value;
    };
    return {
      scale: projection.scale,
      top: take(6),
      bottom: take(6),
      topGuide: take(6),
      bottomGuide: take(6),
      trajectory: take(trajectory.length),
      axes: take(4),
      collision: collision.length ? take(1)[0] : null,
    };
  }, [samples, hubGeometry, interaction, ballRadius, camera]);

  const ball = scene.trajectory[safeIndex] ?? null;
  const ballPixelRadius = Math.max(4, Math.min(14, scene.scale * ballRadius));

  const selectPreset = (name) => {
    setPreset(name);
    setCamera({...CAMERA_PRESETS[name]});
  };

  const onPointerDown = (event) => {
    if (preset !== 'isometric') return;
    dragRef.current = {x: event.clientX, y: event.clientY, yaw: camera.yaw, pitch: camera.pitch};
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag || preset !== 'isometric') return;
    setCamera({
      yaw: drag.yaw + (event.clientX - drag.x) * 0.008,
      pitch: Math.max(-1.45, Math.min(1.45, drag.pitch - (event.clientY - drag.y) * 0.008)),
    });
  };

  const onPointerUp = () => {
    dragRef.current = null;
  };

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-2">
        {Object.keys(CAMERA_PRESETS).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => selectPreset(name)}
            className={`px-2 py-1 text-xs rounded border ${
              preset === name
                ? 'border-cyan-400 text-cyan-300 bg-cyan-500/10'
                : 'border-slate-600 text-slate-300'
            }`}
          >
            {name[0].toUpperCase() + name.slice(1)}
          </button>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full h-auto bg-slate-900/50 rounded-lg touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        {[0, 1, 2, 3, 4, 5].map((index) => {
          const next = (index + 1) % 6;
          return (
            <path
              key={`panel-${index}`}
              d={pathFrom([scene.top[index], scene.top[next], scene.bottom[next], scene.bottom[index]], true)}
              fill="rgba(34,197,94,0.08)"
              stroke="#22c55e"
              strokeWidth="1.5"
            />
          );
        })}
        <path d={pathFrom(scene.top, true)} fill="none" stroke="#4ade80" strokeWidth="2"/>
        <path d={pathFrom(scene.bottom, true)} fill="none" stroke="#22c55e" strokeWidth="2"/>
        <path d={pathFrom(scene.topGuide, true)} fill="none" stroke="#67e8f9" strokeDasharray="5 5" strokeWidth="1"/>
        <path d={pathFrom(scene.bottomGuide, true)} fill="none" stroke="#67e8f9" strokeDasharray="5 5" strokeWidth="1"/>
        <path d={pathFrom(scene.trajectory)} fill="none" stroke="#818cf8" strokeWidth="3"/>

        {scene.axes.length === 4 && (
          <>
            <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[1].x} y2={scene.axes[1].y} stroke="#f87171"/>
            <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[2].x} y2={scene.axes[2].y} stroke="#60a5fa"/>
            <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[3].x} y2={scene.axes[3].y} stroke="#facc15"/>
          </>
        )}
        {ball && <circle cx={ball.x} cy={ball.y} r={ballPixelRadius} fill="#ef4444" stroke="white" strokeWidth="2"/>}
        {scene.collision && (
          <g>
            <circle cx={scene.collision.x} cy={scene.collision.y} r="8" fill="none" stroke="#fb923c" strokeWidth="3"/>
            <path
              d={`M ${scene.collision.x - 6} ${scene.collision.y - 6} L ${scene.collision.x + 6} ${scene.collision.y + 6} M ${scene.collision.x + 6} ${scene.collision.y - 6} L ${scene.collision.x - 6} ${scene.collision.y + 6}`}
              stroke="#fb923c"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>

      <label className="block mt-2 text-xs text-slate-400">
        Trajectory position
        <input
          type="range"
          min="0"
          max={Math.max(samples.length - 1, 0)}
          value={safeIndex}
          onChange={(event) => setSampleIndex(Number(event.target.value))}
          className="w-full mt-1 accent-indigo-500"
        />
      </label>
    </div>
  );
}
