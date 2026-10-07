import React, {useEffect, useMemo, useRef, useState} from 'react';
import {hexVertices} from './hubGeometry.js';
import {CAMERA_PRESETS, projectTrajectoryGroups} from './trajectory3dProjection.js';

const WIDTH = 600;
const HEIGHT = 400;
const PRESET_LABELS = {isometric: 'Isometric', front: 'Front', side: 'Side', top: 'Top'};
const pointsAttribute = (points) => points.map((p) => `${p.x},${p.y}`).join(' ');

export default function Trajectory3DView({samples, idealSamples = [], envelopeSamples = [], hubGeometry, interaction, ballRadius}) {
  const [preset, setPreset] = useState('isometric');
  const [camera, setCamera] = useState({...CAMERA_PRESETS.isometric});
  const [sampleIndex, setSampleIndex] = useState(Math.max(0, samples.length - 1));
  const dragRef = useRef(null);

  useEffect(() => {
    setSampleIndex((i) => Math.min(i, Math.max(0, samples.length - 1)));
  }, [samples.length]);

  const setCameraPreset = (name) => {
    setPreset(name);
    setCamera({...CAMERA_PRESETS[name]});
  };

  const scene = useMemo(() => {
    const topClearance = hexVertices(Math.max(0.001, hubGeometry.topApothem - ballRadius), hubGeometry.topZ, hubGeometry.centerX, hubGeometry.centerY);
    const bottomClearance = hexVertices(Math.max(0.001, hubGeometry.bottomApothem - ballRadius), hubGeometry.bottomZ, hubGeometry.centerX, hubGeometry.centerY);
    const trajectory = samples.map((sample) => sample.state.slice(0, 3));
    const ideal = idealSamples.map((sample) => sample.state.slice(0, 3));
    const envelopes = envelopeSamples.map((group) => group.map((sample) => sample.state.slice(0, 3)));
    const axes = [
      [hubGeometry.centerX, hubGeometry.centerY, 0],
      [hubGeometry.centerX + 0.6, hubGeometry.centerY, 0],
      [hubGeometry.centerX, hubGeometry.centerY + 0.6, 0],
      [hubGeometry.centerX, hubGeometry.centerY, 0.6],
    ];
    const collision = interaction?.collisionPoint?.state?.slice(0, 3) ?? null;
    const context = [
      ...hubGeometry.topVertices,
      ...hubGeometry.bottomVertices,
      ...topClearance,
      ...bottomClearance,
      ...axes,
      ...(collision ? [collision] : []),
    ];
    const projected = projectTrajectoryGroups(
      {context, actual: trajectory, ideal, envelopes},
      camera,
      {width: WIDTH, height: HEIGHT, padding: 34},
    );
    let offset = 0;
    const takeContext = (count) => {
      const chunk = projected.context.slice(offset, offset + count);
      offset += count;
      return chunk;
    };
    return {
      top: takeContext(6),
      bottom: takeContext(6),
      topClearance: takeContext(6),
      bottomClearance: takeContext(6),
      axes: takeContext(4),
      collision: collision ? takeContext(1)[0] : null,
      trajectory: projected.actual,
      ideal: projected.ideal,
      envelopes: projected.envelopes,
    };
  }, [samples, idealSamples, envelopeSamples, hubGeometry, interaction, ballRadius, camera]);

  const marker = scene.trajectory[Math.min(sampleIndex, Math.max(0, scene.trajectory.length - 1))];
  const onPointerDown = (event) => {
    if (preset !== 'isometric') return;
    dragRef.current = {x: event.clientX, y: event.clientY};
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const onPointerMove = (event) => {
    if (!dragRef.current || preset !== 'isometric') return;
    const dx = event.clientX - dragRef.current.x, dy = event.clientY - dragRef.current.y;
    dragRef.current = {x: event.clientX, y: event.clientY};
    setCamera((current) => ({
      ...current,
      yaw: current.yaw + dx * 0.01,
      pitch: Math.max(-Math.PI * 0.47, Math.min(Math.PI * 0.47, current.pitch + dy * 0.01)),
    }));
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {Object.entries(PRESET_LABELS).map(([name, label]) => (
          <button key={name} type="button" onClick={() => setCameraPreset(name)}
            className={`px-3 py-1 rounded text-xs border ${preset === name ? 'border-cyan-400 bg-cyan-400/10 text-cyan-300' : 'border-slate-600 text-slate-300'}`}>
            {label}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full bg-slate-900/80 rounded-lg touch-none"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={() => { dragRef.current = null; }} onPointerCancel={() => { dragRef.current = null; }}
        aria-label="3-D trajectory and HUB view">
        {[0,1,2,3,4,5].map((i) => {
          const n = (i + 1) % 6;
          return <polygon key={i} points={pointsAttribute([scene.top[i], scene.top[n], scene.bottom[n], scene.bottom[i]])}
            fill="rgba(34,197,94,0.06)" stroke="rgba(34,197,94,0.55)" strokeWidth="1.5" />;
        })}
        <polygon points={pointsAttribute(scene.top)} fill="none" stroke="#22c55e" strokeWidth="2.5" />
        <polygon points={pointsAttribute(scene.bottom)} fill="none" stroke="#22c55e" strokeWidth="2" />
        <polygon points={pointsAttribute(scene.topClearance)} fill="none" stroke="#67e8f9" strokeDasharray="5 4" />
        <polygon points={pointsAttribute(scene.bottomClearance)} fill="none" stroke="#67e8f9" strokeDasharray="5 4" opacity="0.65" />
        {scene.envelopes.map((trajectory, index) => (
          trajectory.length > 1 && (
            <polyline
              key={'envelope-' + index}
              points={pointsAttribute(trajectory)}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1"
              opacity="0.3"
            />
          )
        ))}
        {scene.ideal.length > 1 && (
          <polyline
            points={pointsAttribute(scene.ideal)}
            fill="none"
            stroke="#22d3ee"
            strokeWidth="2"
            strokeDasharray="8 4"
            opacity="0.75"
          />
        )}
        {scene.trajectory.length > 1 && <polyline points={pointsAttribute(scene.trajectory)} fill="none" stroke="#818cf8" strokeWidth="3" strokeLinecap="round" />}
        {scene.axes.length === 4 && <>
          <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[1].x} y2={scene.axes[1].y} stroke="#ef4444" strokeWidth="2" />
          <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[2].x} y2={scene.axes[2].y} stroke="#22c55e" strokeWidth="2" />
          <line x1={scene.axes[0].x} y1={scene.axes[0].y} x2={scene.axes[3].x} y2={scene.axes[3].y} stroke="#38bdf8" strokeWidth="2" />
        </>}
        {marker && <circle cx={marker.x} cy={marker.y} r="7" fill="#f8fafc" stroke="#f97316" strokeWidth="3" />}
        {scene.collision && <>
          <line x1={scene.collision.x-8} y1={scene.collision.y-8} x2={scene.collision.x+8} y2={scene.collision.y+8} stroke="#ef4444" strokeWidth="3" />
          <line x1={scene.collision.x-8} y1={scene.collision.y+8} x2={scene.collision.x+8} y2={scene.collision.y-8} stroke="#ef4444" strokeWidth="3" />
        </>}
      </svg>
      <label className="block text-xs text-slate-400">
        Trajectory position
        <input aria-label="Trajectory position" type="range" min="0" max={Math.max(0, samples.length - 1)}
          value={Math.min(sampleIndex, Math.max(0, samples.length - 1))}
          onChange={(event) => setSampleIndex(Number(event.target.value))}
          className="w-full mt-1 accent-cyan-400" />
      </label>
    </div>
  );
}
