import React, {useEffect, useMemo, useRef, useState} from 'react';
import {targetWireframes} from './scoringTargets.js';
import {gamePieceWireframe} from './gamePieceGeometry.js';
import {CAMERA_PRESETS, projectTrajectoryGroups} from './trajectory3dProjection.js';

const WIDTH = 600;
const HEIGHT = 400;
const PRESET_LABELS = {isometric: 'Isometric', front: 'Front', side: 'Side', top: 'Top'};
const pointsAttribute = (points) => points.map((p) => `${p.x},${p.y}`).join(' ');

export default function Trajectory3DView({samples, idealSamples = [], envelopeSamples = [], hubGeometry, interaction, ballRadius, gamePiece = {shape:'sphere', diameter:0.15}}) {
  const [preset, setPreset] = useState('isometric');
  const [camera, setCamera] = useState({...CAMERA_PRESETS.isometric});
  const [sampleIndex, setSampleIndex] = useState(Math.max(0, samples.length - 1));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.25);
  const dragRef = useRef(null);

  useEffect(() => {
    setSampleIndex((i) => Math.min(i, Math.max(0, samples.length - 1)));
  }, [samples.length]);

  useEffect(() => {
    if (!playing) return;
    if (sampleIndex >= samples.length - 1) return;
    const dt = samples.length > 1 ? Math.max(1e-5, samples[1].time - samples[0].time) : 0.01;
    const advance = Math.max(1,Math.round(0.033 * speed / dt));
    const timer = setInterval(() => setSampleIndex((i)=>Math.min(samples.length - 1,i+advance)),33);
    return () => clearInterval(timer);
  }, [playing, speed, samples, sampleIndex]);

  const setCameraPreset = (name) => {
    setPreset(name);
    setCamera({...CAMERA_PRESETS[name]});
  };

  const scene = useMemo(() => {
    const wires = targetWireframes(hubGeometry, ballRadius);
    const centerX = hubGeometry.kind === 'hub' ? hubGeometry.centerX : hubGeometry.x;
    const centerY = hubGeometry.kind === 'hub' ? hubGeometry.centerY : hubGeometry.lateralY;
    const trajectory = samples.map((sample) => sample.state.slice(0, 3));
    const ideal = idealSamples.map((sample) => sample.state.slice(0, 3));
    const envelopes = envelopeSamples.map((group) => group.map((sample) => sample.state.slice(0, 3)));
    const axes = [
      [centerX, centerY, 0],
      [centerX + 0.6, centerY, 0],
      [centerX, centerY + 0.6, 0],
      [centerX, centerY, 0.6],
    ];
    const collision = interaction?.collisionPoint?.state?.slice(0, 3) ?? null;
    const selectedSample = samples[Math.min(sampleIndex, Math.max(0, samples.length - 1))];
    const piece = selectedSample
      ? gamePieceWireframe(gamePiece, selectedSample.state.slice(0,3), selectedSample.orientation)
      : {lines:[],points:[]};
    const context = [
      ...wires.frames.flat(),
      ...wires.clearance.flat(),
      ...wires.connectors.flat(),
      ...piece.lines.flat(),
      ...piece.points.flat(),
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
      frames: wires.frames.map((frame) => takeContext(frame.length)),
      clearances: wires.clearance.map((frame) => takeContext(frame.length)),
      connectors: wires.connectors.map((pair) => takeContext(pair.length)),
      pieceLines: piece.lines.map((line) => takeContext(line.length)),
      pieceEdges: piece.points.map((line) => takeContext(line.length)),
      axes: takeContext(4),
      collision: collision ? takeContext(1)[0] : null,
      trajectory: projected.actual,
      ideal: projected.ideal,
      envelopes: projected.envelopes,
    };
  }, [samples, idealSamples, envelopeSamples, hubGeometry, interaction, ballRadius, camera, gamePiece, sampleIndex]);

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
        aria-label="3-D trajectory and scoring target view">
        {scene.connectors.map((pair, index) => <line key={'connector-' + index}
          x1={pair[0].x} y1={pair[0].y} x2={pair[1].x} y2={pair[1].y}
          stroke="rgba(34,197,94,0.55)" strokeWidth="1.5" />)}
        {scene.frames.map((frame, index) => <polygon key={'frame-' + index}
          points={pointsAttribute(frame)} fill="rgba(34,197,94,0.06)"
          stroke="#22c55e" strokeWidth="2.5" />)}
        {scene.clearances.map((frame, index) => <polygon key={'clearance-' + index}
          points={pointsAttribute(frame)} fill="none" stroke="#67e8f9"
          strokeDasharray="5 4" opacity="0.8" />)}
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
        {scene.pieceLines.map((line,i) => <polygon key={'piece-outline-'+i}
          points={pointsAttribute(line)} fill={i===0 ? 'rgba(251,191,36,0.10)' : 'none'}
          stroke="#fbbf24" strokeWidth="1.8" strokeLinejoin="round"/>)}
        {scene.pieceEdges.map((line,i) => <polyline key={'piece-edge-'+i}
          points={pointsAttribute(line)} fill="none" stroke="#f59e0b" strokeWidth="1.5"/>)}
        {marker && <circle cx={marker.x} cy={marker.y} r="2" fill="#f8fafc" />}
        {scene.collision && <>
          <line x1={scene.collision.x-8} y1={scene.collision.y-8} x2={scene.collision.x+8} y2={scene.collision.y+8} stroke="#ef4444" strokeWidth="3" />
          <line x1={scene.collision.x-8} y1={scene.collision.y+8} x2={scene.collision.x+8} y2={scene.collision.y-8} stroke="#ef4444" strokeWidth="3" />
        </>}
      </svg>
      <p className="text-xs text-amber-200">Amber outline = true-size {gamePiece.shape} · diameter {(gamePiece.diameter*100).toFixed(1)} cm · drag to orbit isometric view</p>
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
        <button type="button" className="border border-slate-600 rounded px-2 py-1"
          onClick={() => {
            if (sampleIndex >= samples.length - 1) {
              setSampleIndex(0);
              setPlaying(true);
            } else setPlaying((value)=>!value);
          }}>{playing && sampleIndex < samples.length - 1 ? 'Pause playback' : 'Play trajectory'}</button>
        <label>Playback speed <select className="bg-slate-900 border border-slate-600 rounded p-1"
          value={speed} onChange={(event)=>setSpeed(Number(event.target.value))}>
          {[0.25,0.5,1,2].map((value)=><option key={value} value={value}>{value}×</option>)}
        </select></label>
      </div>
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
