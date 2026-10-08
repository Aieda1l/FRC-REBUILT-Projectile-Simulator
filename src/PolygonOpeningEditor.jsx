import React, {useRef, useState} from 'react';
import {polygonClearance, validatePolygonVertices} from './polygonGeometry.js';

const W = 360, H = 260;
const field = 'w-20 rounded border border-slate-600 bg-slate-950 px-1 py-0.5 text-xs text-slate-100';

export default function PolygonOpeningEditor({vertices, onChange}) {
  const [halfSpan, setHalfSpan] = useState(2);
  const [snapping, setSnapping] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const drag = useRef(null);
  const scale = W / (2 * halfSpan);
  const projected = vertices.map(([u, v]) => [W / 2 + u * scale, H / 2 - v * scale]);
  const path = projected.map(([x, y]) => x + ',' + y).join(' ');
  const valid = (() => {
    try {validatePolygonVertices(vertices); return true;} catch {return false;}
  })();
  const localPoint = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) * W / rect.width;
    const y = (event.clientY - rect.top) * H / rect.height;
    const raw = [(x - W / 2) / scale, (H / 2 - y) / scale];
    return raw.map((n) => Math.max(-20, Math.min(20,
      Math.round(n / (snapping ? 0.05 : 0.001)) * (snapping ? 0.05 : 0.001))));
  };
  const updateVertex = (index, pos) =>
    onChange(vertices.map((point, i) => i === index ? pos : point));
  const pointerDown = (event) => {
    if (event.button !== 0) return;
    const index = event.target.dataset.vertex;
    if (index !== undefined) {
      drag.current = Number(index);
      event.currentTarget.setPointerCapture(event.pointerId);
    } else if (drawing && vertices.length < 64) {
      onChange([...vertices, localPoint(event)]);
    }
  };
  const pointerMove = (event) => {
    if (drag.current !== null) updateVertex(drag.current, localPoint(event));
  };
  const stopDrag = () => {drag.current = null;};
  const clearances = valid ? (() => {
    const c = polygonClearance([0, 0], vertices);
    return c.inside ? 'Center-to-edge: ' + (c.distance * 100).toFixed(1) + ' cm' : 'Origin is outside opening';
  })() : 'Connect at least 3 vertices without crossing edges';

  return (
    <div className="rounded-lg border border-cyan-800 bg-slate-950/70 p-3 space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button type="button" className={'rounded px-2 py-1 border ' + (drawing ? 'text-cyan-300 border-cyan-400' : 'text-slate-300 border-slate-600')}
          onClick={() => setDrawing((v) => !v)} aria-pressed={drawing}>
          {drawing ? 'Finish adding vertices' : 'Add vertices'}
        </button>
        <button type="button" className="border border-slate-600 rounded px-2 py-1"
          onClick={() => {onChange([]); setDrawing(true);}}>Start blank</button>
        <button type="button" className="border border-slate-600 rounded px-2 py-1"
          onClick={() => onChange([[-0.5,-0.3],[0.5,-0.3],[0.5,0.3],[-0.5,0.3]])}>Reset rectangle</button>
        <label className="flex gap-1 items-center"><input type="checkbox" checked={snapping}
          onChange={(event) => setSnapping(event.target.checked)}/> Snap 5 cm</label>
      </div>
      <label className="text-xs flex items-center gap-2 text-slate-300">
        View half-width <select className={field} value={halfSpan}
          onChange={(event) => setHalfSpan(Number(event.target.value))}>
          {[0.5,1,2,4,8,16].map((n) => <option key={n} value={n}>{n} m</option>)}
        </select>
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full border border-slate-700 rounded touch-none select-none bg-slate-900"
        aria-label="Custom scoring opening polygon editor" role="img"
        onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={stopDrag} onPointerCancel={stopDrag}>
        <defs><pattern id="polygon-drawing-grid" width={scale / 2} height={scale / 2} patternUnits="userSpaceOnUse">
          <path d={`M ${scale / 2} 0 L 0 0 0 ${scale / 2}`} fill="none" stroke="#334155" strokeWidth="0.8"/>
        </pattern></defs>
        <rect width={W} height={H} fill="url(#polygon-drawing-grid)"/>
        <line x1={W / 2} y1="0" x2={W / 2} y2={H} stroke="#64748b" strokeDasharray="4 4"/>
        <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#64748b" strokeDasharray="4 4"/>
        {vertices.length >= 3 && <polygon points={path} fill="#22d3ee" fillOpacity="0.16"
          stroke={valid ? '#22d3ee' : '#f87171'} strokeWidth="2" />}
        {vertices.length === 2 && <polyline points={path} fill="none" stroke="#22d3ee" strokeWidth="2"/>}
        {projected.map(([x,y],i) => <g key={i}>
          <circle data-vertex={i} cx={x} cy={y} r="9" fill="#67e8f9" fillOpacity="0.2" stroke="none"/>
          <circle data-vertex={i} cx={x} cy={y} r="5" fill="#22d3ee" stroke="#f8fafc" strokeWidth="1"/>
          <text x={x + 9} y={y - 7} fontSize="10" fill="#e2e8f0">{i + 1}</text>
        </g>)}
        <text x="10" y="16" fontSize="10" fill="#94a3b8">{drawing ? 'Click to add · drag vertices' : 'Drag vertices to reshape'}</text>
      </svg>
      <p className={'text-xs ' + (valid ? 'text-green-300' : 'text-amber-300')} role="status">
        {vertices.length} vertices · {clearances}
      </p>
      <p className="text-xs text-slate-400">Coordinates are meters relative to the target center. Vertical: horizontal/height. Horizontal: forward/lateral. Drag nodes or enter exact coordinates below. Use the X button to remove a vertex.</p>
      <div className="max-h-40 overflow-y-auto space-y-1">
        {vertices.map((vertex, index) => <div key={index} className="flex items-center justify-between gap-2 text-xs">
          <span className="text-slate-400 w-6">{index + 1}</span>
          <label className="flex gap-1 items-center">U <input aria-label={`Vertex ${index + 1} U`} type="number" step=".01" className={field}
            value={vertex[0]} onChange={(e) => updateVertex(index, [Number(e.target.value), vertex[1]])}/></label>
          <label className="flex gap-1 items-center">V <input aria-label={`Vertex ${index + 1} V`} type="number" step=".01" className={field}
            value={vertex[1]} onChange={(e) => updateVertex(index, [vertex[0], Number(e.target.value)])}/></label>
          <button aria-label={`Remove vertex ${index + 1}`} className="text-rose-300 hover:text-rose-100 px-2"
            type="button" onClick={() => onChange(vertices.filter((_,i) => i !== index))}>×</button>
        </div>)}
      </div>
    </div>
  );
}
