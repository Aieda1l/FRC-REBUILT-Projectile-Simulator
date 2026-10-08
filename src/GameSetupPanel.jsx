import React, {useEffect, useMemo, useState} from 'react';
import {
  GAME_PIECES, SCORING_TARGETS, PIECE_SHAPES, TARGET_KINDS,
  loadLibrary, newCustomId, resolveLibrarySelection, saveLibrary,
  upsertLibraryItem, validatePiece, validateTarget,
} from './gameCatalog.js';

const fieldClass = 'w-full rounded bg-slate-900 border border-slate-600 px-2 py-1.5 text-sm text-white focus:border-cyan-400';
const buttonClass = 'rounded border border-slate-600 px-3 py-1.5 text-xs text-slate-100 hover:border-cyan-400 disabled:opacity-40';
const kinds = {
  hub: 'Hexagonal funnel / hub',
  hoop: 'Horizontal circular hoop',
  slot: 'Vertical rectangular slot / hole',
  'round-slot': 'Vertical circular hole',
};

function NumericField({label, value, onChange, step = 0.01, min, max}) {
  return (
    <label className="block text-xs text-slate-300">
      {label}
      <input className={fieldClass} type="number" step={step} min={min} max={max}
        required value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function PieceEditor({draft, onChange, onSave, onCancel, isNew, disabled}) {
  const update = (key, value) => onChange((current) => ({...current, [key]: value}));
  return (
    <form onSubmit={(event) => {event.preventDefault(); onSave();}}
      className="mt-3 space-y-3 rounded-lg border border-slate-600 bg-slate-900/60 p-3">
      <h4 className="text-sm font-semibold text-cyan-300">{isNew ? 'Create game piece' : 'Edit saved game piece'}</h4>
      <label className="block text-xs text-slate-300">Name
        <input className={fieldClass} required maxLength={80} value={draft.name}
          onChange={(event) => update('name', event.target.value)} />
      </label>
      <label className="block text-xs text-slate-300">Shape
        <select className={fieldClass} value={draft.shape} onChange={(event) => update('shape', event.target.value)}>
          {PIECE_SHAPES.map((shape) => <option key={shape} value={shape}>{shape}</option>)}
        </select>
      </label>
      <div className="grid grid-cols-2 gap-3">
        <NumericField label="Mass (kg)" min={0.00001} step={0.001} value={draft.mass} onChange={(value) => update('mass', value)} />
        <NumericField label="Diameter (m)" min={0.00001} value={draft.diameter} onChange={(value) => update('diameter', value)} />
        <NumericField label="Drag coefficient Cd" min={0} value={draft.dragCoeff} onChange={(value) => update('dragCoeff', value)} />
        <NumericField label="Lift coefficient Cl" min={0} value={draft.liftCoeff} onChange={(value) => update('liftCoeff', value)} />
      </div>
      {draft.shape !== 'sphere' && <p className="text-xs text-amber-300" role="note">
        Disc and ring flight is a sphere-equivalent approximation. The simulator does not model orientation, tumbling, or non-spherical opening clearance.
      </p>}
      <div className="flex gap-2">
        <button className={buttonClass + ' border-cyan-500 text-cyan-200'} disabled={disabled} type="submit">Save game piece</button>
        <button className={buttonClass} type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function TargetEditor({draft, onChange, onSave, onCancel, isNew, disabled}) {
  const update = (key, value) => onChange((current) => ({...current, [key]: value}));
  const setKind = (kind) => {
    const template = SCORING_TARGETS.find((target) => target.kind === kind);
    onChange((current) => ({
      ...template, id: current.id, name: current.name, x: current.x,
      lateralY: current.lateralY, z: current.z,
    }));
  };
  return (
    <form onSubmit={(event) => {event.preventDefault(); onSave();}}
      className="mt-3 space-y-3 rounded-lg border border-slate-600 bg-slate-900/60 p-3">
      <h4 className="text-sm font-semibold text-cyan-300">{isNew ? 'Create scoring target' : 'Edit saved scoring target'}</h4>
      <label className="block text-xs text-slate-300">Name
        <input className={fieldClass} required maxLength={80} value={draft.name}
          onChange={(event) => update('name', event.target.value)} />
      </label>
      <label className="block text-xs text-slate-300">Scoring geometry
        <select className={fieldClass} value={draft.kind} onChange={(event) => setKind(event.target.value)}>
          {TARGET_KINDS.map((kind) => <option key={kind} value={kind}>{kinds[kind]}</option>)}
        </select>
      </label>
      <div className="grid grid-cols-2 gap-3">
        <NumericField label="Forward X (m)" value={draft.x} onChange={(value) => update('x', value)} />
        <NumericField label="Lateral center (m)" value={draft.lateralY} onChange={(value) => update('lateralY', value)} />
        <NumericField label={draft.kind === 'hub' ? 'Top height (m)' : 'Opening center height (m)'}
          value={draft.z} min={0.001} onChange={(value) => update('z', value)} />
        <NumericField label="Points per score" step={1} min={0} max={1000}
          value={draft.points ?? 1} onChange={(value) => update('points', value)} />
        {draft.kind === 'hub' && <>
          <NumericField label="Top across flats (m)" min={0.001} value={draft.topAcrossFlats} onChange={(value) => update('topAcrossFlats', value)} />
          <NumericField label="Bottom hex side (m)" min={0.001} value={draft.bottomSide} onChange={(value) => update('bottomSide', value)} />
          <NumericField label="Funnel panel height (m)" min={0.001} value={draft.panelHeight} onChange={(value) => update('panelHeight', value)} />
        </>}
        {(draft.kind === 'hoop' || draft.kind === 'round-slot') &&
          <NumericField label="Opening diameter (m)" min={0.001} value={draft.diameter} onChange={(value) => update('diameter', value)} />}
        {draft.kind === 'slot' && <>
          <NumericField label="Opening width (m)" min={0.001} value={draft.width} onChange={(value) => update('width', value)} />
          <NumericField label="Opening height (m)" min={0.001} value={draft.height} onChange={(value) => update('height', value)} />
        </>}
      </div>
      <p className="text-xs text-slate-400">
        Hoops score on downward passage through a horizontal rim. Holes/slots score while traveling forward (+X) through a vertical opening. The ball must clear the opening with its radius.
      </p>
      <div className="flex gap-2">
        <button className={buttonClass + ' border-cyan-500 text-cyan-200'} disabled={disabled} type="submit">Save scoring target</button>
        <button className={buttonClass} type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

export default function GameSetupPanel({onSelectionChange, disabled = false}) {
  const [library, setLibrary] = useState(() => loadLibrary());
  const [message, setMessage] = useState('');
  const [pieceDraft, setPieceDraft] = useState({...GAME_PIECES[0]});
  const [targetDraft, setTargetDraft] = useState({...SCORING_TARGETS[0]});
  const [editPiece, setEditPiece] = useState(false);
  const [editTarget, setEditTarget] = useState(false);
  const [pieceNew, setPieceNew] = useState(false);
  const [targetNew, setTargetNew] = useState(false);
  const selection = useMemo(() => resolveLibrarySelection(library), [library]);

  useEffect(() => {onSelectionChange(selection);}, [onSelectionChange, selection]);

  const persist = (next, successMessage) => {
    try {
      const saved = saveLibrary(next);
      setLibrary(saved);
      setMessage(successMessage);
      return true;
    } catch (error) {
      setMessage('Could not save in this browser: ' + (error instanceof Error ? error.message : String(error)));
      return false;
    }
  };

  const choose = (category, id) => {
    if (disabled) return;
    setEditPiece(false);
    setEditTarget(false);
    persist({...library, [category]: id}, 'Selection saved on this device');
  };

  const beginPiece = (copy = false) => {
    if (disabled) return;
    setPieceDraft(copy
      ? {...selection.piece, id: newCustomId(), name: selection.piece.name + ' copy'}
      : {...selection.piece});
    setPieceNew(copy || !selection.piece.id.startsWith('custom-'));
    setEditPiece(true);
  };

  const beginTarget = (copy = false) => {
    if (disabled) return;
    setTargetDraft(copy
      ? {...selection.target, id: newCustomId(), name: selection.target.name + ' copy'}
      : {...selection.target});
    setTargetNew(copy || !selection.target.id.startsWith('custom-'));
    setEditTarget(true);
  };

  const savePiece = () => {
    try {
      const id = pieceNew ? newCustomId() : pieceDraft.id;
      const item = validatePiece({...pieceDraft, id});
      const next = upsertLibraryItem(library, 'pieces', item);
      if (persist(next, 'Game piece saved in this browser')) setEditPiece(false);
    } catch (error) {setMessage(error.message);}
  };

  const saveTarget = () => {
    try {
      const id = targetNew ? newCustomId() : targetDraft.id;
      const item = validateTarget({...targetDraft, id});
      const next = upsertLibraryItem(library, 'targets', item);
      if (persist(next, 'Scoring target saved in this browser')) setEditTarget(false);
    } catch (error) {setMessage(error.message);}
  };

  const remove = (category) => {
    if (disabled) return;
    const id = category === 'pieces' ? selection.piece.id : selection.target.id;
    if (!id.startsWith('custom-')) return;
    const selectedKey = category === 'pieces' ? 'pieceId' : 'targetId';
    const defaultId = category === 'pieces' ? GAME_PIECES[0].id : SCORING_TARGETS[0].id;
    const next = {...library, [category]: library[category].filter((item) => item.id !== id), [selectedKey]: defaultId};
    if (persist(next, 'Saved profile deleted')) {
      setEditPiece(false);
      setEditTarget(false);
    }
  };

  return (
    <section className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 space-y-4" aria-label="Game setup">
      <div>
        <h2 className="text-lg font-semibold text-indigo-400">Game Setup Library</h2>
        <p className="text-xs text-slate-400 mt-1">Use a historic preset or build your own. All custom profiles and selections save to this browser.</p>
      </div>
      <div>
        <label className="block text-sm text-slate-200 mb-1" htmlFor="piece-selector">Game piece</label>
        <select id="piece-selector" className={fieldClass} disabled={disabled} value={library.pieceId}
          onChange={(event) => choose('pieceId', event.target.value)}>
          <optgroup label="FRC presets">
            {GAME_PIECES.map((piece) => <option key={piece.id} value={piece.id}>{piece.name}</option>)}
          </optgroup>
          {library.pieces.length > 0 && <optgroup label="Saved custom pieces">
            {library.pieces.map((piece) => <option key={piece.id} value={piece.id}>{piece.name}</option>)}
          </optgroup>}
        </select>
        <div className="mt-2 flex flex-wrap gap-2">
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => beginPiece(false)}>
            {selection.piece.id.startsWith('custom-') ? 'Edit' : 'Customize'}
          </button>
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => beginPiece(true)}>Duplicate</button>
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => {
            setPieceDraft({...GAME_PIECES[0], id: newCustomId(), name: 'New game piece'});
            setPieceNew(true); setEditPiece(true);
          }}>New piece</button>
          {selection.piece.id.startsWith('custom-') &&
            <button type="button" className={buttonClass + ' text-rose-300'} disabled={disabled} onClick={() => remove('pieces')}>Delete</button>}
        </div>
        {editPiece && <PieceEditor draft={pieceDraft} onChange={setPieceDraft} isNew={pieceNew}
          onSave={savePiece} onCancel={() => setEditPiece(false)} disabled={disabled} />}
      </div>
      <div className="border-t border-slate-700 pt-3">
        <label className="block text-sm text-slate-200 mb-1" htmlFor="target-selector">Scoring method / target</label>
        <select id="target-selector" className={fieldClass} disabled={disabled} value={library.targetId}
          onChange={(event) => choose('targetId', event.target.value)}>
          <optgroup label="FRC target presets">
            {SCORING_TARGETS.map((target) => <option key={target.id} value={target.id}>{target.name}</option>)}
          </optgroup>
          {library.targets.length > 0 && <optgroup label="Saved custom targets">
            {library.targets.map((target) => <option key={target.id} value={target.id}>{target.name}</option>)}
          </optgroup>}
        </select>
        <div className="mt-2 flex flex-wrap gap-2">
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => beginTarget(false)}>
            {selection.target.id.startsWith('custom-') ? 'Edit' : 'Customize'}
          </button>
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => beginTarget(true)}>Duplicate</button>
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => {
            setTargetDraft({...SCORING_TARGETS[0], id: newCustomId(), name: 'New scoring target'});
            setTargetNew(true); setEditTarget(true);
          }}>New target</button>
          {selection.target.id.startsWith('custom-') &&
            <button type="button" className={buttonClass + ' text-rose-300'} disabled={disabled} onClick={() => remove('targets')}>Delete</button>}
        </div>
        {editTarget && <TargetEditor draft={targetDraft} onChange={setTargetDraft} isNew={targetNew}
          onSave={saveTarget} onCancel={() => setEditTarget(false)} disabled={disabled} />}
      </div>
      {selection.piece.shape !== 'sphere' &&
        <p className="text-xs text-amber-300">Non-spherical pieces use a sphere-equivalent flight and clearance approximation; do not use for final shooter design without calibration.</p>}
      {message && <p role="status" className="text-xs text-cyan-300">{message}</p>}
    </section>
  );
}
