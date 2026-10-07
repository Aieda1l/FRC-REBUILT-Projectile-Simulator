import React, {useMemo, useState} from 'react';
import {
  DEFAULT_GAME_PROFILE,
  GAME_PROFILE_SCHEMA,
  getGamePiece,
  normalizeGamePiece,
  normalizeGameProfile,
} from './gameProfiles.js';
import {
  deleteGameProfile,
  loadSavedGameProfiles,
  saveGameProfile,
} from './gameProfileStorage.js';

function numeric(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function resolvePiece(profile) {
  return typeof profile.gamePiece === 'string'
    ? getGamePiece(profile.gamePiece)
    : normalizeGamePiece(profile.gamePiece);
}

function draftFromProfile(profile) {
  const piece = resolvePiece(profile);
  const scoring = profile.scoring;
  const isPlane = scoring.kind === 'plane-aperture';
  const planeCenter = scoring.apertureCenter ?? scoring.planePoint ?? [0, 0, 2];
  return {
    profileId: profile.id,
    profileName: profile.name,
    pieceName: piece.name,
    mass: piece.mass,
    diameter: piece.radius * 2,
    collisionDiameter: piece.collisionRadius * 2,
    dragCoeff: piece.dragCoeff,
    liftCoeff: piece.liftCoeff,
    scoringType: scoring.kind === '2026-hex-hub'
      ? '2026-hex-hub'
      : scoring.kind === 'top-circle'
        ? 'top-circle'
        : isPlane && scoring.shape === 'circle'
          ? 'plane-circle'
          : 'plane-rectangle',
    targetX: scoring.centerX ?? planeCenter[0] ?? 0,
    targetHeight: scoring.height ?? planeCenter[2] ?? 2,
    openingDiameter: scoring.openingRadius ? scoring.openingRadius * 2 : 1,
    openingWidth: scoring.width ?? 1,
    openingHeight: scoring.height && isPlane ? scoring.height : 0.5,
    points: numeric(scoring.points?.default ?? scoring.points, 1),
  };
}

function buildProfile(draft) {
  const id = draft.profileId.trim();
  const pieceId = `${id}-piece`;
  const center = [numeric(draft.targetX), 0, numeric(draft.targetHeight)];
  let scoring;

  if (draft.scoringType === '2026-hex-hub') {
    scoring = {
      kind: '2026-hex-hub',
      centerX: numeric(draft.targetX),
      centerY: 0,
      points: {default: numeric(draft.points, 1)},
    };
  } else if (draft.scoringType === 'top-circle') {
    scoring = {
      kind: 'top-circle',
      height: numeric(draft.targetHeight),
      centerX: numeric(draft.targetX),
      centerY: 0,
      openingRadius: numeric(draft.openingDiameter) / 2,
      crossingDirection: -1,
      points: {default: numeric(draft.points, 1)},
    };
  } else {
    scoring = {
      kind: 'plane-aperture',
      planePoint: center,
      planeNormal: [-1, 0, 0],
      apertureCenter: center,
      up: [0, 0, 1],
      crossingDirection: -1,
      shape: draft.scoringType === 'plane-circle' ? 'circle' : 'rectangle',
      ...(draft.scoringType === 'plane-circle'
        ? {openingRadius: numeric(draft.openingDiameter) / 2}
        : {
            width: numeric(draft.openingWidth),
            height: numeric(draft.openingHeight),
          }),
      points: {default: numeric(draft.points, 1)},
    };
  }

  return normalizeGameProfile({
    schema: GAME_PROFILE_SCHEMA,
    id,
    name: draft.profileName.trim(),
    gamePiece: {
      id: pieceId,
      name: draft.pieceName.trim(),
      mass: numeric(draft.mass),
      radius: numeric(draft.diameter) / 2,
      collisionRadius: numeric(draft.collisionDiameter) / 2,
      dragCoeff: numeric(draft.dragCoeff),
      liftCoeff: numeric(draft.liftCoeff),
    },
    scoring,
  });
}

function Field({label, value, onChange, type = 'number', step = 'any', min, unit}) {
  return (
    <label className="block text-xs text-slate-300">
      <span className="mb-1 flex justify-between gap-2">
        <span>{label}</span>
        {unit ? <span className="text-slate-500">{unit}</span> : null}
      </span>
      <input
        type={type}
        value={value}
        step={type === 'number' ? step : undefined}
        min={type === 'number' ? min : undefined}
        onChange={(event) => onChange(type === 'number' ? numeric(event.target.value) : event.target.value)}
        className="w-full rounded border border-slate-600 bg-slate-900/60 px-2 py-2 text-base text-cyan-300"
      />
    </label>
  );
}

export default function GameProfilePanel({activeProfile, onActiveProfileChange}) {
  const [open, setOpen] = useState(false);
  const [savedProfiles, setSavedProfiles] = useState(() => {
    try {
      return loadSavedGameProfiles(window.localStorage).profiles;
    } catch {
      return [];
    }
  });
  const [draft, setDraft] = useState(() => draftFromProfile(activeProfile));
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const profiles = useMemo(
    () => [DEFAULT_GAME_PROFILE, ...savedProfiles.filter((profile) => profile.id !== DEFAULT_GAME_PROFILE.id)],
    [savedProfiles],
  );
  const activePiece = useMemo(() => resolvePiece(activeProfile), [activeProfile]);

  const patchDraft = (key, value) => setDraft((current) => ({...current, [key]: value}));

  const selectProfile = (id) => {
    const profile = profiles.find((entry) => entry.id === id) ?? DEFAULT_GAME_PROFILE;
    onActiveProfileChange(profile);
    setDraft(draftFromProfile(profile));
    setMessage(profile.id === DEFAULT_GAME_PROFILE.id ? 'Using built-in 2026 profile.' : `Loaded ${profile.name}.`);
    setError('');
  };

  const createNew = () => {
    const next = draftFromProfile(DEFAULT_GAME_PROFILE);
    next.profileId = '';
    next.profileName = '';
    next.pieceName = '';
    next.scoringType = 'top-circle';
    setDraft(next);
    setMessage('');
    setError('');
  };

  const saveDraft = () => {
    try {
      const profile = buildProfile(draft);
      if (profile.id === DEFAULT_GAME_PROFILE.id) {
        throw new RangeError('the built-in 2026 profile ID is reserved');
      }
      const next = saveGameProfile(profile, window.localStorage);
      setSavedProfiles(next);
      onActiveProfileChange(profile);
      setDraft(draftFromProfile(profile));
      setMessage(`Saved and activated ${profile.name}.`);
      setError('');
    } catch (saveError) {
      setMessage('');
      setError(saveError instanceof Error ? saveError.message : String(saveError));
    }
  };

  const deleteActive = () => {
    if (activeProfile.id === DEFAULT_GAME_PROFILE.id) return;
    try {
      const next = deleteGameProfile(activeProfile.id, window.localStorage);
      setSavedProfiles(next);
      onActiveProfileChange(DEFAULT_GAME_PROFILE);
      setDraft(draftFromProfile(DEFAULT_GAME_PROFILE));
      setMessage('Deleted saved profile and restored the 2026 default.');
      setError('');
    } catch (deleteError) {
      setMessage('');
      setError(deleteError instanceof Error ? deleteError.message : String(deleteError));
    }
  };

  const isPlaneRectangle = draft.scoringType === 'plane-rectangle';
  const usesDiameter = draft.scoringType === 'top-circle' || draft.scoringType === 'plane-circle';

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-4 backdrop-blur">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between text-left"
        aria-expanded={open}
      >
        <span>
          <span className="block text-lg font-semibold text-indigo-400">Game Piece & Scoring</span>
          <span className="block text-xs font-normal text-slate-400">
            {activeProfile.name} · {activePiece.name}
          </span>
        </span>
        <span className="text-slate-400">{open ? '▼' : '▶'}</span>
      </button>

      {open ? (
        <div className="mt-4 space-y-5 border-t border-slate-700 pt-4">
          <section className="space-y-2">
            <label className="block text-xs text-slate-300">
              <span className="mb-1 block">Active saved profile</span>
              <select
                value={activeProfile.id}
                onChange={(event) => selectProfile(event.target.value)}
                className="w-full rounded border border-slate-600 bg-slate-900/60 px-2 py-2 text-base text-cyan-300"
              >
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>{profile.name}</option>
                ))}
              </select>
            </label>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={createNew}
                className="rounded border border-cyan-500/60 px-3 py-2 text-xs text-cyan-300 hover:bg-cyan-500/10">
                New Profile
              </button>
              {activeProfile.id !== DEFAULT_GAME_PROFILE.id ? (
                <button type="button" onClick={deleteActive}
                  className="rounded border border-red-500/60 px-3 py-2 text-xs text-red-300 hover:bg-red-500/10">
                  Delete Active
                </button>
              ) : null}
            </div>
          </section>

          <section>
            <h3 className="mb-2 text-sm font-semibold text-slate-200">Profile</h3>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Profile Name" type="text" value={draft.profileName}
                onChange={(value) => patchDraft('profileName', value)} />
              <Field label="Profile ID" type="text" value={draft.profileId}
                onChange={(value) => patchDraft('profileId', value)} />
            </div>
          </section>

          <section>
            <h3 className="mb-2 text-sm font-semibold text-slate-200">Game Piece</h3>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Piece Name" type="text" value={draft.pieceName}
                onChange={(value) => patchDraft('pieceName', value)} />
              <Field label="Mass" value={draft.mass} min={0.001} step={0.001} unit="kg"
                onChange={(value) => patchDraft('mass', value)} />
              <Field label="Diameter" value={draft.diameter} min={0.001} step={0.001} unit="m"
                onChange={(value) => patchDraft('diameter', value)} />
              <Field label="Collision Diameter" value={draft.collisionDiameter} min={0.001} step={0.001} unit="m"
                onChange={(value) => patchDraft('collisionDiameter', value)} />
              <Field label="Drag Coefficient" value={draft.dragCoeff} min={0} step={0.01}
                onChange={(value) => patchDraft('dragCoeff', value)} />
              <Field label="Lift Coefficient" value={draft.liftCoeff} min={0} step={0.01}
                onChange={(value) => patchDraft('liftCoeff', value)} />
            </div>
          </section>

          <section>
            <h3 className="mb-2 text-sm font-semibold text-slate-200">Scoring Target</h3>
            <label className="block text-xs text-slate-300">
              <span className="mb-1 block">Opening Type</span>
              <select
                value={draft.scoringType}
                onChange={(event) => patchDraft('scoringType', event.target.value)}
                className="w-full rounded border border-slate-600 bg-slate-900/60 px-2 py-2 text-base text-cyan-300"
              >
                <option value="top-circle">Top circular hub / opening</option>
                <option value="plane-rectangle">Vertical rectangular opening</option>
                <option value="plane-circle">Vertical circular opening</option>
                <option value="2026-hex-hub">2026 REBUILT hex hub</option>
              </select>
            </label>

            <div className="mt-2 grid grid-cols-2 gap-2">
              <Field label="Target X" value={draft.targetX} step={0.05} unit="m"
                onChange={(value) => patchDraft('targetX', value)} />
              {draft.scoringType !== '2026-hex-hub' ? (
                <Field label="Target Height" value={draft.targetHeight} min={0.01} step={0.01} unit="m"
                  onChange={(value) => patchDraft('targetHeight', value)} />
              ) : null}
              {usesDiameter ? (
                <Field label="Opening Diameter" value={draft.openingDiameter} min={0.01} step={0.01} unit="m"
                  onChange={(value) => patchDraft('openingDiameter', value)} />
              ) : null}
              {isPlaneRectangle ? (
                <>
                  <Field label="Opening Width" value={draft.openingWidth} min={0.01} step={0.01} unit="m"
                    onChange={(value) => patchDraft('openingWidth', value)} />
                  <Field label="Opening Height" value={draft.openingHeight} min={0.01} step={0.01} unit="m"
                    onChange={(value) => patchDraft('openingHeight', value)} />
                </>
              ) : null}
              <Field label="Points" value={draft.points} min={0} step={1}
                onChange={(value) => patchDraft('points', value)} />
            </div>
            {draft.scoringType.startsWith('plane-') ? (
              <p className="mt-2 text-xs text-slate-500">
                Vertical openings face the shooter and are centered at Target X / Target Height.
              </p>
            ) : null}
          </section>

          <button type="button" onClick={saveDraft}
            className="w-full rounded-lg bg-gradient-to-r from-indigo-500 to-cyan-500 py-2 font-semibold hover:from-indigo-400 hover:to-cyan-400">
            Save & Activate Profile
          </button>
          {message ? <p role="status" className="text-xs text-green-300">{message}</p> : null}
          {error ? <p role="alert" className="text-xs text-red-300">{error}</p> : null}
          <p className="text-xs text-slate-500">
            Saved profiles are stored in this browser and remain available after reloads.
          </p>
        </div>
      ) : null}
    </div>
  );
}
