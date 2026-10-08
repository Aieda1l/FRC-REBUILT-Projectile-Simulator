// Profiles describe shooter projectiles and simple scoring apertures in SI units.
// Nominal historical dimensions are starting points, not calibrated flight models.
export const GAME_PIECES = [
  {id: 'fuel-2026', name: '2026 REBUILT · FUEL', shape: 'sphere', mass: 0.215, diameter: 0.15, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'cargo-2022', name: '2022 RAPID REACT · CARGO', shape: 'sphere', mass: 0.270, diameter: 0.2413, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'power-cell-2020', name: '2020 INFINITE RECHARGE · POWER CELL', shape: 'sphere', mass: 0.14, diameter: 0.1778, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'fuel-2017', name: '2017 STEAMWORKS · FUEL', shape: 'sphere', mass: 0.074, diameter: 0.127, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'ball-2012', name: '2012 REBOUND RUMBLE · basketball', shape: 'sphere', mass: 0.20, diameter: 0.2032, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'disc-2013', name: '2013 ULTIMATE ASCENT · disc (approx.)', shape: 'disc', mass: 0.18, diameter: 0.23495, dragCoeff: 0.47, liftCoeff: 0.25},
  {id: 'note-2024', name: '2024 CRESCENDO · NOTE (approx.)', shape: 'ring', mass: 0.235, diameter: 0.3556, dragCoeff: 0.47, liftCoeff: 0.25},
];

export const SCORING_TARGETS = [
  {id: 'hub-2026', name: '2026 REBUILT · HUB', kind: 'hub', x: 0, lateralY: 0, z: 1.8288, topAcrossFlats: 41.727 * 0.0254, bottomSide: 18.92 * 0.0254, panelHeight: 17.90 * 0.0254},
  {id: 'upper-hub-2022', name: '2022 RAPID REACT · upper hub (opening)', kind: 'hoop', x: 0, lateralY: 0, z: 2.6416, diameter: 1.2192},
  {id: 'lower-hub-2022', name: '2022 RAPID REACT · lower hub (opening)', kind: 'hoop', x: 0, lateralY: 0, z: 1.0414, diameter: 1.5272},
  {id: 'upper-boiler-2017', name: '2017 STEAMWORKS · upper boiler (opening)', kind: 'hoop', x: 0, lateralY: 0, z: 2.4638, diameter: 0.5461},
  {id: 'low-boiler-2017', name: '2017 STEAMWORKS · low boiler', kind: 'slot', x: 0, lateralY: 0, z: 0.5683, width: 0.635, height: 0.22225},
  {id: 'bottom-port-2020', name: '2020 INFINITE RECHARGE · bottom port', kind: 'slot', x: 0, lateralY: 0, z: 0.5842, width: 0.8636, height: 0.254},
  {id: 'inner-port-2020', name: '2020 INFINITE RECHARGE · inner port', kind: 'round-slot', x: 0, lateralY: 0, z: 2.49555, diameter: 0.3302},
  {id: 'high-goal-2013', name: '2013 ULTIMATE ASCENT · high disc slot', kind: 'slot', x: 0, lateralY: 0, z: 2.797, width: 1.3716, height: 0.3048},
  {id: 'hoop-2012', name: '2012 REBOUND RUMBLE · basket (approx.)', kind: 'hoop', x: 0, lateralY: 0, z: 2.4, diameter: 0.4572},
  {id: 'speaker-2024', name: '2024 CRESCENDO · SPEAKER (approx.)', kind: 'slot', x: 0, lateralY: 0, z: 2.0, width: 1.05, height: 0.45},
  {id: 'amp-2024', name: '2024 CRESCENDO · AMP (approx.)', kind: 'slot', x: 0, lateralY: 0, z: 1.2, width: 0.52, height: 0.26},
];

export const DEFAULT_PIECE_ID = 'fuel-2026';
export const DEFAULT_TARGET_ID = 'hub-2026';
export const STORAGE_KEY = 'frc-projectile-game-library-v1';
export const TARGET_KINDS = ['hub', 'hoop', 'slot', 'round-slot'];
export const PIECE_SHAPES = ['sphere', 'disc', 'ring'];

const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
const positive = (number, label, max = 100) => {
  if (typeof number !== 'number' || !Number.isFinite(number) || number <= 0 || number > max) {
    throw new RangeError(label + ' must be greater than 0 and at most ' + max);
  }
  return number;
};
const nonnegative = (number, label) => {
  if (typeof number !== 'number' || !Number.isFinite(number) || number < 0 || number > 10) {
    throw new RangeError(label + ' must be between 0 and 10');
  }
  return number;
};
const textField = (value) => {
  if (typeof value !== 'string' || !value.trim() || value.trim().length > 80) {
    throw new RangeError('Name must contain 1–80 characters');
  }
  return value.trim();
};
const position = (number, name) => {
  if (typeof number !== 'number' || !Number.isFinite(number) || Math.abs(number) > 50) {
    throw new RangeError(name + ' must be between -50 and 50 meters');
  }
  return number;
};

export function validatePiece(piece) {
  if (!piece || typeof piece !== 'object') throw new TypeError('Invalid game piece');
  if (!PIECE_SHAPES.includes(piece.shape)) throw new RangeError('Unsupported game piece shape');
  return {
    id: String(piece.id ?? ''),
    name: textField(piece.name),
    shape: piece.shape,
    mass: positive(piece.mass, 'Mass', 100),
    diameter: positive(piece.diameter, 'Diameter', 10),
    dragCoeff: nonnegative(piece.dragCoeff, 'Drag coefficient'),
    liftCoeff: nonnegative(piece.liftCoeff, 'Lift coefficient'),
  };
}

export function validateTarget(target) {
  if (!target || typeof target !== 'object' || !TARGET_KINDS.includes(target.kind)) {
    throw new RangeError('Unsupported scoring target type');
  }
  const common = {
    id: String(target.id ?? ''),
    name: textField(target.name),
    kind: target.kind,
    x: position(target.x, 'X'),
    lateralY: position(target.lateralY, 'Lateral Y'),
    z: positive(target.z, 'Height', 50),
  };
  if (target.kind === 'hub') {
    const topAcrossFlats = positive(target.topAcrossFlats, 'Top across flats', 20);
    const bottomSide = positive(target.bottomSide, 'Bottom hex side', 20);
    const panelHeight = positive(target.panelHeight, 'Panel height', 20);
    if (panelHeight >= target.z) throw new RangeError('Hub bottom must be above the ground');
    return {...common, topAcrossFlats, bottomSide, panelHeight};
  }
  if (target.kind === 'hoop' || target.kind === 'round-slot') {
    return {...common, diameter: positive(target.diameter, 'Opening diameter', 20)};
  }
  return {
    ...common,
    width: positive(target.width, 'Opening width', 20),
    height: positive(target.height, 'Opening height', 20),
  };
}

function safeItems(items, validation) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const result = [];
  for (const item of items.slice(0, 200)) {
    try {
      if (typeof item?.id !== 'string' || !/^custom-[\w-]{1,90}$/.test(item.id) || seen.has(item.id)) continue;
      const validated = validation(item);
      seen.add(validated.id);
      result.push(validated);
    } catch {
      // Corrupt or old saved entries are ignored, never passed into simulation.
    }
  }
  return result;
}

export function emptyLibrary() {
  return {pieces: [], targets: [], pieceId: DEFAULT_PIECE_ID, targetId: DEFAULT_TARGET_ID};
}

export function parseLibrary(raw) {
  try {
    const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (!data || data.version !== 1) return emptyLibrary();
    const pieces = safeItems(data.pieces, validatePiece);
    const targets = safeItems(data.targets, validateTarget);
    const pieceIds = new Set([...GAME_PIECES, ...pieces].map((item) => item.id));
    const targetIds = new Set([...SCORING_TARGETS, ...targets].map((item) => item.id));
    return {
      pieces, targets,
      pieceId: pieceIds.has(data.pieceId) ? data.pieceId : DEFAULT_PIECE_ID,
      targetId: targetIds.has(data.targetId) ? data.targetId : DEFAULT_TARGET_ID,
    };
  } catch {
    return emptyLibrary();
  }
}

export function loadLibrary(storage = typeof window !== 'undefined' ? window.localStorage : null) {
  try {
    return parseLibrary(storage?.getItem(STORAGE_KEY));
  } catch {
    return emptyLibrary();
  }
}

export function saveLibrary(library, storage = typeof window !== 'undefined' ? window.localStorage : null) {
  if (!storage) throw new Error('Browser storage is unavailable');
  const normalized = parseLibrary({...library, version: 1});
  storage.setItem(STORAGE_KEY, JSON.stringify({...normalized, version: 1}));
  return normalized;
}

export function resolveLibrarySelection(library) {
  const pieces = [...GAME_PIECES, ...library.pieces];
  const targets = [...SCORING_TARGETS, ...library.targets];
  return {
    piece: pieces.find((entry) => entry.id === library.pieceId) ?? GAME_PIECES[0],
    target: targets.find((entry) => entry.id === library.targetId) ?? SCORING_TARGETS[0],
  };
}

export function newCustomId() {
  return 'custom-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
}

export function upsertLibraryItem(library, category, item, selectedId) {
  if (!hasOwn(library, category) || !['pieces', 'targets'].includes(category)) {
    throw new RangeError('Unknown library category');
  }
  const validate = category === 'pieces' ? validatePiece : validateTarget;
  const validated = validate(item);
  if (!validated.id.startsWith('custom-')) throw new RangeError('Built-in presets cannot be overwritten');
  const items = library[category].filter((entry) => entry.id !== validated.id);
  if (items.length >= 200) throw new RangeError('Limit of 200 custom entries reached');
  const selectedKey = category === 'pieces' ? 'pieceId' : 'targetId';
  return {...library, [category]: [...items, validated], [selectedKey]: selectedId ?? validated.id};
}
