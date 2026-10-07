const EPS = 1e-12;
export const DEFAULT_DYNAMIC_VISCOSITY = 1.81e-5;

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return Number(value);
}

function nonNegative(value, name) {
  const out = finite(value, name);
  if (out < 0) throw new RangeError(`${name} must be non-negative`);
  return out;
}

function positive(value, name) {
  const out = finite(value, name);
  if (out <= 0) throw new RangeError(`${name} must be positive`);
  return out;
}

function axis(values, name) {
  if (!Array.isArray(values) || values.length < 2) {
    throw new RangeError(`${name} must contain at least two values`);
  }
  const out = values.map((value, index) => nonNegative(value, `${name}[${index}]`));
  for (let i = 1; i < out.length; i += 1) {
    if (out[i] <= out[i - 1]) throw new RangeError(`${name} must be strictly increasing`);
  }
  return out;
}

function coefficients(values, expectedLength, name = 'coefficients') {
  if (!Array.isArray(values) || values.length !== expectedLength) {
    throw new RangeError(`${name} must contain exactly ${expectedLength} values`);
  }
  return values.map((value, index) => nonNegative(value, `${name}[${index}]`));
}

export function reynoldsNumber({
  airDensity,
  speed,
  diameter,
  dynamicViscosity = DEFAULT_DYNAMIC_VISCOSITY,
}) {
  const rho = nonNegative(airDensity, 'airDensity');
  const v = nonNegative(speed, 'speed');
  const d = positive(diameter, 'diameter');
  const mu = positive(dynamicViscosity, 'dynamicViscosity');
  return rho * v * d / mu;
}

export function spinParameter({radius, perpendicularSpin, speed}) {
  const r = positive(radius, 'radius');
  const omega = nonNegative(Math.abs(finite(perpendicularSpin, 'perpendicularSpin')), 'perpendicularSpin');
  const v = nonNegative(speed, 'speed');
  if (v <= EPS) return 0;
  return r * omega / v;
}

export function normalizeDragModel(model, fallbackCoefficient) {
  const fallback = nonNegative(fallbackCoefficient, 'fallbackCoefficient');
  const input = model ?? {kind: 'constant', coefficient: fallback};
  if (input.kind === 'constant') {
    return {kind: 'constant', coefficient: nonNegative(input.coefficient, 'coefficient')};
  }
  if (input.kind === 'table1d') {
    const reynolds = axis(input.reynolds, 'reynolds');
    return {
      kind: 'table1d',
      reynolds,
      coefficients: coefficients(input.coefficients, reynolds.length),
    };
  }
  throw new RangeError(`unknown drag model kind: ${input.kind}`);
}

export function normalizeLiftModel(model, fallbackCoefficient) {
  const fallback = nonNegative(fallbackCoefficient, 'fallbackCoefficient');
  const input = model ?? {kind: 'legacy-spin-cap', maxCoefficient: fallback, saturationSpin: 0.5};
  if (input.kind === 'legacy-spin-cap') {
    return {
      kind: 'legacy-spin-cap',
      maxCoefficient: nonNegative(input.maxCoefficient, 'maxCoefficient'),
      saturationSpin: positive(input.saturationSpin ?? 0.5, 'saturationSpin'),
    };
  }
  if (input.kind === 'table1d') {
    const spinParameters = axis(input.spinParameters, 'spinParameters');
    return {
      kind: 'table1d',
      spinParameters,
      coefficients: coefficients(input.coefficients, spinParameters.length),
    };
  }
  if (input.kind === 'table2d') {
    const reynolds = axis(input.reynolds, 'reynolds');
    const spinParameters = axis(input.spinParameters, 'spinParameters');
    if (!Array.isArray(input.coefficients) || input.coefficients.length !== reynolds.length) {
      throw new RangeError(`coefficients must contain exactly ${reynolds.length} rows`);
    }
    return {
      kind: 'table2d',
      reynolds,
      spinParameters,
      coefficients: input.coefficients.map((row, index) => (
        coefficients(row, spinParameters.length, `coefficients[${index}]`)
      )),
    };
  }
  throw new RangeError(`unknown lift model kind: ${input.kind}`);
}

function bracket(values, query) {
  const q = nonNegative(query, 'query');
  if (q < values[0]) return {low: 0, high: 0, fraction: 0, clamped: true};
  const last = values.length - 1;
  if (q > values[last]) return {low: last, high: last, fraction: 0, clamped: true};
  if (q === values[0]) return {low: 0, high: 0, fraction: 0, clamped: false};
  if (q === values[last]) return {low: last, high: last, fraction: 0, clamped: false};
  for (let i = 0; i < last; i += 1) {
    if (q >= values[i] && q <= values[i + 1]) {
      return {
        low: i,
        high: i + 1,
        fraction: (q - values[i]) / (values[i + 1] - values[i]),
        clamped: false,
      };
    }
  }
  throw new RangeError('query could not be bracketed');
}

function linear(values, outputs, query) {
  const b = bracket(values, query);
  if (b.low === b.high) return {value: outputs[b.low], clamped: b.clamped};
  return {
    value: outputs[b.low] + b.fraction * (outputs[b.high] - outputs[b.low]),
    clamped: b.clamped,
  };
}

export function evaluateDragModel(model, reynolds) {
  if (model.kind === 'constant') return {coefficient: model.coefficient, clamped: false};
  const out = linear(model.reynolds, model.coefficients, reynolds);
  return {coefficient: out.value, clamped: out.clamped};
}

export function evaluateLiftModel(model, reynolds, spinParameterValue) {
  const s = nonNegative(spinParameterValue, 'spinParameter');
  if (model.kind === 'legacy-spin-cap') {
    return {
      coefficient: model.maxCoefficient * Math.min(s / model.saturationSpin, 1),
      clamped: false,
    };
  }
  if (model.kind === 'table1d') {
    const out = linear(model.spinParameters, model.coefficients, s);
    return {coefficient: out.value, clamped: out.clamped};
  }

  const rb = bracket(model.reynolds, reynolds);
  const sb = bracket(model.spinParameters, s);
  const rowValue = (row) => {
    if (sb.low === sb.high) return model.coefficients[row][sb.low];
    const a = model.coefficients[row][sb.low];
    const b = model.coefficients[row][sb.high];
    return a + sb.fraction * (b - a);
  };
  const low = rowValue(rb.low);
  const value = rb.low === rb.high
    ? low
    : low + rb.fraction * (rowValue(rb.high) - low);
  return {coefficient: value, clamped: rb.clamped || sb.clamped};
}
