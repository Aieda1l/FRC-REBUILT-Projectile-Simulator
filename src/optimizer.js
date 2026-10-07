import {simulateShot} from './trajectory2d.js';

const CLASSIFICATION_RANK = {
  'clean-entry': 3,
  'rim-collision': 2,
  'funnel-collision': 2,
  miss: 1,
};

function referenceDistance(candidate, reference) {
  return Math.hypot(
    (candidate.velocity ?? reference.velocity) - reference.velocity,
    (candidate.angle ?? reference.angle) - reference.angle,
  );
}

export function rankCandidate(a, b, reference) {
  const ia = a.result.hubInteraction;
  const ib = b.result.hubInteraction;
  const rankA = CLASSIFICATION_RANK[ia.classification] ?? 0;
  const rankB = CLASSIFICATION_RANK[ib.classification] ?? 0;

  if (rankA !== rankB) return rankB - rankA;

  if (ia.classification === 'clean-entry') {
    const diff = (ib.clearanceMargin ?? -Infinity) - (ia.clearanceMargin ?? -Infinity);
    if (diff !== 0) return diff;
  } else if (
    ia.classification === 'rim-collision'
    || ia.classification === 'funnel-collision'
  ) {
    const diff = (ib.clearanceMargin ?? -Infinity) - (ia.clearanceMargin ?? -Infinity);
    if (diff !== 0) return diff;
  } else {
    const missA = ia.missDistance ?? Infinity;
    const missB = ib.missDistance ?? Infinity;
    if (missA !== missB) return missA - missB;
  }

  return referenceDistance(a, reference) - referenceDistance(b, reference);
}

function numericRange(start, end, step) {
  const values = [];
  const count = Math.floor((end - start) / step + 1e-9);
  for (let index = 0; index <= count; index += 1) {
    values.push(Number((start + index * step).toFixed(10)));
  }
  if (values.at(-1) < end - 1e-9) values.push(end);
  return values;
}

function createEvaluator(params, callbacks, totalCandidates) {
  const reference = {velocity: params.velocity, angle: params.angleDeg};
  const cache = new Map();
  let evaluatedCandidates = 0;
  let bestCandidate = null;

  const evaluate = (velocity, angle, dt) => {
    const key = `${velocity.toFixed(8)}|${angle.toFixed(8)}|${dt.toFixed(8)}`;
    if (cache.has(key)) return cache.get(key);

    const result = simulateShot(
      {...params, velocity, angleDeg: angle},
      {dt},
    );
    const candidate = {velocity, angle, result};
    cache.set(key, candidate);
    evaluatedCandidates += 1;

    if (!bestCandidate || rankCandidate(candidate, bestCandidate, reference) < 0) {
      bestCandidate = candidate;
    }

    if (callbacks.onProgress && evaluatedCandidates % 25 === 0) {
      callbacks.onProgress({
        evaluatedCandidates,
        totalCandidates,
        bestCandidate,
        classification: bestCandidate.result.hubInteraction.classification,
        clearanceMargin: bestCandidate.result.hubInteraction.clearanceMargin,
      });
    }
    return candidate;
  };

  const emitFinalProgress = () => {
    if (callbacks.onProgress) {
      callbacks.onProgress({
        evaluatedCandidates,
        totalCandidates,
        bestCandidate,
        classification: bestCandidate?.result.hubInteraction.classification ?? 'miss',
        clearanceMargin: bestCandidate?.result.hubInteraction.clearanceMargin ?? -Infinity,
      });
    }
  };

  return {
    evaluate,
    emitFinalProgress,
    get evaluatedCandidates() { return evaluatedCandidates; },
    get bestCandidate() { return bestCandidate; },
    reference,
  };
}

function sortCandidates(candidates, reference) {
  return [...candidates].sort((a, b) => rankCandidate(a, b, reference));
}

function uniqueCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.velocity.toFixed(8)}|${candidate.angle.toFixed(8)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function finalize(candidates, evaluator) {
  const ranked = sortCandidates(uniqueCandidates(candidates), evaluator.reference);
  const bestNearMiss = ranked.find(
    (candidate) => candidate.result.hubInteraction.classification !== 'clean-entry',
  ) ?? null;

  for (const candidate of ranked) {
    if (candidate.result.hubInteraction.classification !== 'clean-entry') continue;
    const validated = evaluator.evaluate(candidate.velocity, candidate.angle, 0.001);
    if (validated.result.hubInteraction.classification === 'clean-entry') {
      evaluator.emitFinalProgress();
      return {
        solution: validated,
        bestNearMiss,
        evaluatedCandidates: evaluator.evaluatedCandidates,
      };
    }
  }

  evaluator.emitFinalProgress();
  return {
    solution: null,
    bestNearMiss: bestNearMiss ?? evaluator.bestCandidate,
    evaluatedCandidates: evaluator.evaluatedCandidates,
  };
}

export function optimizeAngle(params, callbacks = {}) {
  const evaluator = createEvaluator(params, callbacks, 160);
  const coarse = numericRange(5, 85, 2).map((angle) => (
    evaluator.evaluate(params.velocity, angle, 0.005)
  ));
  const seeds = sortCandidates(coarse, evaluator.reference).slice(0, 6);
  const refined = [];

  for (const seed of seeds) {
    for (const angle of numericRange(
      Math.max(5, seed.angle - 2),
      Math.min(85, seed.angle + 2),
      0.25,
    )) {
      refined.push(evaluator.evaluate(params.velocity, angle, 0.002));
    }
  }

  return finalize([...coarse, ...refined], evaluator);
}

export function optimizeVelocity(params, callbacks = {}) {
  const evaluator = createEvaluator(params, callbacks, 100);
  const coarse = numericRange(5, 25, 1).map((velocity) => (
    evaluator.evaluate(velocity, params.angleDeg, 0.005)
  ));
  const seeds = sortCandidates(coarse, evaluator.reference).slice(0, 6);
  const refined = [];

  for (const seed of seeds) {
    for (const velocity of numericRange(
      Math.max(5, seed.velocity - 1),
      Math.min(25, seed.velocity + 1),
      0.2,
    )) {
      refined.push(evaluator.evaluate(velocity, params.angleDeg, 0.002));
    }
  }

  return finalize([...coarse, ...refined], evaluator);
}

export function optimizeBoth(params, callbacks = {}) {
  const evaluator = createEvaluator(params, callbacks, 980);
  const coarse = [];

  for (const velocity of numericRange(5, 25, 1)) {
    for (const angle of numericRange(20, 80, 2)) {
      coarse.push(evaluator.evaluate(velocity, angle, 0.005));
    }
  }

  const seeds = sortCandidates(coarse, evaluator.reference).slice(0, 12);
  const refined = [];
  const velocityOffsets = [-0.4, -0.2, 0, 0.2, 0.4];
  const angleOffsets = [-0.5, -0.25, 0, 0.25, 0.5];

  for (const seed of seeds) {
    for (const dv of velocityOffsets) {
      for (const da of angleOffsets) {
        const velocity = Math.max(5, Math.min(25, seed.velocity + dv));
        const angle = Math.max(20, Math.min(80, seed.angle + da));
        refined.push(evaluator.evaluate(velocity, angle, 0.002));
      }
    }
  }

  return finalize([...coarse, ...refined], evaluator);
}
