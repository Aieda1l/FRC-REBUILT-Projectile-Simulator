import {simulateShot} from './trajectory2d.js';
import {evaluateShotUncertainty} from './uncertainty.js';

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const EPS = 1e-12;

const CLASSIFICATION_RANK = {
  'clean-entry': 3,
  'rim-collision': 2,
  'funnel-collision': 2,
  miss: 1,
};

function interactionOf(result) {
  return result.scoringInteraction ?? result.hubInteraction;
}

function interactionIsScore(interaction) {
  if (typeof interaction?.isScore === 'boolean') return interaction.isScore;
  return interaction?.classification === 'clean-entry';
}

function interactionRank(interaction) {
  if (Number.isFinite(interaction?.scoreRank)) return Number(interaction.scoreRank);
  if (interactionIsScore(interaction)) return 3;
  if (interaction?.status === 'collision') return 2;
  return CLASSIFICATION_RANK[interaction?.classification] ?? 0;
}

function robustScoreProbability(robust) {
  return robust?.scoreProbability ?? robust?.probabilities?.['clean-entry'] ?? 0;
}

function referenceDistance(candidate, reference) {
  const referenceAzimuth = reference.azimuth ?? 0;
  return Math.hypot(
    (candidate.velocity ?? reference.velocity) - reference.velocity,
    (candidate.angle ?? reference.angle) - reference.angle,
    (candidate.azimuth ?? referenceAzimuth) - referenceAzimuth,
  );
}

function lateralCompensation(params, velocity, angle) {
  const robotLateral = Number(params.robotVelocity?.[1] ?? 0);
  const referenceAzimuth = Number(params.azimuthDeg ?? 0);
  if (!Number.isFinite(robotLateral) || !Number.isFinite(referenceAzimuth)) {
    throw new RangeError('robot lateral velocity and azimuth must be finite');
  }
  if (Math.abs(robotLateral) <= EPS) {
    return {azimuth: referenceAzimuth, feasible: true};
  }

  const horizontalSpeed = velocity * Math.cos(angle * DEG_TO_RAD);
  if (horizontalSpeed <= EPS) {
    return {
      azimuth: -Math.sign(robotLateral) * 90,
      feasible: false,
    };
  }

  const ratio = -robotLateral / horizontalSpeed;
  const feasible = Math.abs(ratio) <= 1 + EPS;
  const clampedRatio = Math.max(-1, Math.min(1, ratio));
  return {
    azimuth: Math.asin(clampedRatio) * RAD_TO_DEG,
    feasible,
  };
}

export function rankCandidate(a, b, reference) {
  const ia = interactionOf(a.result);
  const ib = interactionOf(b.result);
  const rankA = interactionRank(ia);
  const rankB = interactionRank(ib);

  if (rankA !== rankB) return rankB - rankA;

  if (interactionIsScore(ia)) {
    const diff = (ib.clearanceMargin ?? -Infinity) - (ia.clearanceMargin ?? -Infinity);
    if (diff !== 0) return diff;
  } else if (ia.status === 'collision') {
    const diff = (ib.clearanceMargin ?? -Infinity) - (ia.clearanceMargin ?? -Infinity);
    if (diff !== 0) return diff;
  } else {
    const missA = ia.missDistance ?? Infinity;
    const missB = ib.missDistance ?? Infinity;
    if (missA !== missB) return missA - missB;
  }

  return referenceDistance(a, reference) - referenceDistance(b, reference);
}

export function rankRobustCandidate(a, b, reference) {
  const probabilityA = robustScoreProbability(a.robust);
  const probabilityB = robustScoreProbability(b.robust);
  if (probabilityA !== probabilityB) return probabilityB - probabilityA;

  const clearanceA = a.robust?.clearance?.p10 ?? -Infinity;
  const clearanceB = b.robust?.clearance?.p10 ?? -Infinity;
  if (clearanceA !== clearanceB) return clearanceB - clearanceA;

  return rankCandidate(a, b, reference);
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
  const reference = {
    velocity: params.velocity,
    angle: params.angleDeg,
    azimuth: params.azimuthDeg ?? 0,
  };
  const cache = new Map();
  let evaluatedCandidates = 0;
  let bestCandidate = null;

  const evaluate = (velocity, angle, dt) => {
    const key = `${velocity.toFixed(8)}|${angle.toFixed(8)}|${dt.toFixed(8)}`;
    if (cache.has(key)) return cache.get(key);

    const compensation = lateralCompensation(params, velocity, angle);
    const result = simulateShot(
      {
        ...params,
        velocity,
        angleDeg: angle,
        azimuthDeg: compensation.azimuth,
      },
      {dt},
    );
    const candidate = {
      velocity,
      angle,
      azimuth: compensation.azimuth,
      lateralCompensationFeasible: compensation.feasible,
      result,
    };
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
        classification: interactionOf(bestCandidate.result).classification,
        clearanceMargin: interactionOf(bestCandidate.result).clearanceMargin,
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
        classification: interactionOf(bestCandidate?.result ?? {})?.classification ?? 'miss',
        clearanceMargin: interactionOf(bestCandidate?.result ?? {})?.clearanceMargin ?? -Infinity,
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
    (candidate) => !interactionIsScore(interactionOf(candidate.result)),
  ) ?? null;
  const anyLateralCompensationFeasible = ranked.some(
    (candidate) => candidate.lateralCompensationFeasible !== false,
  );

  for (const candidate of ranked) {
    if (!interactionIsScore(interactionOf(candidate.result))) continue;
    const validated = evaluator.evaluate(candidate.velocity, candidate.angle, 0.001);
    if (interactionIsScore(interactionOf(validated.result))) {
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
    reason: anyLateralCompensationFeasible ? null : 'lateral-compensation-infeasible',
  };
}

function searchAngle(params, callbacks = {}) {
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
  return {candidates: [...coarse, ...refined], evaluator};
}

export function optimizeAngle(params, callbacks = {}) {
  const search = searchAngle(params, callbacks);
  return finalize(search.candidates, search.evaluator);
}

function searchVelocity(params, callbacks = {}) {
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
  return {candidates: [...coarse, ...refined], evaluator};
}

export function optimizeVelocity(params, callbacks = {}) {
  const search = searchVelocity(params, callbacks);
  return finalize(search.candidates, search.evaluator);
}

function searchBoth(params, callbacks = {}) {
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
  return {candidates: [...coarse, ...refined], evaluator};
}

export function optimizeBoth(params, callbacks = {}) {
  const search = searchBoth(params, callbacks);
  return finalize(search.candidates, search.evaluator);
}

export function optimizeRobust(params, {
  mode = 'both',
  uncertainty = {},
  coarseSamples = 64,
  finalSamples = 512,
  seed = 2026,
} = {}, callbacks = {}) {
  const searchers = {angle: searchAngle, velocity: searchVelocity, both: searchBoth};
  const searcher = searchers[mode];
  if (!searcher) throw new RangeError(`unknown robust optimization mode: ${mode}`);

  const search = searcher(params, callbacks);
  const ranked = sortCandidates(uniqueCandidates(search.candidates), search.evaluator.reference);
  const clean = ranked.filter((candidate) => (
    interactionIsScore(interactionOf(candidate.result))
  ));
  const shortlist = (clean.length ? clean : ranked).slice(0, 10);
  const robustCandidates = shortlist.map((candidate, index) => {
    const robust = evaluateShotUncertainty(
      {
        ...params,
        velocity: candidate.velocity,
        angleDeg: candidate.angle,
        azimuthDeg: candidate.azimuth,
      },
      uncertainty,
      {sampleCount: coarseSamples, seed, dt: 0.002},
    );
    const evaluated = {...candidate, robust};
    callbacks.onProgress?.({
      stage: 'robust',
      evaluatedCandidates: index + 1,
      totalCandidates: shortlist.length,
      bestCandidate: evaluated,
      cleanEntryProbability: robustScoreProbability(robust),
      scoreProbability: robustScoreProbability(robust),
      clearanceP10: robust.clearance.p10,
    });
    return evaluated;
  });

  robustCandidates.sort((a, b) => rankRobustCandidate(a, b, search.evaluator.reference));
  const best = robustCandidates[0] ?? null;
  const bestNearMiss = ranked.find(
    (candidate) => !interactionIsScore(interactionOf(candidate.result)),
  ) ?? null;

  if (!best || !interactionIsScore(interactionOf(best.result))) {
    return {
      solution: null,
      bestNearMiss: bestNearMiss ?? search.evaluator.bestCandidate,
      evaluatedCandidates: search.evaluator.evaluatedCandidates,
      monteCarloEvaluations: robustCandidates.length,
    };
  }

  const finalParams = {
    ...params,
    velocity: best.velocity,
    angleDeg: best.angle,
    azimuthDeg: best.azimuth,
  };
  const result = simulateShot(finalParams, {dt: 0.001});
  if (!interactionIsScore(interactionOf(result))) {
    return {
      solution: null,
      bestNearMiss: {...best, result},
      evaluatedCandidates: search.evaluator.evaluatedCandidates,
      monteCarloEvaluations: robustCandidates.length,
    };
  }
  const robust = evaluateShotUncertainty(finalParams, uncertainty, {
    sampleCount: finalSamples,
    seed,
    dt: 0.001,
  });
  return {
    solution: {
      velocity: best.velocity,
      angle: best.angle,
      azimuth: best.azimuth,
      result,
      robust,
    },
    bestNearMiss,
    evaluatedCandidates: search.evaluator.evaluatedCandidates,
    monteCarloEvaluations: robustCandidates.length + 1,
  };
}
