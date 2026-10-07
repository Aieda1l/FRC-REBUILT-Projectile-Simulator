import {optimizeAngle, optimizeBoth, optimizeVelocity} from './optimizer.js';

const optimizers = {
  angle: optimizeAngle,
  velocity: optimizeVelocity,
  both: optimizeBoth,
};

self.onmessage = (event) => {
  const {type, requestId, mode, params} = event.data ?? {};
  if (type !== 'optimize') return;

  const optimize = optimizers[mode];
  if (!optimize) {
    self.postMessage({type: 'error', requestId, message: `Unknown optimization mode: ${mode}`});
    return;
  }

  try {
    const result = optimize(params, {
      onProgress: (progress) => self.postMessage({type: 'progress', requestId, progress}),
    });
    self.postMessage({type: 'complete', requestId, result});
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
