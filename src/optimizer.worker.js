import {optimizeAngle, optimizeBoth, optimizeRobust, optimizeVelocity} from './optimizer.js';

const optimizers = {
  angle: optimizeAngle,
  velocity: optimizeVelocity,
  both: optimizeBoth,
  robust: optimizeRobust,
};

self.onmessage = (event) => {
  const {type, requestId, mode, params, options} = event.data ?? {};
  if (type !== 'optimize') return;

  const optimize = optimizers[mode];
  if (!optimize) {
    self.postMessage({type: 'error', requestId, message: `Unknown optimization mode: ${mode}`});
    return;
  }

  try {
    const progressCallbacks = {
      onProgress: (progress) => self.postMessage({type: 'progress', requestId, progress}),
    };
    const result = mode === 'robust'
      ? optimizeRobust(params, options ?? {}, progressCallbacks)
      : optimize(params, progressCallbacks);
    /*
      legacy: true,
    });*/
    self.postMessage({type: 'complete', requestId, result});
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
