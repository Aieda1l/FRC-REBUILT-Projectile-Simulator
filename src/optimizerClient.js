function defaultWorkerFactory() {
  return new Worker(new URL('./optimizer.worker.js', import.meta.url), {type: 'module'});
}

export function createOptimizerClient({
  workerFactory = defaultWorkerFactory,
  onProgress = () => {},
  onComplete = () => {},
  onError = () => {},
} = {}) {
  let nextRequestId = 1;
  let active = null;

  const terminateActive = () => {
    if (active?.worker) active.worker.terminate();
    active = null;
  };

  const start = (mode, params, options) => {
    terminateActive();
    const requestId = nextRequestId++;
    const worker = workerFactory();
    active = {requestId, worker};

    worker.onmessage = (event) => {
      const message = event.data ?? {};
      if (!active || message.requestId !== active.requestId) return;

      if (message.type === 'progress') {
        onProgress(message.progress);
      } else if (message.type === 'complete') {
        const result = message.result;
        terminateActive();
        onComplete(result);
      } else if (message.type === 'error') {
        const messageText = message.message ?? 'Optimization failed';
        terminateActive();
        onError(messageText);
      }
    };

    worker.onerror = (event) => {
      if (!active || active.requestId !== requestId) return;
      const messageText = event?.message ?? 'Optimization worker failed';
      terminateActive();
      onError(messageText);
    };

    const message = {type: 'optimize', requestId, mode, params};
    if (options !== undefined) message.options = options;
    worker.postMessage(message);
    return requestId;
  };

  return {
    start,
    cancel: terminateActive,
    dispose: terminateActive,
  };
}
