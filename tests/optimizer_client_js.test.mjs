import test from 'node:test';
import assert from 'node:assert/strict';
import {createOptimizerClient} from '../src/optimizerClient.js';

class FakeWorker {
  constructor() {
    this.posted = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }
  postMessage(message) { this.posted.push(message); }
  terminate() { this.terminated = true; }
}

function factoryCollector(workers) {
  return () => {
    const worker = new FakeWorker();
    workers.push(worker);
    return worker;
  };
}

test('start posts one request and cancel terminates active worker', () => {
  const workers = [];
  const client = createOptimizerClient({workerFactory: factoryCollector(workers)});
  const requestId = client.start('angle', {velocity: 10});
  assert.deepEqual(workers[0].posted[0], {
    type: 'optimize', requestId, mode: 'angle', params: {velocity: 10},
  });
  client.cancel();
  assert.equal(workers[0].terminated, true);
});

test('starting a new optimization terminates the previous worker', () => {
  const workers = [];
  const client = createOptimizerClient({workerFactory: factoryCollector(workers)});
  client.start('angle', {});
  client.start('velocity', {});
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
});

test('stale completion from an old request is ignored', () => {
  const workers = [];
  const completed = [];
  const client = createOptimizerClient({
    workerFactory: factoryCollector(workers),
    onComplete: (result) => completed.push(result),
  });
  const oldId = client.start('angle', {});
  const oldHandler = workers[0].onmessage;
  const newId = client.start('velocity', {});
  oldHandler({data: {type: 'complete', requestId: oldId, result: {solution: 'old'}}});
  assert.equal(completed.length, 0);
  workers[1].onmessage({data: {type: 'complete', requestId: newId, result: {solution: 'new'}}});
  assert.deepEqual(completed, [{solution: 'new'}]);
});

test('completion terminates active worker', () => {
  const workers = [];
  const client = createOptimizerClient({workerFactory: factoryCollector(workers)});
  const requestId = client.start('angle', {});
  workers[0].onmessage({data: {type: 'complete', requestId, result: {solution: null}}});
  assert.equal(workers[0].terminated, true);
});

test('dispose terminates active worker', () => {
  const workers = [];
  const client = createOptimizerClient({workerFactory: factoryCollector(workers)});
  client.start('both', {});
  client.dispose();
  assert.equal(workers[0].terminated, true);
});
