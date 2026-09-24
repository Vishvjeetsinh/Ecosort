import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  makeApp,
  makeRequest,
  resetDatabase,
  closeDatabase,
  seedClassification,
  assertIsoTimestamp,
} from './helpers.js';

describe('GET /api/health', () => {
  let app;
  let request;

  before(async () => {
    app = await makeApp();
    request = await makeRequest();
  });

  beforeEach(async () => {
    await resetDatabase();
  });

  after(async () => {
    await closeDatabase();
  });

  test('answers 200 with the documented shape', async () => {
    const res = await request.get('/api/health');

    assert.equal(res.status, 200);
    assert.match(res.headers['content-type'] ?? '', /application\/json/);
    assert.equal(res.body.status, 'ok');

    assert.equal(typeof res.body.version, 'string', 'version must be a string');
    assert.ok(res.body.version.length > 0, 'version must not be empty');

    assert.equal(typeof res.body.uptimeSeconds, 'number');
    assert.ok(Number.isFinite(res.body.uptimeSeconds), 'uptimeSeconds must be finite');
    assert.ok(res.body.uptimeSeconds >= 0, 'uptimeSeconds must not be negative');

    const stamp = assertIsoTimestamp(res.body.timestamp, 'timestamp');
    const skew = Math.abs(Date.now() - stamp);
    assert.ok(skew < 60_000, `timestamp is ${skew}ms away from now`);
  });

  test('reports a reachable database', async () => {
    const res = await request.get('/api/health');

    assert.equal(res.status, 200);
    assert.ok(res.body.db && typeof res.body.db === 'object', 'db block is required');
    assert.equal(res.body.db.ok, true, 'db.ok must be true when the API is serving');
    assert.equal(typeof res.body.db.path, 'string');
    assert.ok(res.body.db.path.length > 0, 'db.path must not be empty');
    assert.ok(
      Number.isInteger(res.body.db.classifications),
      'db.classifications must be an integer',
    );
    assert.equal(res.body.db.classifications, 0, 'a freshly reset database holds no rows');
  });

  test('db.classifications tracks the stored rows', async () => {
    await seedClassification(app);
    await seedClassification(app);

    const res = await request.get('/api/health');

    assert.equal(res.status, 200);
    assert.equal(res.body.db.ok, true);
    assert.equal(res.body.db.classifications, 2);
  });

  test('uptimeSeconds does not go backwards between calls', async () => {
    const first = await request.get('/api/health');
    const second = await request.get('/api/health');

    assert.equal(first.status, 200);
    assert.equal(second.status, 200);
    assert.ok(
      second.body.uptimeSeconds >= first.body.uptimeSeconds,
      'uptime must be monotonic',
    );
  });
});
