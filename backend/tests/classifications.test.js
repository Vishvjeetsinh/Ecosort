import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  makeApp,
  makeRequest,
  resetDatabase,
  closeDatabase,
  seedClassification,
  validBody,
  prediction,
  oversizedDataUrl,
  assertErrorEnvelope,
  assertHistoryItemShape,
  assertIsoTimestamp,
  TINY_PNG_DATA_URL,
  DEFAULT_REGION_ID,
} from './helpers.js';

describe('POST /api/classifications', () => {
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

  test('stores a valid classification and echoes it back', async () => {
    const body = validBody();
    const res = await request.post('/api/classifications').send(body);

    assert.equal(res.status, 201, `expected 201, got ${res.status}: ${JSON.stringify(res.body)}`);
    const item = res.body.item;
    assertHistoryItemShape(item);

    assert.deepEqual(item.predictions, body.predictions, 'predictions must round-trip verbatim');
    assert.deepEqual(item.rawLabels, body.rawLabels, 'rawLabels must round-trip verbatim');

    // The three "top" columns are derived, not accepted from the client.
    assert.equal(item.topCategory, body.predictions[0].category);
    assert.equal(item.topLabel, body.predictions[0].label);
    assert.equal(item.topConfidence, body.predictions[0].confidence);

    assert.equal(item.source, body.source);
    assert.equal(item.modelKind, body.modelKind);
    assert.equal(item.regionId, body.regionId);
    assert.equal(item.imageDataUrl, body.imageDataUrl);
    assert.equal(item.notes, body.notes);
    assert.equal(item.durationMs, body.durationMs);

    assert.equal(item.correctedCategory, null, 'a fresh row carries no correction');
    assert.equal(item.effectiveCategory, item.topCategory);

    const created = assertIsoTimestamp(item.createdAt, 'createdAt');
    assert.ok(Math.abs(Date.now() - created) < 60_000, 'createdAt must be roughly now');
  });

  test('accepts the minimal body and nulls the optional fields', async () => {
    const res = await request.post('/api/classifications').send({
      predictions: [prediction('paper', 0.5, 'Paper — newspaper')],
      source: 'webcam',
      modelKind: 'custom',
      regionId: DEFAULT_REGION_ID,
    });

    assert.equal(res.status, 201, JSON.stringify(res.body));
    const item = res.body.item;
    assertHistoryItemShape(item);
    assert.equal(item.rawLabels, null);
    assert.equal(item.imageDataUrl, null);
    assert.equal(item.notes, null);
    assert.equal(item.durationMs, null);
    assert.equal(item.effectiveCategory, 'paper');
  });

  test('rejects a body with no predictions', async () => {
    const { predictions, ...withoutPredictions } = validBody();
    assert.ok(predictions, 'the fixture really does carry predictions');

    const res = await request.post('/api/classifications').send(withoutPredictions);

    const error = assertErrorEnvelope(res, 400, 'bad_request');
    assert.notEqual(error.details, null, 'a validation failure must report details');
    assert.ok(
      JSON.stringify(error.details).includes('predictions'),
      'details must name the offending field',
    );
  });

  test('rejects an empty predictions array', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ predictions: [] }));
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('rejects more than ten predictions', async () => {
    const eleven = Array.from({ length: 11 }, () => prediction('trash', 0.1, 'General Waste'));
    const res = await request.post('/api/classifications').send(validBody({ predictions: eleven }));
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('rejects a confidence outside 0..1', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ predictions: [prediction('plastic', 1.5, 'Plastic — bottle')] }));
    assertErrorEnvelope(res, 400, 'bad_request');

    const negative = await request
      .post('/api/classifications')
      .send(validBody({ predictions: [prediction('plastic', -0.1, 'Plastic — bottle')] }));
    assertErrorEnvelope(negative, 400, 'bad_request');
  });

  test('rejects an unknown category', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ predictions: [prediction('unobtainium', 0.9, 'Mystery')] }));
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('rejects an unknown regionId', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ regionId: 'atlantis' }));
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('rejects an unknown source and modelKind', async () => {
    const badSource = await request
      .post('/api/classifications')
      .send(validBody({ source: 'telepathy' }));
    assertErrorEnvelope(badSource, 400, 'bad_request');

    const badKind = await request
      .post('/api/classifications')
      .send(validBody({ modelKind: 'quantum' }));
    assertErrorEnvelope(badKind, 400, 'bad_request');
  });

  test('rejects notes longer than 500 characters', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ notes: 'n'.repeat(501) }));
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('rejects an image data URL past the limit with 413', async () => {
    const res = await request
      .post('/api/classifications')
      .send(validBody({ imageDataUrl: oversizedDataUrl() }));

    assertErrorEnvelope(res, 413, 'payload_too_large');

    const list = await request.get('/api/classifications');
    assert.equal(list.body.total, 0, 'a rejected payload must not be stored');
  });
});

describe('GET /api/classifications', () => {
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

  test('is empty on a fresh database', async () => {
    const res = await request.get('/api/classifications');

    assert.equal(res.status, 200);
    assert.deepEqual(res.body.items, []);
    assert.equal(res.body.total, 0);
    assert.equal(res.body.limit, 25, 'the documented default limit is 25');
    assert.equal(res.body.offset, 0);
  });

  test('paginates with limit and offset while reporting the full total', async () => {
    await seedClassification(app);
    await seedClassification(app);
    await seedClassification(app);

    const firstPage = await request.get('/api/classifications?limit=2');
    assert.equal(firstPage.status, 200);
    assert.equal(firstPage.body.items.length, 2);
    assert.equal(firstPage.body.total, 3, 'total counts every matching row, not the page');
    assert.equal(firstPage.body.limit, 2);
    assert.equal(firstPage.body.offset, 0);

    const secondPage = await request.get('/api/classifications?limit=2&offset=2');
    assert.equal(secondPage.body.items.length, 1);
    assert.equal(secondPage.body.total, 3);
    assert.equal(secondPage.body.offset, 2);

    const firstIds = firstPage.body.items.map((i) => i.id);
    const secondIds = secondPage.body.items.map((i) => i.id);
    assert.equal(
      new Set([...firstIds, ...secondIds]).size,
      3,
      'pages must not overlap or drop rows',
    );
  });

  test('never honours a limit above the documented maximum of 100', async () => {
    await seedClassification(app);

    const capped = await request.get('/api/classifications?limit=1000');
    if (capped.status === 200) {
      assert.ok(capped.body.limit <= 100, `limit must be capped at 100, got ${capped.body.limit}`);
      assert.ok(capped.body.items.length <= 100);
    } else {
      assertErrorEnvelope(capped, 400, 'bad_request');
    }
  });

  // The contract fixes the defaults but not the reaction to garbage, so either a 400 or a
  // fall back to the documented defaults is acceptable - a 500 or a broken page is not.
  test('survives nonsense paging parameters', async () => {
    await seedClassification(app);

    for (const query of ['limit=abc', 'offset=-1', 'limit=0', 'offset=nope']) {
      const res = await request.get(`/api/classifications?${query}`);
      if (res.status === 400) {
        assertErrorEnvelope(res, 400, 'bad_request');
        continue;
      }
      assert.equal(res.status, 200, `"${query}" must answer 200 or 400, got ${res.status}`);
      assert.ok(res.body.limit >= 1 && res.body.limit <= 100, `"${query}" produced a bad limit`);
      assert.ok(res.body.offset >= 0, `"${query}" produced a negative offset`);
      assert.equal(res.body.total, 1, `"${query}" must not lose the stored row`);
      assert.ok(Array.isArray(res.body.items));
    }
  });

  test('returns the newest row first', async () => {
    const oldest = await seedClassification(app, { notes: 'first' });
    const middle = await seedClassification(app, { notes: 'second' });
    const newest = await seedClassification(app, { notes: 'third' });

    const res = await request.get('/api/classifications');

    assert.equal(res.status, 200);
    assert.deepEqual(
      res.body.items.map((i) => i.id),
      [newest.id, middle.id, oldest.id],
      'items must be ordered newest first',
    );
    assert.deepEqual(
      res.body.items.map((i) => i.notes),
      ['third', 'second', 'first'],
    );
  });

  test('filters by category', async () => {
    await seedClassification(app, {
      predictions: [prediction('glass', 0.91, 'Glass — jar')],
    });
    await seedClassification(app, {
      predictions: [prediction('glass', 0.77, 'Glass — bottle')],
    });
    await seedClassification(app); // plastic

    const res = await request.get('/api/classifications?category=glass');

    assert.equal(res.status, 200);
    assert.equal(res.body.total, 2);
    assert.equal(res.body.items.length, 2);
    for (const item of res.body.items) assert.equal(item.topCategory, 'glass');

    const empty = await request.get('/api/classifications?category=textile');
    assert.equal(empty.body.total, 0);
    assert.deepEqual(empty.body.items, []);

    // An unknown filter value may be rejected or simply match nothing; it must never
    // fall through and return unrelated rows.
    const unknown = await request.get('/api/classifications?category=unobtainium');
    if (unknown.status === 400) {
      assertErrorEnvelope(unknown, 400, 'bad_request');
    } else {
      assert.equal(unknown.status, 200);
      assert.equal(unknown.body.total, 0);
      assert.deepEqual(unknown.body.items, []);
    }
  });

  test('filters by source and by modelKind', async () => {
    await seedClassification(app, { source: 'webcam', modelKind: 'custom' });
    await seedClassification(app, { source: 'upload', modelKind: 'fallback' });
    await seedClassification(app, { source: 'upload', modelKind: 'fallback' });

    const webcam = await request.get('/api/classifications?source=webcam');
    assert.equal(webcam.body.total, 1);
    assert.equal(webcam.body.items[0].source, 'webcam');

    const upload = await request.get('/api/classifications?source=upload');
    assert.equal(upload.body.total, 2);
    for (const item of upload.body.items) assert.equal(item.source, 'upload');

    const custom = await request.get('/api/classifications?modelKind=custom');
    assert.equal(custom.body.total, 1);
    assert.equal(custom.body.items[0].modelKind, 'custom');

    const unknown = await request.get('/api/classifications?source=telepathy');
    if (unknown.status === 400) {
      assertErrorEnvelope(unknown, 400, 'bad_request');
    } else {
      assert.equal(unknown.status, 200);
      assert.equal(unknown.body.total, 0, 'an unmatched source filter matches nothing');
    }
  });

  test('filters by date range against created_at', async () => {
    await seedClassification(app);
    const today = new Date().toISOString().slice(0, 10);
    const tomorrow = new Date(Date.now() + 86_400_000).toISOString().slice(0, 10);
    const yesterday = new Date(Date.now() - 86_400_000).toISOString().slice(0, 10);

    const inRange = await request.get(`/api/classifications?from=${yesterday}&to=${tomorrow}`);
    assert.equal(inRange.status, 200);
    assert.equal(inRange.body.total, 1, `row from ${today} must fall inside the range`);

    const past = await request.get(`/api/classifications?to=${yesterday}`);
    assert.equal(past.body.total, 0, 'a row created today is after yesterday');

    const bad = await request.get('/api/classifications?from=not-a-date');
    if (bad.status === 400) {
      assertErrorEnvelope(bad, 400, 'bad_request');
    } else {
      assert.equal(bad.status, 200, 'an unparseable date must not produce a 500');
      assert.ok(Array.isArray(bad.body.items));
    }
  });

  test('includeImage=false strips the stored image', async () => {
    await seedClassification(app);

    const withImage = await request.get('/api/classifications');
    assert.equal(withImage.body.items[0].imageDataUrl, TINY_PNG_DATA_URL, 'images are on by default');

    const without = await request.get('/api/classifications?includeImage=false');
    assert.equal(without.status, 200);
    assert.equal(without.body.items.length, 1);
    assert.equal(without.body.items[0].imageDataUrl, null, 'includeImage=false nulls the image');
    assert.ok('imageDataUrl' in without.body.items[0], 'the key must stay, only the value goes');
    assert.equal(without.body.items[0].topCategory, 'plastic', 'the rest of the row is untouched');
  });
});

describe('GET /api/classifications/:id', () => {
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

  test('returns the stored row', async () => {
    const seeded = await seedClassification(app);

    const res = await request.get(`/api/classifications/${seeded.id}`);

    assert.equal(res.status, 200);
    assertHistoryItemShape(res.body.item);
    assert.deepEqual(res.body.item, seeded, 'GET /:id must match what POST returned');
  });

  test('answers 404 for a row that does not exist', async () => {
    const res = await request.get('/api/classifications/999999');
    assertErrorEnvelope(res, 404, 'not_found');
  });

  test('answers 400 for an id that is not a number', async () => {
    const res = await request.get('/api/classifications/abc');
    assertErrorEnvelope(res, 400, 'bad_request');
  });
});

describe('PATCH /api/classifications/:id', () => {
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

  test('a correction changes the effective category but not the prediction', async () => {
    const seeded = await seedClassification(app);
    assert.equal(seeded.effectiveCategory, 'plastic');

    const res = await request
      .patch(`/api/classifications/${seeded.id}`)
      .send({ correctedCategory: 'glass' });

    assert.equal(res.status, 200, JSON.stringify(res.body));
    const item = res.body.item;
    assertHistoryItemShape(item);
    assert.equal(item.correctedCategory, 'glass');
    assert.equal(item.effectiveCategory, 'glass');
    assert.equal(item.topCategory, 'plastic', 'the model output is never rewritten');
    assert.deepEqual(item.predictions, seeded.predictions);

    const reread = await request.get(`/api/classifications/${seeded.id}`);
    assert.equal(reread.body.item.effectiveCategory, 'glass', 'the correction must persist');
  });

  test('a correction can be cleared again', async () => {
    const seeded = await seedClassification(app);
    await request.patch(`/api/classifications/${seeded.id}`).send({ correctedCategory: 'metal' });

    const res = await request
      .patch(`/api/classifications/${seeded.id}`)
      .send({ correctedCategory: null });

    assert.equal(res.status, 200);
    assert.equal(res.body.item.correctedCategory, null);
    assert.equal(res.body.item.effectiveCategory, 'plastic', 'it falls back to the prediction');
  });

  test('notes can be edited on their own', async () => {
    const seeded = await seedClassification(app);

    const res = await request
      .patch(`/api/classifications/${seeded.id}`)
      .send({ notes: 'checked by hand' });

    assert.equal(res.status, 200);
    assert.equal(res.body.item.notes, 'checked by hand');
    assert.equal(
      res.body.item.correctedCategory,
      null,
      'an untouched field must keep its value',
    );
  });

  test('rejects an invalid corrected category', async () => {
    const seeded = await seedClassification(app);

    const res = await request
      .patch(`/api/classifications/${seeded.id}`)
      .send({ correctedCategory: 'unobtainium' });

    assertErrorEnvelope(res, 400, 'bad_request');

    const reread = await request.get(`/api/classifications/${seeded.id}`);
    assert.equal(reread.body.item.correctedCategory, null, 'the row must be untouched');
  });

  test('rejects notes longer than 500 characters', async () => {
    const seeded = await seedClassification(app);
    const res = await request
      .patch(`/api/classifications/${seeded.id}`)
      .send({ notes: 'n'.repeat(501) });
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('answers 404 for a row that does not exist', async () => {
    const res = await request
      .patch('/api/classifications/999999')
      .send({ correctedCategory: 'glass' });
    assertErrorEnvelope(res, 404, 'not_found');
  });
});

describe('DELETE /api/classifications', () => {
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

  test('deletes one row, then reports it gone', async () => {
    const seeded = await seedClassification(app);
    const other = await seedClassification(app);

    const res = await request.delete(`/api/classifications/${seeded.id}`);
    assert.equal(res.status, 204);
    assert.deepEqual(res.body, {}, '204 carries no body');

    assertErrorEnvelope(await request.get(`/api/classifications/${seeded.id}`), 404, 'not_found');
    assertErrorEnvelope(
      await request.delete(`/api/classifications/${seeded.id}`),
      404,
      'not_found',
    );

    const list = await request.get('/api/classifications');
    assert.equal(list.body.total, 1, 'only the requested row is deleted');
    assert.equal(list.body.items[0].id, other.id);
  });

  test('answers 400 for an id that is not a number', async () => {
    assertErrorEnvelope(await request.delete('/api/classifications/abc'), 400, 'bad_request');
  });

  test('refuses to clear the history without confirm=true', async () => {
    await seedClassification(app);

    assertErrorEnvelope(await request.delete('/api/classifications'), 400, 'bad_request');
    assertErrorEnvelope(
      await request.delete('/api/classifications?confirm=false'),
      400,
      'bad_request',
    );

    const list = await request.get('/api/classifications');
    assert.equal(list.body.total, 1, 'an unconfirmed clear must not delete anything');
  });

  test('clears the whole history with confirm=true', async () => {
    await seedClassification(app);
    await seedClassification(app);

    const res = await request.delete('/api/classifications?confirm=true');
    assert.equal(res.status, 204);

    const list = await request.get('/api/classifications');
    assert.equal(list.body.total, 0);
    assert.deepEqual(list.body.items, []);

    const health = await request.get('/api/health');
    assert.equal(health.body.db.classifications, 0);
  });
});
