import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  makeRequest,
  resetDatabase,
  closeDatabase,
  assertErrorEnvelope,
  CATEGORY_IDS,
  DEFAULT_REGION_ID,
  HEX_COLOR,
} from './helpers.js';

const REGION_KEYS = ['id', 'name', 'country', 'authority', 'updated', 'notes'];
const BIN_KEYS = ['id', 'name', 'colorName', 'colorHex', 'textColorHex', 'accepts', 'description'];
const GUIDANCE_KEYS = [
  'categoryId',
  'binId',
  'binName',
  'colorName',
  'colorHex',
  'textColorHex',
  'recyclable',
  'disposal',
  'prepSteps',
  'acceptedExamples',
  'rejectedExamples',
  'notes',
  'dropOff',
];

function assertBinShape(bin, where) {
  for (const key of BIN_KEYS) {
    assert.ok(key in bin, `${where}: bin "${bin.id}" is missing "${key}"`);
  }
  for (const key of ['id', 'name', 'colorName', 'description']) {
    assert.equal(typeof bin[key], 'string', `${where}: bin.${key} must be a string`);
    assert.ok(bin[key].length > 0, `${where}: bin.${key} must not be empty`);
  }
  assert.match(bin.colorHex, HEX_COLOR, `${where}: bin "${bin.id}" has a bad colorHex`);
  assert.match(bin.textColorHex, HEX_COLOR, `${where}: bin "${bin.id}" has a bad textColorHex`);
  assert.ok(Array.isArray(bin.accepts), `${where}: bin "${bin.id}".accepts must be an array`);
  assert.ok(bin.accepts.length > 0, `${where}: bin "${bin.id}" accepts nothing`);
  for (const id of bin.accepts) {
    assert.ok(CATEGORY_IDS.includes(id), `${where}: bin "${bin.id}" accepts unknown "${id}"`);
  }
}

function assertGuidanceShape(guidance, categoryId, where) {
  for (const key of GUIDANCE_KEYS) {
    assert.ok(key in guidance, `${where}: guidance for "${categoryId}" is missing "${key}"`);
  }
  assert.equal(guidance.categoryId, categoryId, `${where}: guidance.categoryId must be the key`);
  assert.equal(typeof guidance.binId, 'string');
  assert.ok(guidance.binId.length > 0);
  assert.equal(typeof guidance.recyclable, 'boolean', `${where}: recyclable must be a boolean`);
  assert.equal(typeof guidance.disposal, 'string');
  assert.ok(guidance.disposal.length > 0, `${where}: ${categoryId} needs disposal text`);
  assert.match(guidance.colorHex, HEX_COLOR);
  assert.match(guidance.textColorHex, HEX_COLOR);

  for (const key of ['prepSteps', 'acceptedExamples', 'rejectedExamples']) {
    assert.ok(Array.isArray(guidance[key]), `${where}: ${categoryId}.${key} must be an array`);
    for (const entry of guidance[key]) {
      assert.equal(typeof entry, 'string');
      assert.ok(entry.length > 0, `${where}: ${categoryId}.${key} holds an empty string`);
    }
  }
  assert.ok(guidance.prepSteps.length > 0, `${where}: ${categoryId} needs at least one prep step`);
  assert.equal(typeof guidance.notes, 'string');
  assert.ok(
    guidance.dropOff === null || typeof guidance.dropOff === 'string',
    `${where}: ${categoryId}.dropOff must be a string or null`,
  );
}

describe('rules API', () => {
  let request;
  let index;

  before(async () => {
    request = await makeRequest();
    const res = await request.get('/api/rules');
    assert.equal(res.status, 200, `GET /api/rules returned ${res.status}`);
    index = res.body;
  });

  beforeEach(async () => {
    await resetDatabase();
  });

  after(async () => {
    await closeDatabase();
  });

  test('GET /api/rules lists at least seven regions and a resolvable default', () => {
    assert.ok(Array.isArray(index.regions), 'regions must be an array');
    assert.ok(
      index.regions.length >= 7,
      `expected at least 7 regions, got ${index.regions.length}`,
    );

    const ids = index.regions.map((r) => r.id);
    assert.equal(new Set(ids).size, ids.length, 'region ids must be unique');

    assert.equal(typeof index.defaultRegion, 'string');
    assert.ok(
      ids.includes(index.defaultRegion),
      `defaultRegion "${index.defaultRegion}" is not among ${ids.join(', ')}`,
    );
    assert.equal(index.defaultRegion, DEFAULT_REGION_ID);

    for (const region of index.regions) {
      for (const key of REGION_KEYS) {
        assert.ok(key in region, `region "${region.id}" is missing "${key}"`);
        assert.equal(typeof region[key], 'string', `region.${key} must be a string`);
      }
      // `notes` may legitimately be blank; the identifying fields may not.
      for (const key of ['id', 'name', 'country', 'authority', 'updated']) {
        assert.ok(region[key].length > 0, `region "${region.id}".${key} must not be empty`);
      }
    }

    // The index is a summary: heavy payloads belong on the per-region endpoint.
    for (const region of index.regions) {
      assert.ok(!('bins' in region), 'the region index must not inline bins');
      assert.ok(!('categories' in region), 'the region index must not inline guidance');
    }
  });

  test('GET /api/rules/us-generic returns bins plus all ten category keys', async () => {
    const res = await request.get('/api/rules/us-generic');

    assert.equal(res.status, 200);
    assert.equal(res.body.region.id, 'us-generic');
    for (const key of REGION_KEYS) {
      assert.ok(key in res.body.region, `region is missing "${key}"`);
    }

    assert.ok(Array.isArray(res.body.bins), 'bins must be an array');
    assert.ok(res.body.bins.length > 0, 'a region must define at least one bin');
    const binIds = res.body.bins.map((b) => b.id);
    assert.equal(new Set(binIds).size, binIds.length, 'bin ids must be unique within a region');
    for (const bin of res.body.bins) assertBinShape(bin, 'us-generic');

    const categoryKeys = Object.keys(res.body.categories);
    assert.equal(categoryKeys.length, 10, 'a region must cover all ten categories');
    assert.deepEqual(
      [...categoryKeys].sort(),
      [...CATEGORY_IDS].sort(),
      'region guidance keys must be exactly the canonical ids',
    );

    for (const id of CATEGORY_IDS) {
      const guidance = res.body.categories[id];
      assertGuidanceShape(guidance, id, 'us-generic');
      assert.ok(
        binIds.includes(guidance.binId),
        `guidance for "${id}" points at unknown bin "${guidance.binId}"`,
      );
    }
  });

  test('every listed region is fully resolvable', async () => {
    for (const summary of index.regions) {
      const res = await request.get(`/api/rules/${encodeURIComponent(summary.id)}`);
      assert.equal(res.status, 200, `region "${summary.id}" did not resolve`);
      assert.equal(res.body.region.id, summary.id);
      assert.deepEqual(
        Object.keys(res.body.categories).sort(),
        [...CATEGORY_IDS].sort(),
        `region "${summary.id}" does not cover all ten categories`,
      );
      for (const bin of res.body.bins) assertBinShape(bin, summary.id);
    }
  });

  test('GET /api/rules/does-not-exist answers 404 with the error envelope', async () => {
    const res = await request.get('/api/rules/does-not-exist');

    const error = assertErrorEnvelope(res, 404, 'not_found');
    assert.ok(error.message.length > 5, 'a 404 must explain what was not found');
  });

  test('GET /api/rules/:region/:category returns guidance and its resolved bin', async () => {
    const res = await request.get('/api/rules/us-generic/plastic');

    assert.equal(res.status, 200);
    assert.equal(res.body.region.id, 'us-generic');
    assert.equal(res.body.category.id, 'plastic');
    assert.equal(typeof res.body.category.label, 'string');
    assert.match(res.body.category.colorHex, HEX_COLOR);

    assertGuidanceShape(res.body.guidance, 'plastic', 'us-generic/plastic');
    assertBinShape(res.body.bin, 'us-generic/plastic');

    assert.equal(
      res.body.bin.id,
      res.body.guidance.binId,
      'the resolved bin must be the one the guidance names',
    );
    assert.equal(res.body.bin.name, res.body.guidance.binName);
    assert.equal(res.body.bin.colorHex, res.body.guidance.colorHex);
    assert.equal(res.body.bin.colorName, res.body.guidance.colorName);
    assert.ok(
      res.body.bin.accepts.includes('plastic'),
      'the resolved bin must accept the category it was resolved for',
    );
  });

  test('every region/category pair resolves consistently', async () => {
    for (const summary of index.regions) {
      for (const categoryId of CATEGORY_IDS) {
        const res = await request.get(`/api/rules/${summary.id}/${categoryId}`);
        assert.equal(res.status, 200, `${summary.id}/${categoryId} returned ${res.status}`);
        assert.equal(res.body.guidance.categoryId, categoryId);
        assert.equal(res.body.bin.id, res.body.guidance.binId);
        assert.ok(
          res.body.bin.accepts.includes(categoryId),
          `${summary.id}: bin "${res.body.bin.id}" does not accept "${categoryId}"`,
        );
      }
    }
  });

  test('an unknown category answers 404', async () => {
    const res = await request.get('/api/rules/us-generic/unobtainium');
    const error = assertErrorEnvelope(res, 404, 'not_found');
    assert.ok(error.message.length > 5, 'a 404 must explain what was not found');
  });

  test('an unknown region answers 404 even with a valid category', async () => {
    const res = await request.get('/api/rules/atlantis/plastic');
    assertErrorEnvelope(res, 404, 'not_found');
  });
});
