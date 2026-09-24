import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  makeApp,
  makeRequest,
  resetDatabase,
  closeDatabase,
  seedClassification,
  prediction,
  assertErrorEnvelope,
  assertIsoTimestamp,
  CATEGORY_IDS,
  CATEGORY_COLORS,
  DEFAULT_REGION_ID,
  HEX_COLOR,
  ISO_DAY,
} from './helpers.js';

const TOP_CONFIDENCE = 0.6;

const LABELS = {
  plastic: 'Plastic — water bottle',
  glass: 'Glass — jar',
  hazardous: 'Hazardous — AA battery',
};

/** 5 rows: 3 plastic / 1 glass / 1 hazardous, 3 upload / 2 webcam, 3 fallback / 2 custom. */
const MIX = [
  { category: 'plastic', source: 'upload', modelKind: 'fallback' },
  { category: 'plastic', source: 'upload', modelKind: 'fallback' },
  { category: 'plastic', source: 'webcam', modelKind: 'custom' },
  { category: 'glass', source: 'upload', modelKind: 'fallback' },
  { category: 'hazardous', source: 'webcam', modelKind: 'custom' },
];

function assertClose(actual, expected, what, tolerance = 1e-6) {
  assert.equal(typeof actual, 'number', `${what} must be a number`);
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${what}: expected ~${expected}, got ${actual}`,
  );
}

function countBy(rows, key) {
  const counts = new Map();
  for (const row of rows) counts.set(row[key], (counts.get(row[key]) ?? 0) + 1);
  return counts;
}

function todayUtc() {
  return new Date().toISOString().slice(0, 10);
}

describe('GET /api/stats', () => {
  let app;
  let request;
  let guidance;

  before(async () => {
    app = await makeApp();
    request = await makeRequest();

    // recyclableRate is defined against the region's rules, so the expectation is derived
    // from the same source of truth instead of being hard-coded here.
    const rules = await request.get(`/api/rules/${DEFAULT_REGION_ID}`);
    assert.equal(rules.status, 200, 'the stats suite needs the default region to resolve');
    guidance = rules.body.categories;
    assert.equal(
      guidance.trash.recyclable,
      false,
      'general waste must not be marked recyclable',
    );
  });

  beforeEach(async () => {
    await resetDatabase();
  });

  after(async () => {
    await closeDatabase();
  });

  async function seedMix() {
    for (const row of MIX) {
      await seedClassification(app, {
        predictions: [prediction(row.category, TOP_CONFIDENCE, LABELS[row.category])],
        source: row.source,
        modelKind: row.modelKind,
        regionId: DEFAULT_REGION_ID,
      });
    }
  }

  test('an empty database reports zeroes, not nulls', async () => {
    const res = await request.get('/api/stats');

    assert.equal(res.status, 200);
    assert.equal(res.body.total, 0);
    assert.equal(res.body.totalInWindow, 0);
    assert.equal(res.body.windowDays, 30, 'the documented default window is 30 days');
    assert.equal(res.body.avgConfidence, 0, 'avgConfidence is 0, never null, on an empty set');
    assert.equal(res.body.recyclableRate, 0);
    assert.equal(res.body.lastClassifiedAt, null);

    assert.ok(Array.isArray(res.body.byCategory));
    assert.equal(res.body.byCategory.length, 10, 'all ten categories are always present');
    assert.deepEqual(
      [...res.body.byCategory.map((c) => c.category)].sort(),
      [...CATEGORY_IDS].sort(),
    );
    for (const entry of res.body.byCategory) {
      assert.equal(entry.count, 0, `${entry.category} must be at zero`);
      assert.equal(entry.share, 0, `${entry.category}.share must be 0, not NaN`);
      assert.equal(typeof entry.label, 'string');
      assert.ok(entry.label.length > 0);
      assert.match(entry.colorHex, HEX_COLOR);
      assert.equal(entry.colorHex.toLowerCase(), CATEGORY_COLORS[entry.category]);
    }

    assert.ok(Array.isArray(res.body.bySource));
    assert.ok(Array.isArray(res.body.byModelKind));
    assert.ok(Array.isArray(res.body.topLabels));
    assert.equal(
      res.body.bySource.reduce((sum, e) => sum + e.count, 0),
      0,
      'bySource cannot count rows that do not exist',
    );
  });

  test('byDay is dense over the window even with no data', async () => {
    const res = await request.get('/api/stats?days=7');

    assert.equal(res.status, 200);
    assert.equal(res.body.windowDays, 7);
    assert.ok(Array.isArray(res.body.byDay));
    assert.equal(res.body.byDay.length, 7, 'byDay holds one entry per day in the window');

    for (const entry of res.body.byDay) {
      assert.match(entry.day, ISO_DAY, `"${entry.day}" is not a YYYY-MM-DD day`);
      assert.equal(entry.count, 0);
    }

    const days = res.body.byDay.map((e) => e.day);
    assert.deepEqual([...days].sort(), days, 'byDay runs oldest to newest');
    assert.equal(new Set(days).size, days.length, 'byDay must not repeat a day');
    assert.equal(days.at(-1), todayUtc(), 'the window ends today');

    // Dense means consecutive: every step is exactly one day.
    for (let i = 1; i < days.length; i += 1) {
      const step = Date.parse(`${days[i]}T00:00:00Z`) - Date.parse(`${days[i - 1]}T00:00:00Z`);
      assert.equal(step, 86_400_000, `gap between ${days[i - 1]} and ${days[i]}`);
    }
  });

  test('counts a known mix correctly', async () => {
    await seedMix();

    const res = await request.get('/api/stats');
    assert.equal(res.status, 200);

    assert.equal(res.body.total, 5);
    assert.equal(res.body.totalInWindow, 5);
    assertClose(res.body.avgConfidence, TOP_CONFIDENCE, 'avgConfidence');

    const expectedByCategory = countBy(MIX, 'category');
    const byCategory = new Map(res.body.byCategory.map((e) => [e.category, e]));
    assert.equal(byCategory.size, 10, 'all ten categories stay present once there is data');
    for (const id of CATEGORY_IDS) {
      const expected = expectedByCategory.get(id) ?? 0;
      assert.equal(byCategory.get(id).count, expected, `byCategory count for ${id}`);
      assertClose(byCategory.get(id).share, expected / 5, `byCategory share for ${id}`);
    }

    const counts = res.body.byCategory.map((e) => e.count);
    assert.deepEqual([...counts].sort((a, b) => b - a), counts, 'byCategory is sorted desc');

    const shareSum = res.body.byCategory.reduce((sum, e) => sum + e.share, 0);
    assertClose(shareSum, 1, 'the shares of every category', 1e-6);

    const bySource = new Map(res.body.bySource.map((e) => [e.source, e.count]));
    assert.equal(bySource.get('upload'), 3);
    assert.equal(bySource.get('webcam'), 2);
    assert.equal(
      res.body.bySource.reduce((sum, e) => sum + e.count, 0),
      5,
      'bySource must account for every row',
    );

    const byModelKind = new Map(res.body.byModelKind.map((e) => [e.modelKind, e.count]));
    assert.equal(byModelKind.get('fallback'), 3);
    assert.equal(byModelKind.get('custom'), 2);
    assert.equal(
      res.body.byModelKind.reduce((sum, e) => sum + e.count, 0),
      5,
      'byModelKind must account for every row',
    );
  });

  test('byDay ends with today and counts todays rows', async () => {
    await seedMix();

    const res = await request.get('/api/stats');
    assert.equal(res.status, 200);
    assert.equal(res.body.byDay.length, 30);

    const last = res.body.byDay.at(-1);
    assert.equal(last.day, todayUtc());
    assert.equal(last.count, 5, 'every seeded row was created today');

    const total = res.body.byDay.reduce((sum, e) => sum + e.count, 0);
    assert.equal(total, res.body.totalInWindow, 'byDay must sum to the window total');
  });

  test('topLabels ranks the labels that were actually stored', async () => {
    await seedMix();

    const res = await request.get('/api/stats');
    assert.ok(Array.isArray(res.body.topLabels));
    assert.ok(res.body.topLabels.length <= 10, 'topLabels is capped at 10 entries');
    assert.ok(res.body.topLabels.length > 0);

    const labelCounts = res.body.topLabels.map((e) => e.count);
    assert.deepEqual(
      [...labelCounts].sort((a, b) => b - a),
      labelCounts,
      'topLabels is sorted desc by count',
    );

    const top = res.body.topLabels[0];
    assert.equal(top.label, LABELS.plastic, 'the most frequent label leads');
    assert.equal(top.count, 3);
    assert.equal(
      res.body.topLabels.reduce((sum, e) => sum + e.count, 0),
      5,
      'three distinct labels over five rows must account for all of them',
    );
  });

  test('lastClassifiedAt is the newest row', async () => {
    await seedMix();

    const list = await request.get('/api/classifications?limit=1');
    const newest = list.body.items[0];

    const res = await request.get('/api/stats');
    assertIsoTimestamp(res.body.lastClassifiedAt, 'lastClassifiedAt');
    assert.equal(res.body.lastClassifiedAt, newest.createdAt);
  });

  test('recyclableRate follows the regions rules and the effective category', async () => {
    await seedMix();

    const expectedBefore =
      MIX.filter((row) => guidance[row.category].recyclable).length / MIX.length;

    const before = await request.get(`/api/stats?regionId=${DEFAULT_REGION_ID}`);
    assert.equal(before.status, 200);
    assert.ok(
      before.body.recyclableRate >= 0 && before.body.recyclableRate <= 1,
      `recyclableRate must be a share, got ${before.body.recyclableRate}`,
    );
    assertClose(before.body.recyclableRate, expectedBefore, 'recyclableRate');

    // Correcting a row changes its effective category, which the rate is defined against.
    const plastic = await request.get('/api/classifications?category=plastic&limit=1');
    const target = plastic.body.items[0];
    const patched = await request
      .patch(`/api/classifications/${target.id}`)
      .send({ correctedCategory: 'trash' });
    assert.equal(patched.status, 200);
    assert.equal(patched.body.item.effectiveCategory, 'trash');

    const effective = MIX.map((row) => row.category);
    effective[effective.indexOf('plastic')] = 'trash';
    const expectedAfter = effective.filter((id) => guidance[id].recyclable).length / effective.length;

    const after = await request.get(`/api/stats?regionId=${DEFAULT_REGION_ID}`);
    assertClose(after.body.recyclableRate, expectedAfter, 'recyclableRate after a correction');
    assert.ok(
      after.body.recyclableRate < before.body.recyclableRate,
      'correcting a recyclable row to general waste must lower the rate',
    );
    assert.equal(after.body.total, 5, 'a correction never changes the row count');
  });

  test('days is bounded by the documented maximum of 365', async () => {
    const ok = await request.get('/api/stats?days=365');
    assert.equal(ok.status, 200);
    assert.equal(ok.body.windowDays, 365);
    assert.equal(ok.body.byDay.length, 365);

    const tooMany = await request.get('/api/stats?days=400');
    assertErrorEnvelope(tooMany, 400, 'bad_request');

    const negative = await request.get('/api/stats?days=-1');
    assertErrorEnvelope(negative, 400, 'bad_request');
  });
});
