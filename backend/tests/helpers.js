/**
 * Shared harness for the backend test-suite.
 *
 * `src/config.js` snapshots `process.env` at import time, so nothing from `src/` may
 * appear in this module's *static* import graph: every backend module is pulled in
 * through a dynamic `import()` below, after the environment has been rewritten.
 */
import assert from 'node:assert/strict';
import os from 'node:os';
import { setTimeout as sleep } from 'node:timers/promises';
import supertest from 'supertest';

process.env.NODE_ENV = 'test';
process.env.DB_FILE = ':memory:';
process.env.LOG_LEVEL = 'error';
process.env.MAX_IMAGE_DATA_URL = '400000';
process.env.DEFAULT_REGION = process.env.DEFAULT_REGION ?? 'us-generic';
// A config that derives DB_FILE from DATA_DIR must not try to mkdir the container's
// /data path when the suite runs on a developer machine.
process.env.DATA_DIR = process.env.DATA_DIR ?? os.tmpdir();

/** Character budget for `imageDataUrl`, mirrored from the env set above. */
export const MAX_IMAGE_DATA_URL = Number(process.env.MAX_IMAGE_DATA_URL);

export const DEFAULT_REGION_ID = 'us-generic';

/** Canonical taxonomy, in the order given by ARCHITECTURE.md section 3. */
export const CATEGORY_IDS = Object.freeze([
  'plastic',
  'paper',
  'cardboard',
  'glass',
  'metal',
  'organic',
  'ewaste',
  'hazardous',
  'textile',
  'trash',
]);

/** Canonical category colours, ARCHITECTURE.md section 3. */
export const CATEGORY_COLORS = Object.freeze({
  plastic: '#2563eb',
  paper: '#0ea5e9',
  cardboard: '#b45309',
  glass: '#059669',
  metal: '#64748b',
  organic: '#65a30d',
  ewaste: '#7c3aed',
  hazardous: '#dc2626',
  textile: '#db2777',
  trash: '#44403c',
});

export const HEX_COLOR = /^#[0-9a-f]{6}$/i;
export const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

/** 1x1 transparent PNG — the smallest thing that is still a real data URL. */
export const TINY_PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk' +
  '+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

let dbModulePromise = null;
let appPromise = null;

/** Lazily loads `src/db.js` (never at module scope — see the header comment). */
export function dbModule() {
  if (!dbModulePromise) dbModulePromise = import('../src/db.js');
  return dbModulePromise;
}

/**
 * Builds the express app once per test process. Migrations are run explicitly first so
 * that the schema exists even for a test that only talks to the database.
 */
export async function makeApp() {
  if (!appPromise) {
    appPromise = (async () => {
      const db = await dbModule();
      // Passing the handle is safe whether runMigrations takes one or ignores its args.
      db.runMigrations(db.getDb());
      const { createApp } = await import('../src/app.js');
      return createApp();
    })();
  }
  return appPromise;
}

/** A supertest agent bound to the app. */
export async function makeRequest() {
  return supertest(await makeApp());
}

/** Accepts either the express app or an already-built supertest agent. */
function agentFor(target) {
  if (!target) throw new Error('an express app or supertest agent is required');
  return typeof target.listen === 'function' ? supertest(target) : target;
}

export async function resetDatabase() {
  const db = await dbModule();
  db.resetDb();
}

export async function closeDatabase() {
  if (!dbModulePromise) return;
  const db = await dbModule();
  db.closeDb();
}

/** A body that must always be accepted by `POST /api/classifications`. */
export function validBody(overrides = {}) {
  return {
    predictions: [
      { category: 'plastic', label: 'Plastic — water bottle', confidence: 0.82 },
      { category: 'metal', label: 'Metal — beverage can', confidence: 0.11 },
      { category: 'trash', label: 'General Waste', confidence: 0.03 },
    ],
    rawLabels: [
      { label: 'water bottle', confidence: 0.72, index: 898 },
      { label: 'pop bottle, soda bottle', confidence: 0.14, index: 737 },
    ],
    source: 'upload',
    modelKind: 'fallback',
    regionId: DEFAULT_REGION_ID,
    imageDataUrl: TINY_PNG_DATA_URL,
    notes: 'seeded by the backend test-suite',
    durationMs: 128,
    ...overrides,
  };
}

/**
 * POSTs a valid classification and returns the stored HistoryItem.
 * Rows are spaced a couple of milliseconds apart so `created_at` strictly increases and
 * "newest first" ordering is deterministic even when several rows are seeded in one tick.
 */
export async function seedClassification(app, overrides = {}) {
  await sleep(2);
  const res = await agentFor(app).post('/api/classifications').send(validBody(overrides));
  if (res.status !== 201) {
    throw new Error(
      `seedClassification expected 201, got ${res.status}: ${JSON.stringify(res.body)}`,
    );
  }
  assert.ok(res.body.item, 'POST /api/classifications must answer with { item }');
  return res.body.item;
}

/** A single prediction entry, for building bodies in tests. */
export function prediction(category, confidence, label = `${category} item`) {
  return { category, label, confidence };
}

/** An `imageDataUrl` that is one character past the configured limit. */
export function oversizedDataUrl(length = MAX_IMAGE_DATA_URL + 1) {
  const prefix = 'data:image/png;base64,';
  return prefix + 'A'.repeat(Math.max(1, length - prefix.length));
}

/** Asserts the error envelope from ARCHITECTURE.md section 4. */
export function assertErrorEnvelope(res, status, code) {
  assert.equal(
    res.status,
    status,
    `expected HTTP ${status}, got ${res.status}: ${JSON.stringify(res.body)}`,
  );
  assert.ok(res.body && typeof res.body === 'object', 'error body must be a JSON object');
  const { error } = res.body;
  assert.ok(error && typeof error === 'object', 'error body must carry an "error" object');
  assert.equal(error.code, code, `expected error code "${code}", got "${error.code}"`);
  assert.equal(typeof error.message, 'string', 'error.message must be a string');
  assert.ok(error.message.length > 0, 'error.message must not be empty');
  assert.ok('details' in error, 'the error envelope always carries a details key');
  return error;
}

export function assertIsoTimestamp(value, what) {
  assert.equal(typeof value, 'string', `${what} must be an ISO-8601 string`);
  const parsed = Date.parse(value);
  assert.ok(Number.isFinite(parsed), `${what} must parse as a date, got "${value}"`);
  return parsed;
}

/** Asserts the full HistoryItem shape from ARCHITECTURE.md section 4. */
export function assertHistoryItemShape(item) {
  assert.ok(item && typeof item === 'object', 'item must be an object');
  assert.ok(Number.isInteger(item.id) && item.id > 0, 'id must be a positive integer');
  assertIsoTimestamp(item.createdAt, 'createdAt');

  assert.ok(CATEGORY_IDS.includes(item.topCategory), `topCategory "${item.topCategory}" is not canonical`);
  assert.equal(typeof item.topLabel, 'string');
  assert.ok(item.topLabel.length > 0, 'topLabel must not be empty');
  assert.equal(typeof item.topConfidence, 'number');
  assert.ok(item.topConfidence >= 0 && item.topConfidence <= 1, 'topConfidence must be 0..1');

  assert.ok(Array.isArray(item.predictions), 'predictions must be an array');
  assert.ok(item.predictions.length >= 1 && item.predictions.length <= 10);
  for (const p of item.predictions) {
    assert.ok(CATEGORY_IDS.includes(p.category), `prediction category "${p.category}" is not canonical`);
    assert.equal(typeof p.label, 'string');
    assert.equal(typeof p.confidence, 'number');
    assert.ok(p.confidence >= 0 && p.confidence <= 1);
  }

  if (item.rawLabels !== null) {
    assert.ok(Array.isArray(item.rawLabels), 'rawLabels must be an array or null');
    for (const raw of item.rawLabels) {
      assert.equal(typeof raw.label, 'string');
      assert.equal(typeof raw.confidence, 'number');
      assert.ok(Number.isInteger(raw.index));
    }
  }

  assert.ok(['webcam', 'upload'].includes(item.source), `bad source "${item.source}"`);
  assert.ok(['custom', 'fallback'].includes(item.modelKind), `bad modelKind "${item.modelKind}"`);
  assert.equal(typeof item.regionId, 'string');
  assert.ok(item.regionId.length > 0);

  assert.ok(item.imageDataUrl === null || typeof item.imageDataUrl === 'string');
  assert.ok(item.correctedCategory === null || CATEGORY_IDS.includes(item.correctedCategory));
  assert.ok(item.notes === null || typeof item.notes === 'string');
  assert.ok(item.durationMs === null || Number.isInteger(item.durationMs));

  const expectedEffective = item.correctedCategory ?? item.topCategory;
  assert.equal(
    item.effectiveCategory,
    expectedEffective,
    'effectiveCategory must be correctedCategory ?? topCategory',
  );
}

export { sleep };
