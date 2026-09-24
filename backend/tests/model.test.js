import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import { makeRequest, resetDatabase, closeDatabase, assertErrorEnvelope } from './helpers.js';

const ENGINES = [
  { key: 'custom', modelUrl: '/models/custom/model.json', metadataUrl: '/models/custom/metadata.json' },
  {
    key: 'fallback',
    modelUrl: '/models/mobilenet_v2/model.json',
    metadataUrl: '/models/mobilenet_v2/metadata.json',
  },
];

/** The id alphabet the route validates against: a model id is also a path segment. */
const MODEL_ID = /^[A-Za-z0-9_.-]{1,64}$/;

/** The two directories whose ids carry a fixed meaning, whatever else is installed. */
const CUSTOM_ID = 'custom';
const FALLBACK_ID = 'mobilenet_v2';

/**
 * Whether a model is on disk depends on the environment (the fallback is baked into the
 * image at build time, the custom model only exists once the user trains one), so every
 * assertion here has to hold for both the present and the absent case.
 */
function assertEngineShape(engine, spec) {
  assert.ok(engine && typeof engine === 'object', `${spec.key} block is required`);
  assert.equal(typeof engine.available, 'boolean', `${spec.key}.available must be a boolean`);
  assert.ok('modelUrl' in engine, `${spec.key}.modelUrl key is required`);
  assert.ok('metadataUrl' in engine, `${spec.key}.metadataUrl key is required`);
  assert.ok('metadata' in engine, `${spec.key}.metadata key is required`);

  if (engine.available) {
    assert.equal(engine.modelUrl, spec.modelUrl, `${spec.key}.modelUrl must be the served path`);
    // metadata.json is optional (ARCHITECTURE.md section 2.3 — the service synthesises defaults),
    // so the url is either the served path or null; it is never some other path.
    assert.ok(
      engine.metadataUrl === null || engine.metadataUrl === spec.metadataUrl,
      `${spec.key}.metadataUrl must be "${spec.metadataUrl}" or null, got "${engine.metadataUrl}"`,
    );
    assert.ok(
      engine.metadata === null || typeof engine.metadata === 'object',
      `${spec.key}.metadata must be an object or null`,
    );
    if (engine.metadata) {
      assert.ok(!Array.isArray(engine.metadata), `${spec.key}.metadata must not be an array`);
      // metadata.json is optional, but when present it is the uniform descriptor.
      assert.ok(
        Array.isArray(engine.metadata.inputRange) && engine.metadata.inputRange.length === 2,
        `${spec.key} metadata.inputRange must be a two-element array`,
      );
      assert.deepEqual(
        engine.metadata.inputRange,
        [0, 1],
        'both engines take [0,1] input — see ARCHITECTURE.md section 2',
      );
      assert.equal(typeof engine.metadata.inputSize, 'number');
      assert.ok(engine.metadata.inputSize > 0);
    }
  } else {
    assert.equal(engine.modelUrl, null, `${spec.key}.modelUrl must be null when unavailable`);
    assert.equal(engine.metadataUrl, null, `${spec.key}.metadataUrl must be null when unavailable`);
    assert.equal(engine.metadata, null, `${spec.key}.metadata must be null when unavailable`);
  }
}

/**
 * One registry entry (ARCHITECTURE.md section 2.4). Like the block above this asserts *shape and
 * invariants only*: which models exist is a property of the machine the suite runs on.
 */
function assertRegistryEntry(entry) {
  assert.ok(entry && typeof entry === 'object' && !Array.isArray(entry), 'a model entry is an object');

  assert.equal(typeof entry.id, 'string', 'model.id must be a string');
  assert.match(entry.id, MODEL_ID, `model.id "${entry.id}" must be a safe path segment`);

  assert.equal(typeof entry.displayName, 'string', `${entry.id}.displayName must be a string`);
  assert.ok(entry.displayName.length > 0, `${entry.id}.displayName must not be empty`);
  assert.equal(typeof entry.description, 'string', `${entry.id}.description must be a string`);

  assert.ok(
    ['custom', 'imagenet'].includes(entry.kind),
    `${entry.id}.kind must be "custom" | "imagenet", got "${entry.kind}"`,
  );
  // A discovered entry exists precisely because its model.json is readable.
  assert.equal(entry.available, true, `${entry.id}.available must be true`);
  assert.ok(
    ['models-dir', 'bundled'].includes(entry.source),
    `${entry.id}.source must say which root it came from, got "${entry.source}"`,
  );

  assert.equal(
    entry.modelUrl,
    `/models/${entry.id}/model.json`,
    `${entry.id}.modelUrl must be the served path`,
  );
  assert.ok(
    entry.metadataUrl === null || entry.metadataUrl === `/models/${entry.id}/metadata.json`,
    `${entry.id}.metadataUrl must be the served path or null`,
  );

  assert.ok(
    entry.metadata && typeof entry.metadata === 'object' && !Array.isArray(entry.metadata),
    `${entry.id}.metadata must be an object`,
  );
  // The whole point of the uniform descriptor: classifier.js needs no per-model branch.
  assert.deepEqual(entry.metadata.inputRange, [0, 1], `${entry.id} must take [0,1] input`);
  assert.equal(typeof entry.metadata.inputSize, 'number', `${entry.id}.metadata.inputSize is a number`);
  assert.ok(entry.metadata.inputSize > 0, `${entry.id}.metadata.inputSize must be positive`);
  assert.equal(
    entry.kind,
    entry.metadata.labelKind === 'waste' ? 'custom' : 'imagenet',
    `${entry.id}.kind must follow metadata.labelKind`,
  );

  assert.ok(
    entry.downloadBytes === null ||
      (typeof entry.downloadBytes === 'number' && Number.isFinite(entry.downloadBytes) && entry.downloadBytes > 0),
    `${entry.id}.downloadBytes must be a positive number or null, got ${entry.downloadBytes}`,
  );
  assert.equal(typeof entry.recommended, 'boolean', `${entry.id}.recommended must be a boolean`);
}

/** Sort key used by the registry ordering: an unknown size is offered last. */
function downloadWeight(entry) {
  return entry.downloadBytes === null ? Number.POSITIVE_INFINITY : entry.downloadBytes;
}

function assertRegistryInvariants(models, defaultModelId) {
  assert.ok(Array.isArray(models), 'models must be an array');
  for (const entry of models) assertRegistryEntry(entry);

  const ids = models.map((entry) => entry.id);
  assert.equal(new Set(ids).size, ids.length, 'a model id appears at most once');

  const recommended = models.filter((entry) => entry.recommended);
  if (models.length === 0) {
    assert.equal(defaultModelId, null, 'defaultModelId is null exactly when nothing is installed');
    assert.equal(recommended.length, 0, 'an empty registry recommends nothing');
  } else {
    assert.equal(recommended.length, 1, `exactly one model is recommended, got ${recommended.length}`);
    assert.equal(defaultModelId, recommended[0].id, 'defaultModelId names the recommended model');
    assert.ok(ids.includes(defaultModelId), 'defaultModelId must name a listed model');
  }
}

function assertRegistryOrdering(models) {
  const customIndex = models.findIndex((entry) => entry.id === CUSTOM_ID);
  if (customIndex !== -1) {
    assert.equal(customIndex, 0, 'the custom model is offered first when it exists');
  }

  // Everything after the custom slot is ordered by download size, cheapest first.
  const rest = customIndex === 0 ? models.slice(1) : models;
  for (let i = 1; i < rest.length; i += 1) {
    const previous = rest[i - 1];
    const current = rest[i];
    assert.ok(
      downloadWeight(previous) <= downloadWeight(current),
      `models must be ordered by download size: ${previous.id} (${previous.downloadBytes}) came ` +
        `before ${current.id} (${current.downloadBytes})`,
    );
  }

  // The head of that order is the auto-pick, so the cheapest download is what a new user gets.
  if (models.length > 0) {
    assert.equal(models[0].recommended, true, 'the first model in the order is the recommended one');
  }
}

describe('GET /api/model/status', () => {
  let request;
  let body;
  let headers;

  before(async () => {
    request = await makeRequest();
  });

  beforeEach(async () => {
    await resetDatabase();
    const res = await request.get('/api/model/status');
    assert.equal(res.status, 200, `expected 200, got ${res.status}`);
    body = res.body;
    headers = res.headers;
  });

  test('returns the three-part shape', () => {
    assert.ok('custom' in body, 'custom block is required');
    assert.ok('fallback' in body, 'fallback block is required');
    assert.ok('active' in body, 'active is required');
    assert.ok('searchedPaths' in body, 'searchedPaths is required');
    for (const spec of ENGINES) assertEngineShape(body[spec.key], spec);
  });

  test('active is one of the three literals and agrees with availability', () => {
    assert.ok(
      ['custom', 'fallback', 'none'].includes(body.active),
      `active must be "custom" | "fallback" | "none", got "${body.active}"`,
    );

    // The frontend prefers custom and silently falls back — the API must say the same.
    const expected = body.custom.available ? 'custom' : body.fallback.available ? 'fallback' : 'none';
    assert.equal(body.active, expected, 'active must follow custom > fallback > none');
  });

  test('searchedPaths is a non-empty array of strings', () => {
    assert.ok(Array.isArray(body.searchedPaths), 'searchedPaths must be an array');
    assert.ok(body.searchedPaths.length > 0, 'searchedPaths must not be empty');
    for (const entry of body.searchedPaths) {
      assert.equal(typeof entry, 'string', 'searchedPaths holds strings');
      assert.ok(entry.length > 0, 'searchedPaths holds no empty strings');
    }
    assert.equal(
      new Set(body.searchedPaths).size,
      body.searchedPaths.length,
      'searchedPaths must not repeat itself',
    );
  });

  test('is never cached', () => {
    const cacheControl = headers['cache-control'];
    assert.equal(typeof cacheControl, 'string', 'Cache-Control header is required');
    assert.match(
      cacheControl,
      /no-store/,
      `model status must not be cached, got "${cacheControl}"`,
    );
  });

  test('re-probes on every request rather than freezing the first answer', async () => {
    const again = await request.get('/api/model/status');
    assert.equal(again.status, 200);
    assert.deepEqual(again.body.searchedPaths, body.searchedPaths);
    assert.equal(again.body.active, body.active);
    assert.equal(again.body.custom.available, body.custom.available);
    assert.equal(again.body.fallback.available, body.fallback.available);
  });

  test('carries the model registry and its default', () => {
    assert.ok('models' in body, 'models is required');
    assert.ok('defaultModelId' in body, 'defaultModelId is required');
    assertRegistryInvariants(body.models, body.defaultModelId);
  });

  test('the registry is ordered: custom first, then cheapest download first', () => {
    assertRegistryOrdering(body.models);
  });

  test('the registry agrees with the two fixed slots', () => {
    const ids = new Set(body.models.map((entry) => entry.id));

    assert.equal(
      ids.has(CUSTOM_ID),
      body.custom.available,
      'a "custom" registry entry exists exactly when the custom slot is available',
    );
    assert.equal(
      ids.has(FALLBACK_ID),
      body.fallback.available,
      'a "mobilenet_v2" registry entry exists exactly when the fallback slot is available',
    );

    // The old two-slot answer must stay derivable from the new list, because `model_kind` rows
    // and the existing UI are written in those terms.
    const expected = ids.has(CUSTOM_ID) ? 'custom' : ids.has(FALLBACK_ID) ? 'fallback' : 'none';
    assert.equal(body.active, expected, 'active must still follow custom > fallback > none');

    // A registry entry always carries the descriptor its slot block carries.
    const custom = body.models.find((entry) => entry.id === CUSTOM_ID);
    if (custom) {
      assert.equal(custom.kind, 'custom', 'the custom slot predicts waste ids directly');
      assert.equal(custom.modelUrl, ENGINES[0].modelUrl);
    }
    const fallback = body.models.find((entry) => entry.id === FALLBACK_ID);
    if (fallback) {
      assert.equal(fallback.kind, 'imagenet', 'the pretrained fallback predicts ImageNet classes');
      assert.equal(fallback.modelUrl, ENGINES[1].modelUrl);
    }
  });
});

describe('GET /api/models', () => {
  let request;
  let body;
  let headers;

  before(async () => {
    request = await makeRequest();
    const res = await request.get('/api/models');
    assert.equal(res.status, 200, `expected 200, got ${res.status}`);
    body = res.body;
    headers = res.headers;
  });

  test('returns the registry, its default and the active slot', () => {
    assertRegistryInvariants(body.models, body.defaultModelId);
    assertRegistryOrdering(body.models);
    assert.ok(
      ['custom', 'fallback', 'none'].includes(body.active),
      `active must be "custom" | "fallback" | "none", got "${body.active}"`,
    );
  });

  test('is the same registry the status endpoint publishes', async () => {
    const status = await request.get('/api/model/status');
    assert.equal(status.status, 200);
    assert.deepEqual(body.models, status.body.models, 'both endpoints publish one registry');
    assert.equal(body.defaultModelId, status.body.defaultModelId);
    assert.equal(body.active, status.body.active);
  });

  test('carries no slot-compatibility keys — the picker only needs the list', () => {
    assert.deepEqual(
      Object.keys(body).sort(),
      ['active', 'defaultModelId', 'models'],
      'the picker payload is exactly { models, defaultModelId, active }',
    );
  });

  test('is never cached', () => {
    assert.match(
      String(headers['cache-control']),
      /no-store/,
      `the model list must not be cached, got "${headers['cache-control']}"`,
    );
  });
});

describe('GET /api/models/:id', () => {
  let request;
  let models;

  before(async () => {
    request = await makeRequest();
    const res = await request.get('/api/models');
    assert.equal(res.status, 200);
    models = res.body.models;
  });

  after(async () => {
    await closeDatabase();
  });

  test('returns one installed model, byte for byte as the list has it', async (t) => {
    if (models.length === 0) {
      t.skip('no model is installed in this environment');
      return;
    }
    for (const listed of models) {
      const res = await request.get(`/api/models/${listed.id}`);
      assert.equal(res.status, 200, `expected 200 for "${listed.id}", got ${res.status}`);
      assertRegistryEntry(res.body.model);
      assert.deepEqual(res.body.model, listed, 'the single-model view must match the list entry');
      assert.match(
        String(res.headers['cache-control']),
        /no-store/,
        'a single model must not be cached either',
      );
    }
  });

  test('404s on an id that is well-formed but not installed', async () => {
    const res = await request.get('/api/models/definitely-not-installed-9f3a');
    const error = assertErrorEnvelope(res, 404, 'not_found');
    assert.match(error.message, /definitely-not-installed-9f3a/, 'the message names the id');
  });

  test('400s on a path-traversal attempt rather than touching the filesystem', async () => {
    // "..%2Fetc" arrives as the single path segment "../etc" once express decodes it.
    const res = await request.get('/api/models/..%2Fetc');
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('400s on an id outside the allowed alphabet', async () => {
    // A literal space is not a legal request target, so it travels percent-encoded.
    const res = await request.get('/api/models/a%20b');
    assertErrorEnvelope(res, 400, 'bad_request');
  });

  test('400s on an id longer than a directory name has any business being', async () => {
    const res = await request.get(`/api/models/${'m'.repeat(65)}`);
    assertErrorEnvelope(res, 400, 'bad_request');
  });
});

// Pure function, no database or HTTP: safe to run after the suites above have closed theirs.
describe('defaultMetadata', () => {
  test('does not guess class names for the custom slot', async () => {
    const { defaultMetadata } = await import('../src/services/modelService.js');
    const meta = defaultMetadata(CUSTOM_ID);

    // ml/train.py orders its softmax alphabetically (cardboard, ewaste, glass, ...) while
    // the waste taxonomy is ordered semantically (plastic, paper, cardboard, ...). Both are
    // length 10, so handing the taxonomy out here as if it were authoritative would pass
    // every downstream guard and mislabel every prediction in silence. The synthesised
    // descriptor must say "unknown" and let the frontend refuse to load.
    assert.equal(meta.classes, null, 'synthesised custom metadata must not invent class names');
    assert.equal(meta.classCount, null, 'an unknown class list has an unknown length');
    assert.equal(meta.labelKind, 'waste');
  });

  test('still describes the custom slot completely enough to be selectable', async () => {
    const { defaultMetadata } = await import('../src/services/modelService.js');
    const meta = defaultMetadata(CUSTOM_ID);

    assert.equal(meta.inputSize, 224);
    assert.equal(meta.outputActivation, 'softmax');
    assert.equal(meta.classOffset, 0);
    assert.match(meta.notes, /missing or unreadable/);
  });
});
