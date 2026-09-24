import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { makeRequest, closeDatabase } from './helpers.js';

/**
 * The Live scan object detector lives at models/detectors/<id>/ (scripts/fetch-detector.mjs),
 * one level below the classifiers. These tests pin the two halves of that arrangement against
 * a throwaway models root, so they hold whatever happens to be installed on this machine:
 *   - the classifier registry must never offer the detector (it would fail its warm-up), and
 *   - the static mount must still serve it, because the browser loads it from /models/**.
 *
 * `src/config.js` snapshots the environment when it is first imported, which helpers.js defers
 * until makeRequest(); pointing both model roots at the fixture here is therefore early enough.
 */
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ecosort-detector-'));
process.env.MODELS_DIR = root;
process.env.BUNDLED_MODELS_DIR = root;

const DETECTOR_DIR = path.join(root, 'detectors', 'ssdlite_mobilenet_v2');
const SHARD = 'group1-shard1of1';

function writeModel(dir) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, 'model.json'),
    JSON.stringify({
      modelTopology: {},
      weightsManifest: [{ paths: [SHARD], weights: [{ name: 'w', shape: [2], dtype: 'float32' }] }],
    }),
  );
  fs.writeFileSync(path.join(dir, SHARD), Buffer.alloc(8));
}

describe('the object detector beside the classifier registry', () => {
  let request;

  before(async () => {
    writeModel(path.join(root, 'mobilenet_v2'));
    writeModel(DETECTOR_DIR);
    request = await makeRequest();
  });

  after(async () => {
    await closeDatabase();
    fs.rmSync(root, { recursive: true, force: true });
  });

  test('is never offered as a classifier', async () => {
    const res = await request.get('/api/models').expect(200);
    assert.deepEqual(
      res.body.models.map((entry) => entry.id),
      ['mobilenet_v2'],
      'only direct children of the models root holding a model.json are classifiers',
    );
  });

  test('is not reported among the searched classifier paths', async () => {
    const res = await request.get('/api/model/status').expect(200);
    for (const searched of res.body.searchedPaths) {
      assert.ok(!searched.includes(`${path.sep}detectors${path.sep}`), `unexpected ${searched}`);
    }
  });

  test('is still served from /models/detectors/ for the browser to load', async () => {
    const manifest = await request
      .get('/models/detectors/ssdlite_mobilenet_v2/model.json')
      .expect(200)
      .expect('Content-Type', /json/);
    assert.deepEqual(manifest.body.weightsManifest[0].paths, [SHARD]);

    await request
      .get(`/models/detectors/ssdlite_mobilenet_v2/${SHARD}`)
      .expect(200)
      .expect('Content-Type', 'application/octet-stream');
  });
});
