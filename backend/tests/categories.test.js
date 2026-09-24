import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  makeRequest,
  resetDatabase,
  closeDatabase,
  CATEGORY_IDS,
  CATEGORY_COLORS,
  HEX_COLOR,
} from './helpers.js';

const REQUIRED_KEYS = [
  'id',
  'label',
  'shortLabel',
  'description',
  'icon',
  'colorHex',
  'textColorHex',
  'examples',
];

describe('GET /api/categories', () => {
  let request;
  let categories;

  before(async () => {
    request = await makeRequest();
  });

  beforeEach(async () => {
    await resetDatabase();
    const res = await request.get('/api/categories');
    assert.equal(res.status, 200, `expected 200, got ${res.status}`);
    categories = res.body.categories;
  });

  after(async () => {
    await closeDatabase();
  });

  test('returns exactly the ten canonical categories, in order', () => {
    assert.ok(Array.isArray(categories), 'categories must be an array');
    assert.equal(categories.length, 10, 'the taxonomy has exactly 10 categories');
    assert.deepEqual(
      categories.map((c) => c.id),
      [...CATEGORY_IDS],
      'ids must match the canonical list, in the canonical order',
    );
  });

  test('every category carries the required keys with usable values', () => {
    for (const category of categories) {
      for (const key of REQUIRED_KEYS) {
        assert.ok(key in category, `category "${category.id}" is missing "${key}"`);
      }

      for (const key of ['id', 'label', 'shortLabel', 'description', 'icon']) {
        assert.equal(typeof category[key], 'string', `${category.id}.${key} must be a string`);
        assert.ok(category[key].length > 0, `${category.id}.${key} must not be empty`);
      }

      assert.ok(
        category.shortLabel.length <= category.label.length + 1,
        `${category.id}.shortLabel should be no longer than its label`,
      );

      assert.ok(Array.isArray(category.examples), `${category.id}.examples must be an array`);
      assert.ok(category.examples.length > 0, `${category.id}.examples must not be empty`);
      for (const example of category.examples) {
        assert.equal(typeof example, 'string');
        assert.ok(example.length > 0, `${category.id} has an empty example`);
      }
    }
  });

  test('colours are 6-digit hex and match the contract', () => {
    for (const category of categories) {
      assert.match(
        category.colorHex,
        HEX_COLOR,
        `${category.id}.colorHex "${category.colorHex}" is not a 6-digit hex colour`,
      );
      assert.match(
        category.textColorHex,
        HEX_COLOR,
        `${category.id}.textColorHex "${category.textColorHex}" is not a 6-digit hex colour`,
      );
      assert.equal(
        category.colorHex.toLowerCase(),
        CATEGORY_COLORS[category.id],
        `${category.id}.colorHex must be the colour fixed by ARCHITECTURE.md section 3`,
      );
      assert.notEqual(
        category.textColorHex.toLowerCase(),
        category.colorHex.toLowerCase(),
        `${category.id} needs a text colour distinct from its background`,
      );
    }
  });

  test('ids are unique and label/description text is not duplicated', () => {
    const ids = categories.map((c) => c.id);
    assert.equal(new Set(ids).size, ids.length, 'category ids must be unique');

    const labels = categories.map((c) => c.label.toLowerCase());
    assert.equal(new Set(labels).size, labels.length, 'category labels must be unique');

    const descriptions = categories.map((c) => c.description.toLowerCase());
    assert.equal(
      new Set(descriptions).size,
      descriptions.length,
      'each category needs its own description',
    );
  });

  test('is stable across calls', async () => {
    const again = await request.get('/api/categories');
    assert.equal(again.status, 200);
    assert.deepEqual(again.body.categories, categories);
  });
});
