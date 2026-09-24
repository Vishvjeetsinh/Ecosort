import { test, describe, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { CATEGORY_IDS, CATEGORY_COLORS, HEX_COLOR, DEFAULT_REGION_ID } from './helpers.js';

const DATA_DIR = path.join(import.meta.dirname, '..', 'src', 'data');
const CATEGORIES_FILE = path.join(DATA_DIR, 'waste-categories.json');
const RULES_FILE = path.join(DATA_DIR, 'recycling-rules.json');

function readJson(file) {
  const raw = fs.readFileSync(file, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (cause) {
    throw new Error(`${path.basename(file)} is not valid JSON: ${cause.message}`, { cause });
  }
}

/** The data files may wrap their payload; accept either layout, reject anything else. */
function toCategoryList(raw) {
  if (Array.isArray(raw)) return raw;
  if (Array.isArray(raw?.categories)) return raw.categories;
  throw new Error('waste-categories.json must be an array or { categories: [...] }');
}

/** Returns [{ id, meta, bins, categories }] for either the array or the keyed layout. */
function toRegions(raw) {
  const container = raw?.regions ?? raw;
  const entries = Array.isArray(container)
    ? container.map((region) => [region.id ?? region.region?.id, region])
    : Object.entries(container ?? {});

  if (entries.length === 0) throw new Error('recycling-rules.json defines no regions');

  return entries.map(([key, value]) => {
    const meta = value.region ?? value;
    const id = meta.id ?? key;
    assert.equal(typeof id, 'string', 'every region needs a string id');
    if (typeof key === 'string' && !Array.isArray(container)) {
      assert.equal(id, key, `region keyed as "${key}" declares id "${id}"`);
    }
    assert.ok(Array.isArray(value.bins), `region "${id}" must declare a bins array`);
    assert.ok(
      value.categories && typeof value.categories === 'object' && !Array.isArray(value.categories),
      `region "${id}" must declare a categories map`,
    );
    return { id, meta, bins: value.bins, categories: value.categories };
  });
}

describe('data files', () => {
  let categories;
  let rulesRaw;
  let regions;

  before(() => {
    assert.ok(fs.existsSync(DATA_DIR), `missing data directory ${DATA_DIR}`);

    const files = fs.readdirSync(DATA_DIR).filter((name) => name.endsWith('.json'));
    assert.ok(files.includes('waste-categories.json'), 'waste-categories.json is required');
    assert.ok(files.includes('recycling-rules.json'), 'recycling-rules.json is required');
    for (const name of files) readJson(path.join(DATA_DIR, name)); // every file must parse

    categories = toCategoryList(readJson(CATEGORIES_FILE));
    rulesRaw = readJson(RULES_FILE);
    regions = toRegions(rulesRaw);
  });

  test('waste-categories.json holds the ten canonical categories', () => {
    assert.equal(categories.length, 10, `expected 10 categories, got ${categories.length}`);

    const ids = categories.map((c) => c.id);
    assert.equal(new Set(ids).size, ids.length, 'category ids must be unique');
    assert.deepEqual(
      [...ids].sort(),
      [...CATEGORY_IDS].sort(),
      'category ids must be exactly the canonical set',
    );

    for (const category of categories) {
      for (const key of ['label', 'shortLabel', 'description', 'icon']) {
        assert.equal(typeof category[key], 'string', `${category.id}.${key} must be a string`);
        assert.ok(category[key].trim().length > 0, `${category.id}.${key} must not be blank`);
      }
      assert.ok(Array.isArray(category.examples) && category.examples.length > 0,
        `${category.id}.examples must be a non-empty array`);
      for (const example of category.examples) {
        assert.equal(typeof example, 'string');
        assert.ok(example.trim().length > 0, `${category.id} has a blank example`);
      }
      assert.match(category.colorHex, HEX_COLOR, `${category.id}.colorHex is not a hex colour`);
      assert.match(category.textColorHex, HEX_COLOR, `${category.id}.textColorHex is not a hex colour`);
      assert.equal(
        category.colorHex.toLowerCase(),
        CATEGORY_COLORS[category.id],
        `${category.id}.colorHex must match ARCHITECTURE.md section 3`,
      );
    }
  });

  test('recycling-rules.json declares at least seven regions and a resolvable default', () => {
    assert.ok(regions.length >= 7, `expected at least 7 regions, got ${regions.length}`);

    const ids = regions.map((r) => r.id);
    assert.equal(new Set(ids).size, ids.length, 'region ids must be unique');

    const declaredDefault = rulesRaw.defaultRegion ?? DEFAULT_REGION_ID;
    assert.ok(
      ids.includes(declaredDefault),
      `defaultRegion "${declaredDefault}" is not one of ${ids.join(', ')}`,
    );
    assert.ok(ids.includes(DEFAULT_REGION_ID), 'the us-generic region must exist');

    for (const region of regions) {
      for (const key of ['name', 'country', 'authority', 'updated']) {
        assert.equal(typeof region.meta[key], 'string', `region "${region.id}".${key}`);
        assert.ok(region.meta[key].trim().length > 0, `region "${region.id}".${key} is blank`);
      }
      assert.ok(
        region.meta.notes === undefined || typeof region.meta.notes === 'string',
        `region "${region.id}".notes must be a string when present`,
      );
      assert.match(
        region.meta.updated,
        /\d{4}/,
        `region "${region.id}".updated must carry a year, got "${region.meta.updated}"`,
      );
    }
  });

  test('every region covers all ten categories exactly once', () => {
    for (const region of regions) {
      const keys = Object.keys(region.categories);
      assert.deepEqual(
        [...keys].sort(),
        [...CATEGORY_IDS].sort(),
        `region "${region.id}" does not map exactly the ten canonical categories`,
      );
      // The loader derives categoryId from the key, so stating it is optional - but a stated
      // one that disagrees with its key is a genuine mistake.
      for (const [key, guidance] of Object.entries(region.categories)) {
        if (guidance.categoryId !== undefined) {
          assert.equal(
            guidance.categoryId,
            key,
            `region "${region.id}": guidance keyed "${key}" declares categoryId "${guidance.categoryId}"`,
          );
        }
      }
    }
  });

  test('every bin is well formed and only accepts canonical categories', () => {
    for (const region of regions) {
      assert.ok(region.bins.length > 0, `region "${region.id}" declares no bins`);

      const ids = region.bins.map((b) => b.id);
      assert.equal(new Set(ids).size, ids.length, `region "${region.id}" repeats a bin id`);

      for (const bin of region.bins) {
        for (const key of ['id', 'name', 'colorName', 'description']) {
          assert.equal(typeof bin[key], 'string', `${region.id}/${bin.id}.${key}`);
          assert.ok(bin[key].trim().length > 0, `${region.id}/${bin.id}.${key} is blank`);
        }
        assert.match(bin.colorHex, HEX_COLOR, `${region.id}/${bin.id}.colorHex`);
        assert.match(bin.textColorHex, HEX_COLOR, `${region.id}/${bin.id}.textColorHex`);
        assert.ok(Array.isArray(bin.accepts), `${region.id}/${bin.id}.accepts must be an array`);
        assert.ok(bin.accepts.length > 0, `${region.id}/${bin.id} accepts nothing`);
        assert.equal(
          new Set(bin.accepts).size,
          bin.accepts.length,
          `${region.id}/${bin.id}.accepts repeats a category`,
        );
        for (const id of bin.accepts) {
          assert.ok(
            CATEGORY_IDS.includes(id),
            `${region.id}/${bin.id} accepts unknown category "${id}"`,
          );
        }
      }
    }
  });

  test('a regions bins partition the ten categories', () => {
    for (const region of regions) {
      const seen = region.bins.flatMap((bin) => bin.accepts);
      assert.equal(
        new Set(seen).size,
        seen.length,
        `region "${region.id}" routes a category to more than one bin`,
      );
      assert.deepEqual(
        [...seen].sort(),
        [...CATEGORY_IDS].sort(),
        `region "${region.id}" does not route every category to exactly one bin`,
      );
    }
  });

  test('every guidance resolves to a bin that accepts it', () => {
    for (const region of regions) {
      const binsById = new Map(region.bins.map((bin) => [bin.id, bin]));
      for (const [categoryId, guidance] of Object.entries(region.categories)) {
        const bin = binsById.get(guidance.binId);
        assert.ok(
          bin,
          `region "${region.id}": ${categoryId} points at unknown bin "${guidance.binId}"`,
        );
        assert.ok(
          bin.accepts.includes(categoryId),
          `region "${region.id}": bin "${bin.id}" does not accept "${categoryId}"`,
        );
      }
    }
  });

  // A guidance may inherit these from its bin by leaving them out; what it must never do is
  // carry a copy that has drifted, because the frontend paints the card from the guidance.
  test('denormalised bin fields on a guidance are never stale', () => {
    const MIRRORED = [
      ['binName', 'name'],
      ['colorName', 'colorName'],
      ['colorHex', 'colorHex'],
      ['textColorHex', 'textColorHex'],
    ];

    for (const region of regions) {
      const binsById = new Map(region.bins.map((bin) => [bin.id, bin]));
      for (const [categoryId, guidance] of Object.entries(region.categories)) {
        const bin = binsById.get(guidance.binId);
        const where = `region "${region.id}", category "${categoryId}"`;
        for (const [onGuidance, onBin] of MIRRORED) {
          if (guidance[onGuidance] === undefined) continue;
          assert.equal(
            guidance[onGuidance],
            bin[onBin],
            `${where}: ${onGuidance} "${guidance[onGuidance]}" does not match bin "${bin.id}" (${bin[onBin]})`,
          );
        }
      }
    }
  });

  test('every guidance carries usable advice', () => {
    for (const region of regions) {
      for (const [categoryId, guidance] of Object.entries(region.categories)) {
        const where = `region "${region.id}", category "${categoryId}"`;

        assert.equal(typeof guidance.recyclable, 'boolean', `${where}: recyclable must be boolean`);
        assert.equal(typeof guidance.disposal, 'string', `${where}: disposal must be a string`);
        assert.ok(guidance.disposal.trim().length > 0, `${where}: disposal is blank`);
        // notes and dropOff are the two fields the loader defaults, so absence is legal.
        assert.ok(
          guidance.notes === undefined || typeof guidance.notes === 'string',
          `${where}: notes must be a string when present`,
        );
        assert.ok(
          guidance.dropOff === undefined || guidance.dropOff === null || typeof guidance.dropOff === 'string',
          `${where}: dropOff must be a string or null`,
        );

        for (const key of ['prepSteps', 'acceptedExamples', 'rejectedExamples']) {
          assert.ok(Array.isArray(guidance[key]), `${where}: ${key} must be an array`);
          for (const entry of guidance[key]) {
            assert.equal(typeof entry, 'string', `${where}: ${key} holds a non-string`);
            assert.ok(entry.trim().length > 0, `${where}: ${key} holds a blank string`);
          }
        }
        assert.ok(guidance.prepSteps.length > 0, `${where}: needs at least one prep step`);
        assert.ok(
          guidance.acceptedExamples.length > 0,
          `${where}: needs at least one accepted example`,
        );
      }

      assert.equal(
        region.categories.trash.recyclable,
        false,
        `region "${region.id}": general waste must not be marked recyclable`,
      );
    }
  });

  test('every colour in the data set is a six-digit hex value', () => {
    const colours = [];
    for (const category of categories) colours.push(category.colorHex, category.textColorHex);
    for (const region of regions) {
      for (const bin of region.bins) colours.push(bin.colorHex, bin.textColorHex);
      for (const guidance of Object.values(region.categories)) {
        // Only the copies that were actually written down; the rest come from the bin.
        for (const key of ['colorHex', 'textColorHex']) {
          if (guidance[key] !== undefined) colours.push(guidance[key]);
        }
      }
    }

    assert.ok(colours.length > 0, 'the data set defines no colours at all');
    for (const colour of colours) {
      assert.match(colour, HEX_COLOR, `"${colour}" is not a six-digit hex colour`);
    }
  });
});
