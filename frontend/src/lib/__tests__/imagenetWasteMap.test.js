import { describe, expect, it } from 'vitest';

import { IMAGENET_CLASSES } from '../imagenetClasses.js';
import {
  IMAGENET_WASTE_MAP,
  MAPPED_CATEGORY_IDS,
  WASTE_CATEGORY_IDS,
  WASTE_CATEGORY_LABELS,
  buildIndexMap,
  lookupImagenetClass,
  mapCoverage,
} from '../imagenetWasteMap.js';

const CANONICAL = new Set(WASTE_CATEGORY_IDS);
const CLASS_SET = new Set(IMAGENET_CLASSES);
const KEYS = Object.keys(IMAGENET_WASTE_MAP);

describe('IMAGENET_CLASSES', () => {
  it('has exactly the 1000 ImageNet-1k entries', () => {
    expect(IMAGENET_CLASSES).toHaveLength(1000);
    expect(IMAGENET_CLASSES[0]).toBe('tench, Tinca tinca');
    expect(IMAGENET_CLASSES[999]).toBe('toilet tissue, toilet paper, bathroom tissue');
  });

  // ImageNet-1k genuinely reuses one label: "crane" is both the bird (134) and the
  // construction machine (517). That is a property of the real label set, not a bug in
  // the generated file, so pin it exactly rather than asserting global uniqueness --
  // and make sure no NEW duplicate ever sneaks in.
  it('has exactly the one known upstream duplicate label', () => {
    const seen = new Map();
    IMAGENET_CLASSES.forEach((name, index) => {
      if (!seen.has(name)) seen.set(name, []);
      seen.get(name).push(index);
    });
    const duplicates = [...seen.entries()].filter(([, indices]) => indices.length > 1);
    expect(duplicates).toEqual([['crane', [134, 517]]]);
    expect(CLASS_SET.size).toBe(999);
  });

  // Because the waste map is keyed by NAME, mapping an ambiguous name would tag both
  // indices. "crane" means neither waste stream, so it must stay unmapped.
  it('leaves the ambiguous duplicate label unmapped', () => {
    expect(IMAGENET_WASTE_MAP.crane).toBeUndefined();
  });
});

describe('IMAGENET_WASTE_MAP keys', () => {
  // The single most important assertion in the suite: a typo'd key is silently
  // dead weight at runtime, because the lookup is by exact class name.
  it('every key is an exact IMAGENET_CLASSES entry', () => {
    const offenders = KEYS.filter((key) => !CLASS_SET.has(key));
    expect(
      offenders,
      `These keys are not exact ImageNet class names:\n${offenders
        .map((k) => `  ${JSON.stringify(k)}`)
        .join('\n')}`,
    ).toEqual([]);
  });

  it('maps at least 250 classes', () => {
    expect(KEYS.length).toBeGreaterThanOrEqual(250);
  });
});

describe('IMAGENET_WASTE_MAP values', () => {
  it('only uses the 10 canonical category ids', () => {
    const offenders = KEYS.filter((key) => !CANONICAL.has(IMAGENET_WASTE_MAP[key].category));
    expect(
      offenders,
      `Non-canonical categories: ${offenders
        .map((k) => `${k} -> ${IMAGENET_WASTE_MAP[k].category}`)
        .join(', ')}`,
    ).toEqual([]);
  });

  it('gives every entry a weight in (0, 1]', () => {
    const offenders = KEYS.filter((key) => {
      const { weight } = IMAGENET_WASTE_MAP[key];
      return typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0 || weight > 1;
    });
    expect(
      offenders,
      `Bad weights: ${offenders.map((k) => `${k} -> ${IMAGENET_WASTE_MAP[k].weight}`).join(', ')}`,
    ).toEqual([]);
  });

  it('has a human label for every canonical id', () => {
    for (const id of WASTE_CATEGORY_IDS) {
      expect(typeof WASTE_CATEGORY_LABELS[id]).toBe('string');
      expect(WASTE_CATEGORY_LABELS[id].length).toBeGreaterThan(0);
    }
  });
});

describe('known-correct mappings', () => {
  const EXPECTED = [
    ['water bottle', 'plastic'],
    ['pop bottle, soda bottle', 'plastic'],
    ['plastic bag', 'plastic'],
    ['banana', 'organic'],
    ['pizza, pizza pie', 'organic'],
    ['beer bottle', 'glass'],
    ['wine bottle', 'glass'],
    ['beer glass', 'glass'],
    ['carton', 'cardboard'],
    ['envelope', 'paper'],
    ['cellular telephone, cellular phone, cellphone, cell, mobile phone', 'ewaste'],
    ['laptop, laptop computer', 'ewaste'],
    ['syringe', 'hazardous'],
    ['hair spray', 'hazardous'],
    ['jersey, T-shirt, tee shirt', 'textile'],
    ['running shoe', 'textile'],
    ['frying pan, frypan, skillet', 'metal'],
    ['diaper, nappy, napkin', 'trash'],
  ];

  it.each(EXPECTED)('%s is %s', (name, category) => {
    const entry = lookupImagenetClass(name);
    expect(entry, `${name} is not mapped at all`).not.toBeNull();
    expect(entry.category).toBe(category);
  });

  it('distinguishes plastic bottles from glass bottles', () => {
    expect(lookupImagenetClass('pop bottle, soda bottle').category).toBe('plastic');
    expect(lookupImagenetClass('beer bottle').category).toBe('glass');
  });

  it('gives weak associations a low weight', () => {
    expect(lookupImagenetClass('can opener, tin opener').weight).toBeLessThanOrEqual(0.4);
    expect(lookupImagenetClass('water bottle').weight).toBe(1);
  });
});

describe('deliberately unmapped classes', () => {
  const NOT_WASTE = [
    'tench, Tinca tinca',
    'goldfish, Carassius auratus',
    'volcano',
    'alp',
    'coral reef',
    'airliner',
    'sports car, sport car',
    'church, church building',
    'scuba diver',
    'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
  ];

  it.each(NOT_WASTE)('%s stays unmapped so its mass lands in unmatchedMass', (name) => {
    expect(CLASS_SET.has(name), `${name} is not an ImageNet class — fix the test`).toBe(true);
    expect(lookupImagenetClass(name)).toBeNull();
  });
});

describe('lookupImagenetClass', () => {
  it('returns null for unknown names and non-strings', () => {
    expect(lookupImagenetClass('definitely not a class')).toBeNull();
    expect(lookupImagenetClass(undefined)).toBeNull();
    expect(lookupImagenetClass(42)).toBeNull();
  });

  it('does not leak Object.prototype members', () => {
    expect(lookupImagenetClass('constructor')).toBeNull();
    expect(lookupImagenetClass('toString')).toBeNull();
    expect(lookupImagenetClass('__proto__')).toBeNull();
  });
});

describe('buildIndexMap', () => {
  const indexMap = buildIndexMap(IMAGENET_CLASSES);

  it('has one entry per mapped class', () => {
    expect(indexMap.size).toBe(KEYS.length);
  });

  it('keys by class index and carries the class name', () => {
    const i = IMAGENET_CLASSES.indexOf('water bottle');
    expect(i).toBeGreaterThanOrEqual(0);
    expect(indexMap.get(i)).toEqual({ category: 'plastic', weight: 1, name: 'water bottle' });
  });

  it('omits unmapped indices', () => {
    expect(indexMap.get(IMAGENET_CLASSES.indexOf('volcano'))).toBeUndefined();
  });

  it('rejects a non-array argument', () => {
    expect(() => buildIndexMap('nope')).toThrow(TypeError);
  });
});

describe('mapCoverage', () => {
  const coverage = mapCoverage();

  it('reports the mapped/total counts', () => {
    expect(coverage.total).toBe(1000);
    expect(coverage.mapped).toBe(KEYS.length);
    expect(coverage.mapped).toBeLessThan(coverage.total);
  });

  it('breaks the count down by category, summing to the total mapped', () => {
    const sum = Object.values(coverage.byCategory).reduce((acc, n) => acc + n, 0);
    expect(sum).toBe(coverage.mapped);
    for (const id of WASTE_CATEGORY_IDS) {
      expect(coverage.byCategory[id], `${id} missing from byCategory`).toBeTypeOf('number');
    }
  });

  it('covers every canonical category at least once', () => {
    expect([...MAPPED_CATEGORY_IDS].sort()).toEqual([...WASTE_CATEGORY_IDS].sort());
  });
});
