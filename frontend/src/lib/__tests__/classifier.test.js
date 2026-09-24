import { describe, expect, it, vi } from 'vitest';

// The aggregation under test is pure: no tensors, no DOM. Stubbing tfjs and the
// API client keeps this suite fast and independent of WebGL availability in the
// jsdom environment.
vi.mock('@tensorflow/tfjs', () => ({
  Tensor: class Tensor {},
  tidy: (fn) => fn(),
  softmax: () => {
    throw new Error('tfjs is stubbed in this suite');
  },
  zeros: () => {
    throw new Error('tfjs is stubbed in this suite');
  },
  image: { resizeBilinear: () => undefined },
  browser: { fromPixels: () => undefined },
  setBackend: async () => true,
  ready: async () => undefined,
  getBackend: () => 'cpu',
  loadGraphModel: async () => undefined,
  loadLayersModel: async () => undefined,
  disposeVariables: () => undefined,
}));

vi.mock('../api.js', () => ({
  getModelStatus: vi.fn(async () => ({
    custom: { available: false, modelUrl: null, metadataUrl: null, metadata: null },
    fallback: { available: false, modelUrl: null, metadataUrl: null, metadata: null },
    active: 'none',
    searchedPaths: [],
  })),
}));

const { IMAGENET_CLASSES } = await import('../imagenetClasses.js');
const {
  LOW_CONFIDENCE_THRESHOLD,
  aggregateCustomPredictions,
  aggregateImagenetPredictions,
  clampRegion,
  cropBoxFor,
  deriveCustomClasses,
  regionBatchSize,
  resolveWasteCategory,
} = await import('../classifier.js');
const { buildIndexMap } = await import('../imagenetWasteMap.js');

const INDEX_MAP = buildIndexMap(IMAGENET_CLASSES);

function idx(name) {
  const i = IMAGENET_CLASSES.indexOf(name);
  if (i < 0) throw new Error(`"${name}" is not an ImageNet class — fix the test`);
  return i;
}

/** A 1000-wide distribution built from { className: probability } pairs. */
function distribution(pairs) {
  const probs = new Float32Array(IMAGENET_CLASSES.length);
  for (const [name, value] of Object.entries(pairs)) probs[idx(name)] = value;
  return probs;
}

function aggregate(probs, options = {}) {
  return aggregateImagenetPredictions(probs, { indexMap: INDEX_MAP, ...options });
}

describe('aggregateImagenetPredictions — single confident class', () => {
  const result = aggregate(distribution({ 'water bottle': 1 }));

  it('puts all the mass on plastic', () => {
    expect(result.predictions[0]).toMatchObject({ category: 'plastic', confidence: 1 });
  });

  it('labels it with the strongest contributing ImageNet name', () => {
    expect(result.predictions[0].label).toBe('Plastic — water bottle');
  });

  it('reports no unmatched mass and normal confidence', () => {
    expect(result.unmatchedMass).toBeCloseTo(0, 6);
    expect(result.lowConfidence).toBe(false);
  });

  it('drops the synonym tail from the label', () => {
    const soda = aggregate(distribution({ 'pop bottle, soda bottle': 1 }));
    expect(soda.predictions[0].label).toBe('Plastic — pop bottle');
  });
});

describe('aggregateImagenetPredictions — mixed distribution', () => {
  const result = aggregate(
    distribution({
      'water bottle': 0.4,
      banana: 0.3,
      'beer bottle': 0.2,
      volcano: 0.1,
    }),
  );

  it('orders categories by descending absolute mass', () => {
    expect(result.predictions.map((p) => p.category)).toEqual(['plastic', 'organic', 'glass']);
  });

  it('keeps confidences absolute rather than normalising them to 1', () => {
    expect(result.predictions[0].confidence).toBeCloseTo(0.4, 6);
    expect(result.predictions[1].confidence).toBeCloseTo(0.3, 6);
    expect(result.predictions[2].confidence).toBeCloseTo(0.2, 6);
    const sum = result.predictions.reduce((acc, p) => acc + p.confidence, 0);
    expect(sum).toBeCloseTo(0.9, 6);
    expect(sum).toBeLessThan(1);
  });

  it('routes the mass of unmapped classes into unmatchedMass', () => {
    expect(result.unmatchedMass).toBeCloseTo(0.1, 6);
  });

  it('reports shares relative to the matched mass only', () => {
    expect(result.predictions[0].share).toBeCloseTo(0.4 / 0.9, 6);
  });

  it('honours topK', () => {
    const top1 = aggregate(
      distribution({ 'water bottle': 0.4, banana: 0.3, 'beer bottle': 0.2 }),
      { topK: 1 },
    );
    expect(top1.predictions).toHaveLength(1);
    expect(top1.predictions[0].category).toBe('plastic');
  });

  it('returns the top-10 raw ImageNet rows, strongest first', () => {
    expect(result.rawLabels).toHaveLength(10);
    expect(result.rawLabels[0]).toEqual({
      label: 'water bottle',
      confidence: expect.closeTo(0.4, 6),
      index: idx('water bottle'),
    });
    expect(result.rawLabels[1].label).toBe('banana');
    expect(result.rawLabels[2].label).toBe('beer bottle');
    expect(result.rawLabels[3].label).toBe('volcano');
    const confidences = result.rawLabels.map((r) => r.confidence);
    expect([...confidences].sort((a, b) => b - a)).toEqual(confidences);
  });
});

describe('aggregateImagenetPredictions — nothing recognisable', () => {
  const result = aggregate(
    distribution({ volcano: 0.5, 'coral reef': 0.3, 'tench, Tinca tinca': 0.2 }),
  );

  it('falls back to a single "Unrecognised item" prediction', () => {
    expect(result.predictions).toEqual([
      { category: 'trash', label: 'Unrecognised item', confidence: 0, share: 0 },
    ]);
  });

  it('flags low confidence and accounts for all the mass as unmatched', () => {
    expect(result.lowConfidence).toBe(true);
    expect(result.unmatchedMass).toBeCloseTo(1, 6);
    expect(result.matchedMass).toBe(0);
  });

  it('still reports the raw labels so the user can see what the model saw', () => {
    expect(result.rawLabels[0].label).toBe('volcano');
  });
});

describe('aggregateImagenetPredictions — weights', () => {
  it('scales a class by its weight', () => {
    // "can opener" implies metal is nearby, but weakly: weight 0.3.
    const result = aggregate(distribution({ 'can opener, tin opener': 1 }));
    expect(result.predictions[0].category).toBe('metal');
    expect(result.predictions[0].confidence).toBeCloseTo(0.3, 6);
    expect(result.unmatchedMass).toBeCloseTo(0, 6);
  });

  it('lets a weighted-down class lose to a lighter but certain one', () => {
    const result = aggregate(
      distribution({ 'can opener, tin opener': 0.6, 'wine bottle': 0.4 }),
    );
    expect(result.predictions[0].category).toBe('glass');
    expect(result.predictions[0].confidence).toBeCloseTo(0.4, 6);
    expect(result.predictions[1].confidence).toBeCloseTo(0.18, 6);
  });

  it('sums every contribution to a category and labels it with the strongest', () => {
    const result = aggregate(distribution({ 'water bottle': 0.2, 'plastic bag': 0.35 }));
    expect(result.predictions[0].confidence).toBeCloseTo(0.55, 6);
    expect(result.predictions[0].label).toBe('Plastic — plastic bag');
  });

  it('defaults a weightless map entry to weight 1', () => {
    const indexMap = new Map([[0, { category: 'glass' }]]);
    const result = aggregateImagenetPredictions([0.42], { indexMap, classNames: ['jar'] });
    expect(result.predictions[0].confidence).toBeCloseTo(0.42, 6);
    expect(result.predictions[0].label).toBe('Glass — jar');
  });
});

describe('aggregateImagenetPredictions — low confidence threshold', () => {
  it('is not low at exactly the threshold', () => {
    const result = aggregate(distribution({ 'water bottle': LOW_CONFIDENCE_THRESHOLD }));
    expect(result.predictions[0].confidence).toBeCloseTo(0.2, 6);
    expect(result.lowConfidence).toBe(false);
  });

  it('is low just below it', () => {
    const result = aggregate(distribution({ 'water bottle': 0.199 }));
    expect(result.lowConfidence).toBe(true);
  });
});

describe('aggregateImagenetPredictions — determinism', () => {
  const classNames = ['alpha', 'beta'];
  const indexMap = new Map([
    [0, { category: 'glass', weight: 1 }],
    [1, { category: 'plastic', weight: 1 }],
  ]);

  it('breaks an exact tie in canonical category order, every time', () => {
    for (let i = 0; i < 5; i += 1) {
      const result = aggregateImagenetPredictions([0.5, 0.5], { indexMap, classNames });
      // plastic precedes glass in the taxonomy, so it wins a dead heat.
      expect(result.predictions.map((p) => p.category)).toEqual(['plastic', 'glass']);
    }
  });

  it('breaks a raw-label tie by ascending index', () => {
    const result = aggregateImagenetPredictions([0.5, 0.5], { indexMap, classNames });
    expect(result.rawLabels.map((r) => r.index)).toEqual([0, 1]);
  });

  it('produces identical output for identical input', () => {
    const probs = distribution({ 'water bottle': 0.3, banana: 0.3, 'beer bottle': 0.3 });
    expect(aggregate(probs)).toEqual(aggregate(probs));
  });
});

describe('aggregateImagenetPredictions — input validation', () => {
  it('rejects a non array-like', () => {
    expect(() => aggregateImagenetPredictions(null, { indexMap: INDEX_MAP })).toThrow(TypeError);
  });

  it('treats NaN entries as zero instead of poisoning every score', () => {
    const probs = distribution({ 'water bottle': 0.5 });
    probs[idx('volcano')] = Number.NaN;
    const result = aggregate(probs);
    expect(result.predictions[0].confidence).toBeCloseTo(0.5, 6);
    expect(Number.isFinite(result.unmatchedMass)).toBe(true);
  });
});

describe('custom-model aggregation', () => {
  const classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'];

  it('maps class indices straight onto waste ids', () => {
    const result = aggregateCustomPredictions([0.05, 0.02, 0.03, 0.1, 0.75, 0.05], {
      classes,
      topK: 3,
    });
    expect(result.predictions.map((p) => p.category)).toEqual(['plastic', 'paper', 'cardboard']);
    expect(result.predictions[0].label).toBe('Plastic');
    expect(result.predictions[0].confidence).toBeCloseTo(0.75, 6);
    expect(result.rawLabels).toBeNull();
    expect(result.unmatchedMass).toBe(0);
  });

  it('flags a diffuse softmax as low confidence', () => {
    const flat = new Array(6).fill(1 / 6);
    expect(aggregateCustomPredictions(flat, { classes }).lowConfidence).toBe(true);
  });

  it('normalises a hyphenated id, and keeps an alias visible in the label', () => {
    const result = aggregateCustomPredictions([0.9, 0.1], {
      classes: ['e-waste', 'biological'],
      topK: 2,
    });
    expect(result.predictions[0]).toMatchObject({ category: 'ewaste', label: 'Electronics' });
    expect(result.predictions[1]).toMatchObject({
      category: 'organic',
      label: 'Organic / Food — biological',
    });
  });
});

describe('resolveWasteCategory', () => {
  it.each([
    ['plastic', 'plastic'],
    ['Plastic', 'plastic'],
    ['e-waste', 'ewaste'],
    ['E Waste', 'ewaste'],
    ['electronics', 'ewaste'],
    ['food', 'organic'],
    ['biological', 'organic'],
    ['clothes', 'textile'],
    ['aluminium', 'metal'],
    ['batteries', 'hazardous'],
    ['landfill', 'trash'],
  ])('resolves %s to %s', (input, expected) => {
    expect(resolveWasteCategory(input).category).toBe(expected);
  });

  it('marks canonical ids as exact and everything else as inexact', () => {
    expect(resolveWasteCategory('glass')).toEqual({ category: 'glass', exact: true });
    expect(resolveWasteCategory('bottles').exact).toBe(false);
  });

  it('falls back to trash for anything unrecognised', () => {
    expect(resolveWasteCategory('sploorf').category).toBe('trash');
    expect(resolveWasteCategory('').category).toBe('trash');
    expect(resolveWasteCategory(undefined).category).toBe('trash');
  });
});

describe('deriveCustomClasses', () => {
  // ml/train.py orders its softmax alphabetically; WASTE_CATEGORY_IDS is ordered
  // semantically. Both are length 10, which is exactly why this has to refuse rather
  // than guess: a wrong guess passes every size check and mislabels every prediction.
  const TRAINED_ORDER = [
    'cardboard',
    'ewaste',
    'glass',
    'hazardous',
    'metal',
    'organic',
    'paper',
    'plastic',
    'textile',
    'trash',
  ];

  it('uses the metadata class list when it matches the output width', () => {
    expect(deriveCustomClasses({ classes: TRAINED_ORDER }, 10)).toEqual(TRAINED_ORDER);
  });

  it('refuses rather than guessing when metadata has no class list', () => {
    expect(() => deriveCustomClasses({}, 10)).toThrow(/no `classes` array/);
    expect(() => deriveCustomClasses({ classes: null }, 10)).toThrow(/no `classes` array/);
  });

  it('points at the recorded label order so a hand-written metadata.json can be fixed', () => {
    // Not a hardcoded list of ten: a model trained on a subset of the categories has a
    // shorter order, and dataset.json is the thing that actually records it.
    expect(() => deriveCustomClasses({}, 10)).toThrow(/ml\/dataset\/dataset\.json/);
  });

  it('accepts a model trained on a subset of the taxonomy', () => {
    // The dataset in use covers 7 of the 10 categories, so a 7-wide softmax with 7 names
    // is a legitimate model, not a mismatch to reject.
    const seven = ['ewaste', 'glass', 'hazardous', 'metal', 'organic', 'paper', 'plastic'];
    expect(deriveCustomClasses({ classes: seven }, 7)).toEqual(seven);
  });

  it('refuses a class list whose length disagrees with the output width', () => {
    expect(() => deriveCustomClasses({ classes: ['glass', 'metal'] }, 10)).toThrow(
      /lists 2 classes but the model outputs 10/,
    );
  });

  it('refuses a non-standard output width too', () => {
    expect(() => deriveCustomClasses({}, 4)).toThrow(/outputs 4 classes/);
  });
});

describe('clampRegion', () => {
  it('leaves a region that fits untouched', () => {
    expect(clampRegion({ x: 10, y: 20, width: 30, height: 40 }, 100, 100)).toEqual({
      x: 10,
      y: 20,
      width: 30,
      height: 40,
    });
  });

  it('trims a region overhanging the frame so tf.slice cannot throw', () => {
    // Frame is 48 high, 64 wide.
    expect(clampRegion({ x: 50, y: 40, width: 30, height: 30 }, 48, 64)).toEqual({
      x: 50,
      y: 40,
      width: 14,
      height: 8,
    });
    expect(clampRegion({ x: -5, y: -5, width: 10, height: 10 }, 48, 64)).toEqual({
      x: 0,
      y: 0,
      width: 10,
      height: 10,
    });
  });

  it('never produces an empty slice, even from garbage', () => {
    for (const region of [null, {}, { x: 999, y: 999, width: 0, height: -3 }, { x: 'a', width: NaN }]) {
      const r = clampRegion(region, 48, 64);
      expect(r.width).toBeGreaterThanOrEqual(1);
      expect(r.height).toBeGreaterThanOrEqual(1);
      expect(r.x + r.width).toBeLessThanOrEqual(64);
      expect(r.y + r.height).toBeLessThanOrEqual(48);
    }
  });
});

describe('regionBatchSize', () => {
  it('pads the batch up to a small fixed set of sizes', () => {
    expect([1, 2, 3, 4, 5, 6].map(regionBatchSize)).toEqual([1, 2, 4, 4, 6, 6]);
  });

  it('runs an unusually large batch as it is rather than truncating it', () => {
    expect(regionBatchSize(9)).toBe(9);
  });
});

describe('cropBoxFor', () => {
  it('maps the first and last pixel of the region onto the box edges', () => {
    // 101 x 201 frame: pixel p sits at p / 100 vertically and p / 200 horizontally.
    const [y1, x1, y2, x2] = cropBoxFor({ x: 20, y: 10, width: 41, height: 31 }, 101, 201);
    expect(y1).toBeCloseTo(0.1);
    expect(x1).toBeCloseTo(0.1);
    expect(y2).toBeCloseTo(0.4);
    expect(x2).toBeCloseTo(0.3);
  });

  it('covers the whole frame for a full-frame region', () => {
    expect(cropBoxFor({ x: 0, y: 0, width: 640, height: 480 }, 480, 640)).toEqual([0, 0, 1, 1]);
  });

  it('clamps an overhanging region first', () => {
    const [y1, x1, y2, x2] = cropBoxFor({ x: 600, y: -20, width: 100, height: 100 }, 480, 640);
    expect(y1).toBe(0);
    expect(x2).toBe(1);
    expect(x1).toBeGreaterThan(0.9);
    expect(y2).toBeLessThan(1);
  });
});
