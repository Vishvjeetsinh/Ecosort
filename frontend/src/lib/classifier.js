/**
 * The EcoSort inference engine.
 *
 * Implements docs/ARCHITECTURE.md section 5 literally. Every model shares one
 * preprocessing path because all of them are contracted to take pixels in
 * [0, 1] (the pretrained graphs rescale to [-1, 1] internally, and ml/train.py
 * bakes a `Rescaling(2.0, -1.0)` layer in as its first layer):
 *
 *   fromPixels -> resizeBilinear(inputSize, alignCorners) -> /255 -> expandDims
 *
 * What differs is only the *postprocessing*, which is driven by metadata.json
 * (`outputActivation`, `classOffset`, `labelKind`) rather than by branching on
 * the model kind.
 *
 * Which model runs is a *selection*, not a branch (ARCHITECTURE 2.4):
 * `loadEngine({ modelId })` picks one entry out of the registry the backend
 * reports, and each entry brings its own `inputSize`. That is why a 299px
 * Inception slots in beside the 224px MobileNets without a single new code path
 * below — the only thing that changes is which URL gets loaded.
 */

import * as tf from '@tensorflow/tfjs';

import { getModelStatus } from './api.js';
import { IMAGENET_CLASSES } from './imagenetClasses.js';
import { buildIndexMap, WASTE_CATEGORY_IDS, WASTE_CATEGORY_LABELS } from './imagenetWasteMap.js';

/** ARCHITECTURE 5: a prediction below this is flagged as low confidence. */
export const LOW_CONFIDENCE_THRESHOLD = 0.2;

/** How many raw ImageNet rows the fallback engine reports for the disclosure UI. */
export const RAW_LABEL_COUNT = 10;

const DEFAULT_INPUT_SIZE = 224;
const MAX_TOP_K = 10;

export class EngineUnavailableError extends Error {
  constructor(message, details = null) {
    super(message);
    this.name = 'EngineUnavailableError';
    this.details = details;
  }
}

/* ------------------------------------------------------------------ *
 * Pure helpers (no tfjs, no DOM) — unit tested directly.
 * ------------------------------------------------------------------ */

let cachedIndexMap = null;

function defaultIndexMap() {
  if (!cachedIndexMap) cachedIndexMap = buildIndexMap(IMAGENET_CLASSES);
  return cachedIndexMap;
}

/** "pop bottle, soda bottle" -> "pop bottle" — synonyms are noise in a label. */
function shortName(name) {
  if (typeof name !== 'string') return '';
  const [first] = name.split(',');
  return first.trim();
}

function clampTopK(topK) {
  const n = Number.isFinite(topK) ? Math.floor(topK) : 3;
  return Math.min(MAX_TOP_K, Math.max(1, n));
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Aggregate a 1000-way ImageNet distribution into waste categories.
 *
 * Confidences stay ABSOLUTE probability mass (ARCHITECTURE 5 step 4): they are
 * never renormalised to sum to 1, because "37% of the picture looks like
 * plastic" is an honest answer and "100% plastic" would not be.
 *
 * @param {ArrayLike<number>} probs 1000 probabilities, background class already removed
 * @param {{topK?: number, indexMap?: Map<number, {category: string, weight: number, name?: string}>, classNames?: string[]}} [options]
 * @returns {{predictions: Array<{category: string, label: string, confidence: number, share: number}>,
 *            rawLabels: Array<{label: string, confidence: number, index: number}>,
 *            unmatchedMass: number, matchedMass: number, lowConfidence: boolean}}
 */
export function aggregateImagenetPredictions(probs, options = {}) {
  if (!probs || typeof probs.length !== 'number') {
    throw new TypeError('aggregateImagenetPredictions expects an array-like of probabilities');
  }

  const { topK = 3, classNames = IMAGENET_CLASSES } = options;
  const indexMap = options.indexMap ?? defaultIndexMap();
  const k = clampTopK(topK);

  const scores = new Map();
  let unmatchedMass = 0;
  let matchedMass = 0;

  for (let i = 0; i < probs.length; i += 1) {
    const p = toNumber(probs[i]);
    const entry = indexMap.get(i);

    if (!entry) {
      unmatchedMass += p;
      continue;
    }

    const weight = Number.isFinite(entry.weight) ? entry.weight : 1;
    const contribution = p * weight;
    // A mapped class with no probability mass must not register a category:
    // otherwise `scores` is never empty and the "nothing recognisable" branch
    // below could never fire.
    if (contribution <= 0) continue;
    matchedMass += contribution;

    const name = entry.name ?? classNames[i] ?? `index ${i}`;
    const acc = scores.get(entry.category);
    if (!acc) {
      scores.set(entry.category, {
        score: contribution,
        best: contribution,
        bestName: name,
        bestIndex: i,
      });
    } else {
      acc.score += contribution;
      // Lower index wins a tie so the strongest-contributor label is stable.
      if (contribution > acc.best) {
        acc.best = contribution;
        acc.bestName = name;
        acc.bestIndex = i;
      }
    }
  }

  const rawLabels = topIndices(probs, RAW_LABEL_COUNT).map((i) => ({
    label: classNames[i] ?? `index ${i}`,
    confidence: toNumber(probs[i]),
    index: i,
  }));

  if (scores.size === 0) {
    return {
      predictions: [{ category: 'trash', label: 'Unrecognised item', confidence: 0, share: 0 }],
      rawLabels,
      unmatchedMass,
      matchedMass: 0,
      lowConfidence: true,
    };
  }

  const ranked = [...scores.entries()].sort((a, b) => b[1].score - a[1].score || compareIds(a[0], b[0]));

  const predictions = ranked.slice(0, k).map(([category, acc]) => ({
    category,
    label: `${WASTE_CATEGORY_LABELS[category] ?? category} — ${shortName(acc.bestName)}`,
    confidence: Math.min(1, acc.score),
    share: matchedMass > 0 ? acc.score / matchedMass : 0,
  }));

  return {
    predictions,
    rawLabels,
    unmatchedMass,
    matchedMass,
    lowConfidence: predictions[0].confidence < LOW_CONFIDENCE_THRESHOLD,
  };
}

/** Canonical-order tie-break, so two equal scores never flip between frames. */
function compareIds(a, b) {
  const ia = WASTE_CATEGORY_IDS.indexOf(a);
  const ib = WASTE_CATEGORY_IDS.indexOf(b);
  if (ia !== -1 && ib !== -1) return ia - ib;
  return String(a).localeCompare(String(b));
}

function topIndices(probs, count) {
  const order = new Array(probs.length);
  for (let i = 0; i < probs.length; i += 1) order[i] = i;
  order.sort((a, b) => toNumber(probs[b]) - toNumber(probs[a]) || a - b);
  return order.slice(0, Math.min(count, order.length));
}

const CATEGORY_SYNONYMS = {
  'e-waste': 'ewaste',
  ewaste: 'ewaste',
  electronic: 'ewaste',
  electronics: 'ewaste',
  electronicwaste: 'ewaste',
  battery: 'hazardous',
  batteries: 'hazardous',
  chemical: 'hazardous',
  chemicals: 'hazardous',
  hazard: 'hazardous',
  hazardous: 'hazardous',
  biological: 'organic',
  compost: 'organic',
  food: 'organic',
  foodwaste: 'organic',
  green: 'organic',
  organic: 'organic',
  aluminium: 'metal',
  aluminum: 'metal',
  can: 'metal',
  cans: 'metal',
  metal: 'metal',
  tin: 'metal',
  clothes: 'textile',
  clothing: 'textile',
  fabric: 'textile',
  textile: 'textile',
  textiles: 'textile',
  cardboard: 'cardboard',
  carton: 'cardboard',
  glass: 'glass',
  paper: 'paper',
  plastic: 'plastic',
  general: 'trash',
  landfill: 'trash',
  other: 'trash',
  rubbish: 'trash',
  trash: 'trash',
  unknown: 'trash',
  waste: 'trash',
};

/**
 * Map a custom-model class name onto a canonical waste id.
 * @param {string} name
 * @returns {{category: string, exact: boolean}}
 */
export function resolveWasteCategory(name) {
  const raw = typeof name === 'string' ? name.trim() : '';
  if (!raw) return { category: 'trash', exact: false };

  const normalised = raw.toLowerCase().replace(/[\s_-]+/g, '');
  if (WASTE_CATEGORY_IDS.includes(normalised)) return { category: normalised, exact: true };

  const synonym = CATEGORY_SYNONYMS[normalised];
  if (synonym) return { category: synonym, exact: false };

  return { category: 'trash', exact: false };
}

/**
 * Aggregate a custom model's softmax over waste classes.
 *
 * @param {ArrayLike<number>} probs
 * @param {{topK?: number, classes: string[]}} options
 */
export function aggregateCustomPredictions(probs, { topK = 3, classes = [] } = {}) {
  if (!probs || typeof probs.length !== 'number') {
    throw new TypeError('aggregateCustomPredictions expects an array-like of probabilities');
  }

  const k = clampTopK(topK);
  let total = 0;
  for (let i = 0; i < probs.length; i += 1) total += toNumber(probs[i]);

  const predictions = topIndices(probs, k).map((i) => {
    const rawName = classes[i] ?? `class ${i}`;
    const { category, exact } = resolveWasteCategory(rawName);
    const label = exact
      ? (WASTE_CATEGORY_LABELS[category] ?? rawName)
      : `${WASTE_CATEGORY_LABELS[category] ?? category} — ${shortName(rawName)}`;
    const confidence = Math.min(1, Math.max(0, toNumber(probs[i])));
    return { category, label, confidence, share: total > 0 ? confidence / total : 0 };
  });

  return {
    predictions,
    rawLabels: null,
    unmatchedMass: 0,
    matchedMass: total,
    lowConfidence: (predictions[0]?.confidence ?? 0) < LOW_CONFIDENCE_THRESHOLD,
  };
}

/* ------------------------------------------------------------------ *
 * Model resolution & metadata
 * ------------------------------------------------------------------ */

const METADATA_DEFAULTS = {
  custom: {
    inputSize: DEFAULT_INPUT_SIZE,
    inputRange: [0, 1],
    outputActivation: 'softmax',
    classOffset: 0,
    labelKind: 'waste',
  },
  fallback: {
    inputSize: DEFAULT_INPUT_SIZE,
    inputRange: [0, 1],
    outputActivation: 'logits',
    classOffset: 1,
    labelKind: 'imagenet',
  },
};

/**
 * Model and metadata URLs come back from the API as root-relative paths that
 * the Vite proxy resolves. Honour VITE_API_BASE when the app is deployed
 * against a backend on another origin.
 */
function resolveAssetUrl(url) {
  if (typeof url !== 'string' || url.length === 0) return url;
  if (/^https?:\/\//i.test(url) || url.startsWith('data:')) return url;

  const base = (import.meta?.env?.VITE_API_BASE ?? '').replace(/\/+$/, '');
  if (!base) return url;
  return url.startsWith('/') ? `${base}${url}` : `${base}/${url}`;
}

function pickEngine(status) {
  const custom = status?.custom;
  const fallback = status?.fallback;

  if (custom?.available && custom.modelUrl) {
    return { kind: 'custom', modelUrl: custom.modelUrl, entry: custom };
  }
  if (fallback?.available && fallback.modelUrl) {
    return { kind: 'fallback', modelUrl: fallback.modelUrl, entry: fallback };
  }

  const searched = Array.isArray(status?.searchedPaths) ? status.searchedPaths : [];
  throw new EngineUnavailableError(
    'No TensorFlow.js model is available. Run `make fetch-models` to download the ' +
      'pretrained MobileNetV2 fallback, or `make train` to build your own model.',
    { searchedPaths: searched, active: status?.active ?? 'none' },
  );
}

/**
 * Pre-registry backends expose exactly these two slots (ARCHITECTURE 4), which
 * are also the directory names they are served from, so an engine loaded that
 * way still gets the same `modelId` the registry would have given it.
 */
const LEGACY_SLOT_IDS = { custom: 'custom', fallback: 'mobilenet_v2' };
const LEGACY_DISPLAY_NAMES = { custom: 'Custom model', fallback: 'MobileNetV2 (ImageNet)' };

/** A registry entry is only loadable once the backend has actually found its files. */
function isUsableEntry(entry) {
  return Boolean(entry && entry.available && entry.modelUrl);
}

/**
 * The registry says `kind: 'custom' | 'imagenet'`, but every existing consumer —
 * ModelStatusBadge, POST /api/classifications, the history filter, the SQLite
 * CHECK constraint — speaks `'custom' | 'fallback'`. Translate once, here, so
 * adding models never widens that vocabulary.
 */
function engineKindOf(entry) {
  return entry?.kind === 'custom' ? 'custom' : 'fallback';
}

function toBytes(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * 112262070 -> "107.1 MB". Deliberately local rather than imported from
 * lib/format.js: classifier.js is loaded alongside 100 MB of weights, so it
 * stays dependency-light and importable from plain Node tooling.
 */
function formatBytes(bytes) {
  const n = toBytes(bytes);
  if (n === null) return null;
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${Math.round(n)} bytes`;
}

/**
 * Decide which model to load (ARCHITECTURE 2.4).
 *
 * The requested id wins when it is installed; otherwise the backend's own
 * recommendation does. A selection is persisted in localStorage, so it outlives
 * the model it names: an id whose directory has since been deleted must degrade
 * to a working engine, never to an error screen. Only a backend with *nothing*
 * installed is allowed to fail the load.
 *
 * @param {object} status  the /api/model/status payload
 * @param {string} [modelId]  the caller's choice; '' or absent means "auto"
 */
function resolveChoice(status, modelId) {
  const requested = typeof modelId === 'string' ? modelId.trim() : '';
  const requestedModelId = requested || null;
  const models = Array.isArray(status?.models) ? status.models : [];

  const wanted = requested ? models.find((m) => m?.id === requested && isUsableEntry(m)) : null;
  // Registry order is already deterministic (the custom model first, then every
  // ImageNet model by ascending download size), so the first usable entry is the
  // right last resort if a backend ever reports models without a usable default.
  const auto =
    models.find((m) => m?.id === status?.defaultModelId && isUsableEntry(m)) ??
    models.find(isUsableEntry) ??
    null;

  const entry = wanted ?? auto;
  if (entry) {
    const kind = engineKindOf(entry);
    if (requested && entry.id !== requested) {
      console.warn(
        `[ecosort] the model "${requested}" is not available; loading "${entry.id}" instead.`,
      );
    }
    return {
      kind,
      modelUrl: entry.modelUrl,
      entry,
      modelId: entry.id ?? LEGACY_SLOT_IDS[kind],
      displayName: entry.displayName || entry.id || LEGACY_DISPLAY_NAMES[kind],
      downloadBytes: toBytes(entry.downloadBytes),
      requestedModelId,
    };
  }

  // No usable registry: an older backend that only knows `custom` and `fallback`.
  // pickEngine() is unchanged and still throws when neither of those exists.
  const legacy = pickEngine(status);
  if (requested && LEGACY_SLOT_IDS[legacy.kind] !== requested) {
    console.warn(
      `[ecosort] this backend reports no model registry, so "${requested}" could not be ` +
        `honoured; loading the ${legacy.kind} model instead.`,
    );
  }
  return {
    ...legacy,
    modelId: LEGACY_SLOT_IDS[legacy.kind],
    displayName: LEGACY_DISPLAY_NAMES[legacy.kind],
    downloadBytes: null,
    requestedModelId,
  };
}

function normaliseMetadata(kind, metadata) {
  const defaults = METADATA_DEFAULTS[kind];
  const meta = metadata && typeof metadata === 'object' ? metadata : {};

  const inputSize = Number(meta.inputSize);
  const classOffset = Number(meta.classOffset);

  return {
    ...meta,
    inputSize: Number.isFinite(inputSize) && inputSize > 0 ? Math.round(inputSize) : defaults.inputSize,
    inputRange: Array.isArray(meta.inputRange) && meta.inputRange.length === 2 ? meta.inputRange : defaults.inputRange,
    outputActivation: meta.outputActivation === 'logits' || meta.outputActivation === 'softmax'
      ? meta.outputActivation
      : defaults.outputActivation,
    classOffset: Number.isFinite(classOffset) && classOffset >= 0 ? Math.round(classOffset) : defaults.classOffset,
    labelKind: meta.labelKind === 'waste' || meta.labelKind === 'imagenet' ? meta.labelKind : defaults.labelKind,
    classes: Array.isArray(meta.classes) ? meta.classes : null,
  };
}

async function fetchJson(url, what) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new Error(`Could not fetch ${what} from ${url}: ${err.message}`, { cause: err });
  }
  if (!response.ok) {
    throw new Error(`Could not fetch ${what} from ${url}: HTTP ${response.status}`);
  }
  try {
    return await response.json();
  } catch (err) {
    throw new Error(`${what} at ${url} is not valid JSON: ${err.message}`, { cause: err });
  }
}

async function selectBackend() {
  let webglError = null;
  try {
    const ok = await tf.setBackend('webgl');
    if (!ok) throw new Error('the WebGL backend reported that it could not initialise');
  } catch (err) {
    webglError = err;
    try {
      const ok = await tf.setBackend('cpu');
      if (!ok) throw new Error('the CPU backend reported that it could not initialise');
    } catch (cpuError) {
      throw new Error(
        `No TensorFlow.js backend could start (webgl: ${webglError.message}; cpu: ${cpuError.message}).`,
        { cause: cpuError },
      );
    }
  }

  await tf.ready();
  return { backend: tf.getBackend(), webglError };
}

/* ------------------------------------------------------------------ *
 * Inference
 * ------------------------------------------------------------------ */

/** Some graph models return several tensors; the class scores are the widest 2-D one. */
function pickOutputTensor(output) {
  if (!Array.isArray(output)) return output;
  if (output.length === 0) throw new Error('The model returned no output tensors.');

  const twoD = output.filter((t) => t && Array.isArray(t.shape) && t.shape.length === 2);
  const pool = twoD.length > 0 ? twoD : output;
  return pool.reduce((best, t) => {
    const bw = best.shape[best.shape.length - 1] ?? 0;
    const tw = t.shape[t.shape.length - 1] ?? 0;
    // >= so that, on a tie, the LAST head wins: multi-head graphs put the
    // classifier logits after the feature/embedding outputs.
    return tw >= bw ? t : best;
  });
}

/**
 * A model's raw output -> [batch, classes] probabilities with the offset classes dropped.
 * Must be called inside a tidy scope.
 */
function toProbabilities(output, meta) {
  const scores2d = output.rank === 1 ? output.expandDims(0) : output;

  const width = scores2d.shape[scores2d.shape.length - 1];
  const offset = Math.min(meta.classOffset ?? 0, Math.max(0, width - 1));
  const sliced = offset > 0 ? scores2d.slice([0, offset], [-1, width - offset]) : scores2d;

  // A sliced softmax is intentionally NOT renormalised: the dropped classes
  // are real probability mass and hiding them would inflate confidences.
  return meta.outputActivation === 'logits' ? tf.softmax(sliced, -1) : sliced;
}

/**
 * (alignCorners=false, halfPixelCenters=true) is the exact kernel TF2's
 * tf.image.resize(method="bilinear") uses, which is what ml/train.py trains through.
 * alignCorners=true samples differently and puts the model on pixels it never saw;
 * alignCorners=false alone would select the TF1 legacy kernel, which is a different
 * mismatch again. Both flags are required.
 */
function resizeToInput(pixels, meta) {
  return tf.image.resizeBilinear(pixels, [meta.inputSize, meta.inputSize], false, true);
}

/**
 * One inference, start to finish, inside a single tidy scope.
 * Returns a 1-D probability tensor the caller must dispose.
 */
function runInference(model, source, meta) {
  return tf.tidy(() => {
    const pixels = source instanceof tf.Tensor ? source.clone() : tf.browser.fromPixels(source);
    const batched = resizeToInput(pixels, meta).toFloat().div(255).expandDims(0);

    const probs = toProbabilities(pickOutputTensor(model.predict(batched)), meta);

    return probs.shape[0] === 1 ? probs.squeeze([0]) : probs.slice([0, 0], [1, -1]).squeeze([0]);
  });
}

/**
 * Clamp a pixel rectangle into an H x W frame. Integer, at least 1x1, never outside it:
 * a slice that overhangs the tensor throws, and a detector box is only ever approximately
 * inside the frame.
 */
export function clampRegion(region, frameHeight, frameWidth) {
  const int = (value) => (Number.isFinite(Number(value)) ? Math.round(Number(value)) : 0);
  const x = Math.min(Math.max(0, int(region?.x)), Math.max(0, frameWidth - 1));
  const y = Math.min(Math.max(0, int(region?.y)), Math.max(0, frameHeight - 1));
  const width = Math.min(Math.max(1, int(region?.width)), frameWidth - x);
  const height = Math.min(Math.max(1, int(region?.height)), frameHeight - y);
  return { x, y, width, height };
}

/**
 * Batch sizes Live scan runs the classifier at. The WebGL backend compiles a shader per
 * distinct tensor shape, so feeding it "however many items are in view" would recompile
 * every layer of the network each time that count changed; padding up to the next bucket
 * caps it at four compilations for the life of the engine.
 */
const REGION_BATCH_BUCKETS = [1, 2, 4, 6];

export function regionBatchSize(count) {
  return REGION_BATCH_BUCKETS.find((size) => size >= count) ?? count;
}

/**
 * A pixel rectangle as the normalised [y1, x1, y2, x2] box tf.image.cropAndResize takes,
 * where a coordinate c maps to pixel c * (side - 1): the first and last pixel rows and
 * columns of the region become the first and last samples of the crop.
 */
export function cropBoxFor(region, frameHeight, frameWidth) {
  const r = clampRegion(region, frameHeight, frameWidth);
  const ny = Math.max(1, frameHeight - 1);
  const nx = Math.max(1, frameWidth - 1);
  return [r.y / ny, r.x / nx, (r.y + r.height - 1) / ny, (r.x + r.width - 1) / nx];
}

/**
 * Many crops of ONE frame in ONE predict call (Live scan, ARCHITECTURE 5.1).
 *
 * The crops are cut by a single cropAndResize, not by slice + resizeBilinear per crop as
 * runInference would: every crop has a different pixel size, and on WebGL each new size
 * compiled a fresh slice and resize shader - on every frame, since boxes move. With
 * cropAndResize the boxes are data, so the program only depends on the frame size and the
 * batch bucket. The price is its sampling grid, which lands on the region's first and
 * last pixels instead of TF2's half-pixel centres: a sub-pixel shift, far inside the
 * several pixels a detector box is uncertain by anyway.
 *
 * Returns a [batch, classes] probability tensor the caller must dispose. `batch` is
 * regionBatchSize(regions.length); the padding rows repeat the last region and are
 * ignored by the caller.
 */
function runRegionInference(model, source, regions, meta) {
  return tf.tidy(() => {
    const pixels = source instanceof tf.Tensor ? source : tf.browser.fromPixels(source);
    const [height, width] = pixels.shape;
    const batch = regionBatchSize(regions.length);

    const boxes = [];
    for (let i = 0; i < batch; i += 1) {
      boxes.push(cropBoxFor(regions[Math.min(i, regions.length - 1)], height, width));
    }

    const crops = tf.image.cropAndResize(
      pixels.toFloat().expandDims(0),
      tf.tensor2d(boxes, [batch, 4]),
      tf.zeros([batch], 'int32'),
      [meta.inputSize, meta.inputSize],
      'bilinear',
    );
    return toProbabilities(pickOutputTensor(model.predict(crops.div(255))), meta);
  });
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function makeReporter(onProgress) {
  return (stage, message, fraction) => {
    if (typeof onProgress !== 'function') return;
    try {
      onProgress({ stage, message, fraction: Math.max(0, Math.min(1, fraction)) });
    } catch (err) {
      // A broken progress consumer must not abort a model load.
      console.warn('[ecosort] onProgress callback threw:', err);
    }
  };
}

/**
 * Serialises classify() calls: webcam live mode fires faster than WebGL can
 * finish, and overlapping `predict` calls on one model thrash GPU memory.
 */
function makeQueue() {
  let tail = Promise.resolve();
  return function enqueue(task) {
    const result = tail.then(task, task);
    tail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  };
}

export function deriveCustomClasses(meta, classCount, modelId = 'custom') {
  if (Array.isArray(meta.classes) && meta.classes.length === classCount) return meta.classes;
  if (Array.isArray(meta.classes) && meta.classes.length > 0) {
    throw new Error(
      `models/${modelId}/metadata.json lists ${meta.classes.length} classes but the model ` +
        `outputs ${classCount}. Re-export the model with ml/train.py so the two agree.`,
    );
  }
  // No `classes` at all: refuse rather than guess. The trainer orders its softmax
  // alphabetically; the taxonomy in WASTE_CATEGORY_IDS is ordered semantically. When a
  // model covers the whole taxonomy the two are the same length, so a size check cannot
  // tell them apart - guessing mislabels every prediction with no error and no warning,
  // which is far worse than failing to load.
  throw new Error(
    `The custom model outputs ${classCount} classes but models/${modelId}/metadata.json has ` +
      'no `classes` array, so the outputs cannot be named. Retrain with `make train`, which ' +
      'writes the class order the model was actually trained in. That order is the sorted ' +
      'class list recorded in ml/dataset/dataset.json - not the order the app displays ' +
      'categories in.',
  );
}

/**
 * Load one engine — the requested model when it is installed, the backend's
 * recommendation otherwise (ARCHITECTURE 2.4).
 *
 * @param {{modelId?: string|null,
 *          onProgress?: (p: {stage: string, message: string, fraction: number}) => void}} [options]
 * @returns {Promise<object>} Engine
 */
export async function loadEngine({ modelId, onProgress } = {}) {
  const report = makeReporter(onProgress);

  report('probing', 'Looking for a local model…', 0.02);

  let status;
  try {
    status = await getModelStatus();
  } catch (err) {
    throw new EngineUnavailableError(
      `Could not reach the EcoSort API to look for a model: ${err.message}. ` +
        'Is the backend running (`docker compose up --build`)?',
      { cause: err.message ?? String(err) },
    );
  }

  const choice = resolveChoice(status, modelId);
  const { kind, entry, displayName, requestedModelId } = choice;
  const resolvedModelId = choice.modelId;
  const resolvedModelUrl = resolveAssetUrl(choice.modelUrl);

  let metadataSource = 'defaults';
  let rawMetadata = entry.metadata ?? null;
  if (rawMetadata) {
    metadataSource = 'status';
  } else if (entry.metadataUrl) {
    try {
      rawMetadata = await fetchJson(resolveAssetUrl(entry.metadataUrl), 'model metadata');
      metadataSource = 'fetched';
    } catch (err) {
      // Not fatal per ARCHITECTURE 2.3 — the defaults are the verified values.
      console.warn(`[ecosort] falling back to default metadata for ${displayName}:`, err.message);
      rawMetadata = null;
    }
  }
  const meta = normaliseMetadata(kind, rawMetadata);
  // The registry computes this from the weight manifest; metadata.json carries it
  // too for models converted by ml/convert_pretrained.py, which is the only
  // source an older backend has.
  const downloadBytes = choice.downloadBytes ?? toBytes(meta.downloadBytes);

  report('probing', 'Starting the TensorFlow.js backend…', 0.08);
  const { backend, webglError } = await selectBackend();
  if (webglError) {
    console.warn(`[ecosort] WebGL unavailable, running on ${backend}: ${webglError.message}`);
  }

  // Sniff model.json before handing the URL to tfjs: the pretrained models are
  // graph models (the MobileNetV2 TF-Hub conversion has no `format` field at
  // all, ml/convert_pretrained.py writes "graph-model") while ml/train.py
  // exports a layers model. Loading one with the other's loader throws.
  report('probing', 'Inspecting the model file…', 0.12);
  const modelJson = await fetchJson(resolvedModelUrl, 'model.json');
  const isLayersModel = modelJson?.format === 'layers-model';

  // Naming the size matters once a model can be ~107 MB: a progress bar that
  // creeps without a number reads as a hang, not as a download.
  const sizeLabel = formatBytes(downloadBytes);
  const downloadMessage = sizeLabel
    ? `Downloading model weights (${sizeLabel})…`
    : 'Downloading model weights…';

  report('loading', downloadMessage, 0.15);
  let model;
  try {
    const loadOptions = {
      // tfjs reports a 0..1 fraction over the whole weight manifest; it is the
      // only part of the load that visibly moves, so it owns most of the bar.
      onProgress: (fraction) => {
        report('loading', downloadMessage, 0.15 + toNumber(fraction) * 0.7);
      },
    };
    model = isLayersModel
      ? await tf.loadLayersModel(resolvedModelUrl, loadOptions)
      : await tf.loadGraphModel(resolvedModelUrl, loadOptions);
  } catch (err) {
    throw new Error(
      `Failed to load ${displayName} from ${resolvedModelUrl}: ${err.message}`,
      { cause: err },
    );
  }

  report('warmup', 'Warming up the model…', 0.9);

  let classCount = 0;
  const warmupStarted = nowMs();
  // meta.inputSize, never a constant: 224 for the MobileNets, 299 for Inception.
  const warmupInput = tf.zeros([meta.inputSize, meta.inputSize, 3], 'float32');
  try {
    const probs = runInference(model, warmupInput, meta);
    classCount = probs.shape[0];
    probs.dispose();
  } catch (err) {
    model.dispose();
    throw new Error(
      `${displayName} loaded but failed its warm-up inference at ` +
        `${meta.inputSize}x${meta.inputSize}: ${err.message}`,
      { cause: err },
    );
  } finally {
    warmupInput.dispose();
  }
  const warmupMs = Math.round(nowMs() - warmupStarted);

  let classes;
  try {
    classes =
      meta.labelKind === 'waste'
        ? deriveCustomClasses(meta, classCount, resolvedModelId)
        : IMAGENET_CLASSES;
  } catch (err) {
    model.dispose();
    throw err;
  }

  if (meta.labelKind === 'imagenet' && classCount !== IMAGENET_CLASSES.length) {
    model.dispose();
    throw new Error(
      `${displayName} produced ${classCount} classes after dropping ${meta.classOffset} ` +
        `offset class(es); ${IMAGENET_CLASSES.length} were expected, so its outputs cannot be ` +
        `named from imagenetClasses.js. The files in models/${resolvedModelId}/ look wrong — ` +
        're-run `make fetch-models`, or re-convert the model with ml/convert_pretrained.py.',
    );
  }

  const enqueue = makeQueue();
  let disposed = false;
  let regionBatchesWarm = null;

  const aggregate = (probs, topK) =>
    meta.labelKind === 'waste'
      ? aggregateCustomPredictions(probs, { topK, classes })
      : aggregateImagenetPredictions(probs, { topK, indexMap: defaultIndexMap() });

  const engine = {
    kind,
    modelId: resolvedModelId,
    displayName,
    downloadBytes,
    // The id the caller asked for, or null when it asked for "auto". It differs
    // from `modelId` exactly when the request could not be honoured, which is
    // how the UI knows to tell the user their stored choice is gone.
    requestedModelId,
    classes,
    inputSize: meta.inputSize,
    metadata: meta,
    metadataSource,
    modelUrl: resolvedModelUrl,
    backend,
    classCount,
    warmupMs,

    get disposed() {
      return disposed;
    },

    async classify(source, { topK = 3 } = {}) {
      if (disposed) throw new Error('This classifier engine has already been disposed.');
      if (!source) {
        throw new TypeError('classify(source) needs an image, video or canvas element.');
      }

      return enqueue(async () => {
        if (disposed) throw new Error('This classifier engine has already been disposed.');

        const started = nowMs();
        const probsTensor = runInference(model, source, meta);
        let probs;
        try {
          probs = await probsTensor.data();
        } finally {
          probsTensor.dispose();
        }

        const summary = aggregate(probs, topK);

        return {
          predictions: summary.predictions,
          rawLabels: summary.rawLabels,
          modelKind: kind,
          durationMs: Math.round(nowMs() - started),
          lowConfidence: summary.lowConfidence,
          unmatchedMass: summary.unmatchedMass,
        };
      });
    },

    /**
     * Classify several regions of one frame in a single batched inference.
     *
     * @param {tf.Tensor3D|HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source
     *   a [H,W,3] pixel tensor (the caller keeps ownership) or anything fromPixels takes
     * @param {Array<{x:number, y:number, width:number, height:number}>} regions pixels
     * @returns {Promise<{results: Array<object>, modelKind: string, durationMs: number}>}
     *   one entry per region, in order, shaped like classify()'s result
     */
    async classifyRegions(source, regions, { topK = 3 } = {}) {
      if (disposed) throw new Error('This classifier engine has already been disposed.');
      if (!source) {
        throw new TypeError('classifyRegions(source) needs an image, video, canvas or tensor.');
      }
      const list = Array.isArray(regions) ? regions : [];
      if (list.length === 0) return { results: [], modelKind: kind, durationMs: 0 };

      return enqueue(async () => {
        if (disposed) throw new Error('This classifier engine has already been disposed.');

        const started = nowMs();
        const probsTensor = runRegionInference(model, source, list, meta);
        let flat;
        let width;
        try {
          width = probsTensor.shape[1];
          flat = await probsTensor.data();
        } finally {
          probsTensor.dispose();
        }

        const results = list.map((_, i) => {
          const summary = aggregate(flat.subarray(i * width, (i + 1) * width), topK);
          return {
            predictions: summary.predictions,
            rawLabels: summary.rawLabels,
            lowConfidence: summary.lowConfidence,
            unmatchedMass: summary.unmatchedMass,
          };
        });

        return { results, modelKind: kind, durationMs: Math.round(nowMs() - started) };
      });
    },

    /**
     * Compile the network for every region batch size up front. Otherwise the first frame
     * with a third item in view stalls for as long as a whole model warm-up (a second or
     * more on integrated graphics) while WebGL compiles the batch-of-4 shaders. Idempotent;
     * only Live scan calls it, so the Classify tab never pays for it.
     */
    warmRegionBatches() {
      if (disposed) return Promise.reject(new Error('This classifier engine has already been disposed.'));
      if (!regionBatchesWarm) {
        regionBatchesWarm = enqueue(async () => {
          for (const size of REGION_BATCH_BUCKETS) {
            if (disposed) return;
            const probs = tf.tidy(() =>
              toProbabilities(
                pickOutputTensor(model.predict(tf.zeros([size, meta.inputSize, meta.inputSize, 3]))),
                meta,
              ),
            );
            try {
              await probs.data();
            } finally {
              probs.dispose();
            }
          }
        });
      }
      return regionBatchesWarm;
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      // model.dispose() frees this model's own tensors and nothing else. Do NOT add
      // tf.disposeVariables() here: it is global, so tearing down a superseded engine
      // would wipe the weights of the engine that just replaced it — which is exactly
      // what happens on every model switch.
      model.dispose();
    },
  };

  report('ready', `Ready — ${displayName} on ${backend}.`, 1);
  return engine;
}

export default loadEngine;
