/**
 * Discovery of the TFJS engines the browser can load (docs/ARCHITECTURE.md sections 2 and 2.4).
 *
 * A model can appear *while the server runs*: the user trains one with ml/train.py, or converts a
 * pretrained one with ml/convert_pretrained.py, straight into the bind-mounted MODELS_DIR. So this
 * probes the filesystem on demand instead of scanning once at boot — with a short cache, because
 * the frontend polls this endpoint.
 *
 * Two views of the same directories are published side by side:
 *   - `custom` / `fallback` — the two fixed slots the existing UI and the `model_kind` column key
 *     off. Their meaning never changes, whatever else is installed.
 *   - `models` — the open registry: every directory holding a readable model.json, which is what
 *     the model picker renders.
 */
import fs from 'node:fs';
import path from 'node:path';

import { config } from '../config.js';
import { createLogger } from '../logger.js';

const logger = createLogger('model');

/** Short enough that a freshly trained model shows up almost immediately. */
const CACHE_TTL_MS = 5000;

/** The two ids with a fixed contractual meaning; every other directory is just a registry entry. */
const CUSTOM_ID = 'custom';
const FALLBACK_ID = 'mobilenet_v2';

/** Directory on disk -> key in the /api/model/status payload. */
const SLOTS = Object.freeze([
  Object.freeze({ dir: CUSTOM_ID, key: 'custom' }),
  Object.freeze({ dir: FALLBACK_ID, key: 'fallback' }),
]);

/**
 * Ids whose conventional casing no generic rule would ever reproduce. Everything else goes through
 * the underscore/dash -> space path in prettifyId(): guessing where the word boundaries are inside
 * an arbitrary id produces worse names than leaving it alone, and the fix is a `displayName` in
 * the model's own metadata.json anyway.
 */
const PRETTY_IDS = Object.freeze({
  [CUSTOM_ID]: 'Custom model',
  [FALLBACK_ID]: 'MobileNetV2',
  inceptionresnetv2: 'InceptionResNetV2',
});

let cache = null;

/**
 * model.json size+mtime -> measured download size. Every writer (both ml exporters and
 * scripts/fetch-mobilenet.mjs) builds into a scratch directory and renames the finished thing into
 * place, so a changed model.json is a sound signal that the shards beside it changed too. Without
 * this the 5-second cache refresh would re-stat every shard of every model (27 of them for
 * InceptionResNetV2).
 */
const measuredBytes = new Map();

/** Models roots that failed to list, so a chronically missing BUNDLED_MODELS_DIR warns once. */
const warnedRoots = new Set();

function isFile(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch (err) {
    // ENOENT/ENOTDIR are the normal "not trained yet" answer; anything else is worth saying once.
    if (err.code !== 'ENOENT' && err.code !== 'ENOTDIR') {
      logger.warn('Could not stat a model path', { path: filePath, message: err.message });
    }
    return false;
  }
}

function readMetadata(filePath) {
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (err) {
    // Never fatal: the synthesised descriptor below keeps the frontend working.
    logger.warn('Ignoring unreadable model metadata', { path: filePath, message: err.message });
    return null;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    logger.warn('Ignoring model metadata that is not a JSON object', { path: filePath });
    return null;
  }
  return parsed;
}

/**
 * Turns a directory id into something a picker can show. Known ids come from the table above;
 * anything else is split on separators and capitalised, which is right for `food_waste_v2` and
 * harmless for a single lowercase word.
 */
export function prettifyId(id) {
  const key = String(id);
  if (Object.prototype.hasOwnProperty.call(PRETTY_IDS, key)) return PRETTY_IDS[key];
  const words = key.split(/[_\-\s.]+/).filter((word) => word.length > 0);
  if (words.length === 0) return key;
  return words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

/**
 * The defaults from ARCHITECTURE sections 2.3 and 2.4. The fallback numbers are the empirically
 * verified MobileNetV2 facts: [0,1] input, raw 1001-way logits, index 0 is the TF-Slim background
 * class. An id this function does not know still gets a complete descriptor, so a directory that
 * ships no metadata.json at all is selectable rather than broken.
 */
export function defaultMetadata(slotDir) {
  if (slotDir === CUSTOM_ID) {
    return {
      name: 'ecosort-custom',
      displayName: prettifyId(CUSTOM_ID),
      description: 'Trained on your own dataset; predicts the waste categories directly.',
      version: '0.0.0',
      createdAt: null,
      baseModel: 'MobileNetV2',
      inputSize: 224,
      inputRange: [0, 1],
      outputActivation: 'softmax',
      classOffset: 0,
      labelKind: 'waste',
      // Deliberately null, NOT the taxonomy. ml/train.py orders its softmax alphabetically
      // (cardboard, ewaste, glass, ...) while the taxonomy is ordered semantically (plastic,
      // paper, cardboard, ...). A model covering the whole taxonomy matches it in length but
      // not in order, so guessing here would pass every downstream guard and mislabel every
      // prediction in silence. A model trained on a subset of the categories does not match
      // in length either. Unknown is the honest answer in both cases; the frontend refuses
      // to name outputs it cannot name.
      classes: null,
      classCount: null,
      quantization: 'none',
      downloadBytes: null,
      metrics: null,
      notes:
        'Synthesised descriptor: models/custom/metadata.json is missing or unreadable, so the class names are unknown. Re-run ml/train.py to regenerate it.',
    };
  }
  if (slotDir === FALLBACK_ID) {
    return {
      name: 'mobilenet_v2_1.0_224',
      displayName: prettifyId(FALLBACK_ID),
      description: 'Pretrained ImageNet-1k classifier mapped onto the waste categories.',
      version: '1.0.0',
      createdAt: null,
      baseModel: 'MobileNetV2',
      inputSize: 224,
      inputRange: [0, 1],
      outputActivation: 'logits',
      classOffset: 1,
      labelKind: 'imagenet',
      classes: null,
      classCount: 1001,
      quantization: 'none',
      downloadBytes: null,
      metrics: null,
      notes:
        'Synthesised descriptor: models/mobilenet_v2/metadata.json is missing or unreadable. Pretrained ImageNet-1k graph model; slice off the background class at index 0 before softmax.',
    };
  }
  return {
    name: String(slotDir),
    displayName: prettifyId(slotDir),
    description: '',
    version: '0.0.0',
    createdAt: null,
    baseModel: null,
    // The conservative reading of an undescribed model: a plain ImageNet-1k softmax at the size
    // every MobileNet uses. A model that differs says so in its own metadata.json.
    inputSize: 224,
    inputRange: [0, 1],
    outputActivation: 'softmax',
    classOffset: 0,
    labelKind: 'imagenet',
    classes: null,
    classCount: null,
    quantization: 'none',
    downloadBytes: null,
    metrics: null,
    notes:
      'Synthesised descriptor: this model ships no readable metadata.json, so a 1000-class ImageNet softmax over [0,1] input at 224x224 is assumed. Add a metadata.json next to model.json to say otherwise.',
  };
}

function definedEntries(object) {
  return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined));
}

function modelRoots() {
  const roots = [];
  // MODELS_DIR first: a freshly trained model must shadow the image-baked copy.
  for (const entry of [
    { kind: 'models-dir', dir: config.modelsDir },
    { kind: 'bundled', dir: config.bundledModelsDir },
  ]) {
    if (typeof entry.dir !== 'string' || entry.dir === '') continue;
    // On the host both env vars default to <repo>/models; probing it twice is just noise.
    if (roots.some((root) => root.dir === entry.dir)) continue;
    roots.push(entry);
  }
  return roots;
}

/**
 * Locates each slot's model.json and its sibling metadata.json.
 * `metadata` is the parsed file or null — synthesising defaults is getModelStatus()'s job.
 *
 * This deliberately probes the two fixed slots by path instead of going through discovery: the
 * `searchedPaths` it reports are the paths that *would* hold a model, which is what the frontend
 * shows the user when nothing is installed and discovery therefore finds nothing.
 */
export function resolveModelPaths() {
  const roots = modelRoots();
  const searchedPaths = [];
  const slots = {};

  for (const slot of SLOTS) {
    let winner = null;
    for (const root of roots) {
      const modelPath = path.join(root.dir, slot.dir, 'model.json');
      searchedPaths.push(modelPath);
      if (winner === null && isFile(modelPath)) {
        winner = { source: root.kind, dir: path.join(root.dir, slot.dir), modelPath };
      }
    }

    const metadataPath = winner === null ? null : path.join(winner.dir, 'metadata.json');
    const metadataExists = metadataPath !== null && isFile(metadataPath);

    slots[slot.key] = {
      key: slot.key,
      slot: slot.dir,
      available: winner !== null,
      source: winner === null ? null : winner.source,
      dir: winner === null ? null : winner.dir,
      modelPath: winner === null ? null : winner.modelPath,
      metadataPath: metadataExists ? metadataPath : null,
      metadata: metadataExists ? readMetadata(metadataPath) : null,
    };
  }

  return { slots, searchedPaths };
}

/**
 * Directories the picker must never show: dotfiles — which covers the `.<id>.staging-<pid>` scratch
 * directories ml/export_tfjs.py and ml/convert_pretrained.py build into and the `.<id>.tmp-<pid>`
 * one scripts/fetch-mobilenet.mjs downloads into — and, belt and braces, anything else carrying the
 * staging marker. Offering a half-written export would fail in the browser after a 100 MB download.
 */
function isSelectableDirName(name) {
  return !name.startsWith('.') && !name.includes('.staging-');
}

function listRootEntries(root) {
  try {
    return fs.readdirSync(root, { withFileTypes: true });
  } catch (err) {
    // A missing MODELS_DIR is normal before the first `docker compose up`; an unreadable one is a
    // permissions problem the operator needs told about. Neither may break the endpoint, and
    // neither may be repeated every five seconds for the life of the process.
    const key = `${root}:${err.code ?? 'unknown'}`;
    if (!warnedRoots.has(key)) {
      warnedRoots.add(key);
      logger.warn('Skipping a models root that could not be listed', {
        root,
        code: err.code ?? null,
        message: err.message,
      });
    }
    return [];
  }
}

/**
 * Every directory under a models root that holds a readable model.json, MODELS_DIR winning over
 * BUNDLED_MODELS_DIR for the same id exactly as the fixed slots do.
 */
export function discoverModelDirs() {
  const found = new Map();

  for (const root of modelRoots()) {
    const entries = listRootEntries(root.dir)
      // Symlinks are followed: a models root may well point its entries at another volume.
      .filter((entry) => entry.isDirectory() || entry.isSymbolicLink())
      .map((entry) => entry.name)
      .filter(isSelectableDirName)
      // readdir order is filesystem order; sorting keeps searchedPaths stable between requests.
      .sort();

    for (const id of entries) {
      if (found.has(id)) continue; // the MODELS_DIR copy shadows the image-baked one
      const dir = path.join(root.dir, id);
      const modelPath = path.join(dir, 'model.json');
      if (!isFile(modelPath)) continue;
      const metadataPath = path.join(dir, 'metadata.json');
      found.set(id, {
        id,
        source: root.kind,
        dir,
        modelPath,
        metadataPath: isFile(metadataPath) ? metadataPath : null,
      });
    }
  }

  return [...found.values()];
}

/**
 * What the browser actually downloads: model.json plus every shard its weights manifest lists.
 * A shard the manifest names but that is not on disk is skipped rather than fatal, so an
 * interrupted download shows a too-small number instead of removing the model from the picker.
 */
function measureDownloadBytes(modelPath) {
  try {
    const parsed = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const groups = Array.isArray(parsed?.weightsManifest) ? parsed.weightsManifest : [];
    let total = fs.statSync(modelPath).size;

    for (const group of groups) {
      const shards = Array.isArray(group?.paths) ? group.paths : [];
      for (const shard of shards) {
        if (typeof shard !== 'string' || shard === '') continue;
        try {
          const stat = fs.statSync(path.resolve(path.dirname(modelPath), shard));
          if (stat.isFile()) total += stat.size;
        } catch {
          // Missing shard: see the note above.
        }
      }
    }

    return total;
  } catch (err) {
    // A malformed model.json must cost this model its size label, not the whole endpoint.
    logger.warn('Could not measure a model download size', {
      path: modelPath,
      message: err.message,
    });
    return null;
  }
}

function downloadBytesFor(modelPath) {
  let stamp;
  try {
    const stat = fs.statSync(modelPath);
    stamp = `${stat.mtimeMs}:${stat.size}`;
  } catch {
    return null;
  }

  const cached = measuredBytes.get(modelPath);
  if (cached !== undefined && cached.stamp === stamp) return cached.bytes;

  const bytes = measureDownloadBytes(modelPath);
  measuredBytes.set(modelPath, { stamp, bytes });
  return bytes;
}

function describe(slot) {
  if (!slot.available) {
    return { available: false, modelUrl: null, metadataUrl: null, metadata: null };
  }
  const defaults = defaultMetadata(slot.slot);
  return {
    available: true,
    modelUrl: `/models/${slot.slot}/model.json`,
    metadataUrl: slot.metadataPath === null ? null : `/models/${slot.slot}/metadata.json`,
    // Merged, not replaced: a hand-written metadata.json that omits inputRange still yields a
    // descriptor classifier.js can drive a single preprocessing path from.
    metadata: slot.metadata === null ? defaults : { ...defaults, ...definedEntries(slot.metadata) },
  };
}

/** One registry entry. `recommended` is decided once the whole list is ordered. */
function describeModel(found) {
  const defaults = defaultMetadata(found.id);
  const fileMetadata = found.metadataPath === null ? null : readMetadata(found.metadataPath);
  const metadata = fileMetadata === null ? defaults : { ...defaults, ...definedEntries(fileMetadata) };

  // The exporter knows exactly how many bytes it wrote, so a usable declared value wins; weighing
  // the files is the fallback for hand-written descriptors and for the pretrained fallback.
  const declared = metadata.downloadBytes;
  const downloadBytes =
    typeof declared === 'number' && Number.isFinite(declared) && declared > 0
      ? declared
      : downloadBytesFor(found.modelPath);

  const named = typeof metadata.displayName === 'string' ? metadata.displayName.trim() : '';
  const displayName = named === '' ? prettifyId(found.id) : named;
  const description = typeof metadata.description === 'string' ? metadata.description : '';

  // Keep the descriptor and the entry telling the same story — the picker reads the entry, but
  // classifier.js reads the descriptor.
  metadata.displayName = displayName;
  metadata.description = description;
  metadata.downloadBytes = downloadBytes;

  return {
    id: found.id,
    displayName,
    description,
    // 'custom' means "predicts waste ids directly", which is exactly what labelKind says.
    kind: metadata.labelKind === 'waste' ? 'custom' : 'imagenet',
    available: true, // a directory only reaches this point because its model.json is readable
    source: found.source,
    modelUrl: `/models/${found.id}/model.json`,
    metadataUrl: found.metadataPath === null ? null : `/models/${found.id}/metadata.json`,
    metadata: Object.freeze(metadata),
    downloadBytes,
    recommended: false,
  };
}

/** Unknown sizes sort last; the id is the tie-break so the order never wobbles between requests. */
function compareByDownload(a, b) {
  const left = a.downloadBytes === null ? Number.POSITIVE_INFINITY : a.downloadBytes;
  const right = b.downloadBytes === null ? Number.POSITIVE_INFINITY : b.downloadBytes;
  if (left !== right) return left < right ? -1 : 1;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * ARCHITECTURE 2.4: the `custom` slot first when it exists, then everything else by download size
 * ascending so the cheapest option is offered first. The head of that list is the auto-pick.
 */
/**
 * Re-add any fixed slot that resolved by path but that directory listing missed.
 *
 * The two happen through different syscalls: the slots stat a known path, discovery lists the
 * root. A root that is traversable but not readable (mode 0711, or a restrictive bind mount)
 * satisfies the first and fails the second, which would otherwise report `active: "custom"`
 * alongside an empty registry — a model the UI says is running but cannot offer in the picker.
 */
function backfillSlots(discovered, slots) {
  const byId = new Set(discovered.map((entry) => entry.id));
  const extra = [];

  for (const slot of Object.values(slots)) {
    if (!slot.available || byId.has(slot.slot)) continue;
    extra.push({
      id: slot.slot,
      source: slot.source,
      dir: slot.dir,
      modelPath: slot.modelPath,
      metadataPath: slot.metadataPath,
    });
    logger.warn('Model directory resolved by path but was not listable; adding it to the registry', {
      id: slot.slot,
      dir: slot.dir,
    });
  }

  return extra.length === 0 ? discovered : [...discovered, ...extra];
}

function buildRegistry(discovered) {
  const entries = discovered.map(describeModel);
  const custom = entries.filter((entry) => entry.id === CUSTOM_ID);
  const rest = entries.filter((entry) => entry.id !== CUSTOM_ID).sort(compareByDownload);
  const ordered = [...custom, ...rest];

  if (ordered.length > 0) ordered[0].recommended = true;

  return Object.freeze(ordered.map((entry) => Object.freeze(entry)));
}

/** Payload for GET /api/model/status. `now` is injectable so tests can drive the cache clock. */
export function getModelStatus(now = Date.now()) {
  if (cache !== null && now >= cache.at && now - cache.at < CACHE_TTL_MS) {
    return cache.value;
  }

  const { slots, searchedPaths } = resolveModelPaths();
  const custom = describe(slots.custom);
  const fallback = describe(slots.fallback);
  const discovered = backfillSlots(discoverModelDirs(), slots);
  const models = buildRegistry(discovered);
  const recommended = models.find((entry) => entry.recommended) ?? null;

  const value = Object.freeze({
    custom,
    fallback,
    // Unchanged three-valued contract: it names a *slot*, not a registry entry, because the
    // `model_kind` column and the existing UI are written in those terms.
    active: custom.available ? 'custom' : fallback.available ? 'fallback' : 'none',
    // The two slots' candidate paths, plus anything discovery turned up elsewhere, deduplicated.
    searchedPaths: [...new Set([...searchedPaths, ...discovered.map((entry) => entry.modelPath)])],
    models,
    defaultModelId: recommended === null ? null : recommended.id,
  });

  const previousIds = cache === null ? null : cache.value.models.map((entry) => entry.id).join(',');
  const currentIds = value.models.map((entry) => entry.id).join(',');
  if (cache === null || cache.value.active !== value.active || previousIds !== currentIds) {
    logger.info('Model availability resolved', {
      active: value.active,
      custom: custom.available,
      fallback: fallback.available,
      models: value.models.map((entry) => entry.id),
      defaultModelId: value.defaultModelId,
    });
  }

  cache = { at: now, value };
  return value;
}

/** One registry entry by id, or null. Shares the status cache, so the picker costs no extra I/O. */
export function getModel(id, now = Date.now()) {
  if (typeof id !== 'string' || id === '') return null;
  return getModelStatus(now).models.find((entry) => entry.id === id) ?? null;
}

export function clearModelStatusCache() {
  cache = null;
  measuredBytes.clear();
  warnedRoots.clear();
}
