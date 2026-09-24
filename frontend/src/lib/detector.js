/**
 * The EcoSort object detector - the first stage of Live scan (docs/ARCHITECTURE.md 5.1).
 *
 * The classifiers answer "what is this picture of?"; they assume one item filling the
 * frame. Live scan needs "where are the items?" first, so a pretrained COCO SSDLite
 * MobileNetV2 proposes boxes and the active waste classifier then names each crop
 * (classifier.js `classifyRegions`). Two deliberate departures from how COCO-SSD is
 * normally used:
 *
 * 1. The COCO label is NOT the answer. Measured on the waste test set, SSDLite's label is
 *    usually wrong for waste (a phone scored as "bicycle", a battery as "traffic light")
 *    while its box is usually right. So a box's score is its best score over every class
 *    that could plausibly be an item - class-agnostic objectness - and naming it is the
 *    waste classifier's job.
 * 2. The default threshold is 0.2, not COCO-SSD's 0.5. Waste photos are far from COCO:
 *    at 0.5 only 31% of test items produced any box at all, at 0.2 two thirds did, and the
 *    classifier downstream copes with the extra false positives far better than the user
 *    copes with an item that never gets a box.
 *
 * The model contract (input, outputs, class offset) is fixed here rather than read from
 * metadata.json - there is exactly one detector, and scripts/fetch-detector.mjs writes the
 * same facts down for humans.
 */

import * as tf from '@tensorflow/tfjs';

import { API_BASE } from './api.js';
import { boxArea, boxFromYxyx, iou } from './boxes.js';

const MODEL_DIR = '/models/detectors/ssdlite_mobilenet_v2';
export const DETECTOR_MODEL_PATH = `${MODEL_DIR}/model.json`;
const DETECTOR_METADATA_PATH = `${MODEL_DIR}/metadata.json`;

/**
 * The graph resizes any input to 300x300 itself (without keeping the aspect ratio), so
 * handing it exactly that shrinks a 720p frame once on the GPU instead of uploading it
 * whole into the preprocessor's while loop.
 */
const DETECTOR_INPUT_SIZE = 300;

/** Score column j is COCO category id j + 1 (column 0 is id 1, "person"). */
const CLASS_OFFSET = 1;
const CLASS_COUNT = 90;

export const DEFAULT_MIN_SCORE = 0.2;
export const DEFAULT_MAX_DETECTIONS = 6;
export const DEFAULT_IOU_THRESHOLD = 0.45;
/** Below 0.4% of the frame a box is a speck of texture, not an item to sort. */
export const MIN_BOX_AREA = 0.004;

/** COCO 2017 category ids (the 80 in use out of 90). */
export const COCO_LABELS = Object.freeze({
  1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
  8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
  14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
  21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
  28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
  36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
  41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
  47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
  54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
  60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
  67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
  76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
  82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
  89: 'hair drier', 90: 'toothbrush',
});

/**
 * Classes whose box is never an item to sort, whatever it contains: the person holding
 * the item, and the furniture and fixtures it is resting on. Everything else stays a
 * candidate even when its label is absurd, because the label is not what is used.
 */
export const IGNORED_COCO_IDS = Object.freeze(
  new Set([1, 15, 62, 63, 65, 67, 70, 81, 82]),
);

export function cocoLabel(id) {
  return COCO_LABELS[id] ?? 'object';
}

/** 1 for every score column that may propose an item, 0 for the ignored ones. */
export function buildClassMask(classCount = CLASS_COUNT, ignored = IGNORED_COCO_IDS) {
  const mask = new Float32Array(classCount);
  for (let column = 0; column < classCount; column += 1) {
    mask[column] = ignored.has(column + CLASS_OFFSET) ? 0 : 1;
  }
  return mask;
}

/**
 * From per-box best scores to the final list: threshold, drop specks, then greedy
 * class-agnostic non-max suppression. Pure, so the whole policy is unit tested without a
 * model; the tensors are reduced to three flat arrays before this is called.
 *
 * @param {{scores: ArrayLike<number>, classes: ArrayLike<number>, boxes: ArrayLike<number>,
 *          minScore?: number, maxDetections?: number, iouThreshold?: number, minArea?: number}} input
 *   `scores[i]` / `classes[i]` are box i's best allowed score and its score column;
 *   `boxes` holds `[ymin, xmin, ymax, xmax]` per box, normalised.
 * @returns {Array<{box: {x:number,y:number,width:number,height:number}, score: number,
 *          cocoId: number, cocoLabel: string}>} strongest first
 */
export function selectDetections({
  scores,
  classes,
  boxes,
  minScore = DEFAULT_MIN_SCORE,
  maxDetections = DEFAULT_MAX_DETECTIONS,
  iouThreshold = DEFAULT_IOU_THRESHOLD,
  minArea = MIN_BOX_AREA,
}) {
  const count = Math.min(scores.length, classes.length, Math.floor(boxes.length / 4));
  const limit = Math.max(0, Math.floor(maxDetections));

  const candidates = [];
  for (let i = 0; i < count; i += 1) {
    const score = Number(scores[i]);
    if (!(score >= minScore)) continue; // also rejects NaN
    const box = boxFromYxyx(boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]);
    if (!box || boxArea(box) < minArea) continue;
    candidates.push({ box, score, column: Number(classes[i]) });
  }
  // Lower index wins a tie, so equal scores never swap places between frames.
  candidates.sort((a, b) => b.score - a.score);

  const kept = [];
  for (const candidate of candidates) {
    if (kept.length >= limit) break;
    if (kept.some((other) => iou(other.box, candidate.box) > iouThreshold)) continue;
    kept.push(candidate);
  }

  return kept.map(({ box, score, column }) => {
    const cocoId = column + CLASS_OFFSET;
    return { box, score, cocoId, cocoLabel: cocoLabel(cocoId) };
  });
}

/**
 * Tiles for the thorough still-image pass: a 2x2 grid of windows 5/8 of the frame on each
 * side, so neighbours overlap by a quarter and an item cut by one tile's edge is whole in
 * another. Normalised to the frame.
 */
export const STILL_TILES = Object.freeze(
  [
    [0, 0],
    [0.375, 0],
    [0, 0.375],
    [0.375, 0.375],
  ].map(([x, y]) => Object.freeze({ x, y, width: 0.625, height: 0.625 })),
);

/** How close to a tile's inner edge (in tile units) a box must come to count as cut off. */
const EDGE_MARGIN = 0.01;

/** Fraction of the smaller box covered by the larger one; catches fragments IoU misses. */
function intersectionOverSmaller(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smaller = Math.min(boxArea(a), boxArea(b));
  return smaller > 0 ? inter / smaller : 0;
}

/**
 * Merge the full-frame pass with the per-tile passes of a still image (SAHI-style sliced
 * inference). SSDLite sees a 300x300 thumbnail of the whole frame, so small items vanish;
 * each tile gives them ~2.5x the pixels. Measured on a four-item mosaic: the full frame
 * found three boxes, the tiles found the missing glass bottles and split two phones the
 * full pass had merged.
 *
 * A tile box touching an edge the tile shares with the frame interior is dropped - that is
 * an item cut in half, and the overlap guarantees a better view of it elsewhere. The rest
 * is mapped into frame coordinates and suppressed together, by IoU and by containment,
 * because a tile's view of an item and the full view of it rarely overlap enough by IoU.
 *
 * @param {Array<object>} fullFrame detections from the whole frame (frame coordinates)
 * @param {Array<{tile: {x,y,width,height}, detections: Array<object>}>} tiled tile coordinates
 * @param {{maxDetections?: number, iouThreshold?: number, containment?: number, minArea?: number}} [options]
 */
export function mergeTiledDetections(
  fullFrame,
  tiled,
  {
    maxDetections = DEFAULT_MAX_DETECTIONS,
    iouThreshold = DEFAULT_IOU_THRESHOLD,
    containment = 0.85,
    minArea = MIN_BOX_AREA,
  } = {},
) {
  const candidates = [...fullFrame];
  for (const { tile, detections } of tiled) {
    for (const detection of detections) {
      const { box } = detection;
      const cut =
        (tile.x > 0 && box.x <= EDGE_MARGIN) ||
        (tile.y > 0 && box.y <= EDGE_MARGIN) ||
        (tile.x + tile.width < 1 && box.x + box.width >= 1 - EDGE_MARGIN) ||
        (tile.y + tile.height < 1 && box.y + box.height >= 1 - EDGE_MARGIN);
      if (cut) continue;
      const mapped = {
        x: tile.x + box.x * tile.width,
        y: tile.y + box.y * tile.height,
        width: box.width * tile.width,
        height: box.height * tile.height,
      };
      if (boxArea(mapped) < minArea) continue;
      candidates.push({ ...detection, box: mapped });
    }
  }
  candidates.sort((a, b) => b.score - a.score);

  const kept = [];
  for (const candidate of candidates) {
    if (kept.length >= maxDetections) break;
    const duplicate = kept.some(
      (other) =>
        iou(other.box, candidate.box) > iouThreshold ||
        intersectionOverSmaller(other.box, candidate.box) >= containment,
    );
    if (!duplicate) kept.push(candidate);
  }
  return kept;
}

/** The graph has two outputs; tell them apart by rank rather than trusting their order. */
function splitOutputs(outputs) {
  const list = Array.isArray(outputs) ? outputs : [outputs];
  const scores = list.find((t) => t?.rank === 3);
  const boxes = list.find((t) => t?.rank === 4);
  if (!scores || !boxes) {
    throw new Error(
      `The detector returned outputs of shape ${list.map((t) => JSON.stringify(t?.shape)).join(', ')}; ` +
        'expected [1,N,90] scores and [1,N,1,4] boxes. Re-run `make fetch-models`.',
    );
  }
  return { scores, boxes };
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function assetUrl(path) {
  return `${String(API_BASE || '').replace(/\/+$/, '')}${path}`;
}

/** metadata.json is only read for the download size in the progress message. */
async function readDownloadBytes() {
  try {
    const response = await fetch(assetUrl(DETECTOR_METADATA_PATH));
    if (!response.ok) return null;
    const bytes = Number((await response.json())?.downloadBytes);
    return Number.isFinite(bytes) && bytes > 0 ? bytes : null;
  } catch {
    return null;
  }
}

export class DetectorUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DetectorUnavailableError';
  }
}

/**
 * Load the detector, warm it up, and hand back `{detect, dispose}`.
 *
 * `detect` is not queued: its only callers (hooks/useLiveScan.js) already await each
 * frame before starting the next, and wait for an in-flight frame before a still scan.
 *
 * @param {{onProgress?: (p: {stage: string, message: string, fraction: number}) => void}} [options]
 */
export async function loadDetector({ onProgress } = {}) {
  const report = (stage, message, fraction) => {
    try {
      onProgress?.({ stage, message, fraction: Math.max(0, Math.min(1, fraction)) });
    } catch (err) {
      console.warn('[ecosort] detector onProgress callback threw:', err);
    }
  };

  report('probing', 'Looking for the object detector…', 0.02);
  const modelUrl = assetUrl(DETECTOR_MODEL_PATH);

  // A missing model is the one failure worth a specific message: tfjs would report it as
  // a JSON parse error of the backend's 404 envelope. HEAD, so model.json is fetched once.
  let probe;
  try {
    probe = await fetch(modelUrl, { method: 'HEAD' });
  } catch (err) {
    throw new DetectorUnavailableError(
      `Could not reach the EcoSort backend to load the object detector: ${err.message}`,
    );
  }
  if (probe.status === 404) {
    throw new DetectorUnavailableError(
      'The object detector is not installed. Run `make fetch-models` (or ' +
        '`node scripts/fetch-detector.mjs`) to download it into models/detectors/.',
    );
  }
  if (!probe.ok) {
    throw new DetectorUnavailableError(
      `The object detector could not be fetched from ${modelUrl}: HTTP ${probe.status}`,
    );
  }

  const downloadBytes = await readDownloadBytes();
  const sizeLabel = downloadBytes ? ` (${(downloadBytes / (1024 * 1024)).toFixed(1)} MB)` : '';
  const downloadMessage = `Downloading the object detector${sizeLabel}…`;

  await tf.ready();
  report('loading', downloadMessage, 0.1);
  let model;
  try {
    model = await tf.loadGraphModel(modelUrl, {
      onProgress: (fraction) => report('loading', downloadMessage, 0.1 + Number(fraction || 0) * 0.75),
    });
  } catch (err) {
    throw new Error(`Failed to load the object detector from ${modelUrl}: ${err.message}`, {
      cause: err,
    });
  }

  const mask = tf.tensor1d(buildClassMask(), 'float32');

  const runRaw = async (input) => {
    const outputs = await model.executeAsync(input);
    try {
      const { scores, boxes } = splitOutputs(outputs);
      // Reduce on the GPU: reading 1917x90 scores back would be ~690 KB per frame, the
      // per-box best score and its column are 15 KB.
      const [best, column, flatBoxes] = tf.tidy(() => {
        const masked = scores.squeeze([0]).mul(mask);
        return [masked.max(1), masked.argMax(1), boxes.reshape([-1])];
      });
      try {
        const [s, c, b] = await Promise.all([best.data(), column.data(), flatBoxes.data()]);
        return { scores: s, classes: c, boxes: b };
      } finally {
        tf.dispose([best, column, flatBoxes]);
      }
    } finally {
      tf.dispose(outputs);
    }
  };

  /** Any pixel source -> the reduced raw outputs of one detector pass. */
  const runOn = async (source) => {
    const input = tf.tidy(() => {
      const pixels = source instanceof tf.Tensor ? source : tf.browser.fromPixels(source);
      return tf.image
        .resizeBilinear(pixels, [DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE], false, true)
        .toInt()
        .expandDims(0);
    });
    try {
      return await runRaw(input);
    } finally {
      input.dispose();
    }
  };

  report('warmup', 'Warming up the object detector…', 0.9);
  const warmupStarted = nowMs();
  const warmupInput = tf.zeros([1, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3], 'int32');
  try {
    await runRaw(warmupInput);
  } catch (err) {
    mask.dispose();
    model.dispose();
    throw new Error(`The object detector loaded but failed its warm-up: ${err.message}`, {
      cause: err,
    });
  } finally {
    warmupInput.dispose();
  }
  const warmupMs = Math.round(nowMs() - warmupStarted);

  let disposed = false;

  const detector = {
    modelUrl,
    downloadBytes,
    warmupMs,
    backend: tf.getBackend(),

    get disposed() {
      return disposed;
    },

    /**
     * @param {tf.Tensor3D|HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source
     *   an int32 [H,W,3] pixel tensor (the caller keeps ownership) or anything fromPixels takes
     * @param {{minScore?: number, maxDetections?: number, iouThreshold?: number}} [options]
     */
    async detect(source, options = {}) {
      if (disposed) throw new Error('This object detector has already been disposed.');
      return selectDetections({ ...(await runOn(source)), ...options });
    },

    /**
     * The thorough pass for a still image: the whole frame plus STILL_TILES, merged.
     * Five detector runs instead of one, which is why live frames never use it.
     * Same arguments and result shape as detect().
     */
    async detectStill(source, options = {}) {
      if (disposed) throw new Error('This object detector has already been disposed.');
      const pixels = source instanceof tf.Tensor ? source : tf.browser.fromPixels(source);
      try {
        const [height, width] = pixels.shape;
        // Each pass keeps a few extra boxes: the merge, not the pass, decides the final cut.
        const maxDetections = options.maxDetections ?? DEFAULT_MAX_DETECTIONS;
        const perPass = { ...options, maxDetections: maxDetections * 2 };
        const fullFrame = selectDetections({ ...(await runOn(pixels)), ...perPass });

        const tiled = [];
        for (const tile of STILL_TILES) {
          const y = Math.round(tile.y * height);
          const x = Math.round(tile.x * width);
          const tileHeight = Math.min(height - y, Math.round(tile.height * height));
          const tileWidth = Math.min(width - x, Math.round(tile.width * width));
          const crop = pixels.slice([y, x, 0], [tileHeight, tileWidth, 3]);
          try {
            tiled.push({ tile, detections: selectDetections({ ...(await runOn(crop)), ...perPass }) });
          } finally {
            crop.dispose();
          }
        }
        return mergeTiledDetections(fullFrame, tiled, options);
      } finally {
        if (pixels !== source) pixels.dispose();
      }
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      mask.dispose();
      model.dispose();
    },
  };

  report('ready', `Object detector ready on ${detector.backend}.`, 1);
  return detector;
}

export default loadDetector;
