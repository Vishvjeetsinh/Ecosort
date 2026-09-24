#!/usr/bin/env node
// EcoSort - fetch the pretrained MobileNetV2 (1.0, 224) TFJS graph model.
//
// WHY this exists: EcoSort does zero network I/O at runtime, so the fallback engine's
// weights must already be on disk when the app boots. The backend image bakes them in at
// build time; humans refresh them on the host with `make fetch-models`. It deliberately
// has no npm dependencies so it can run inside a bare `node:22-bookworm-slim` build layer
// before (or entirely without) `npm install`.
//
// Usage: node scripts/fetch-mobilenet.mjs [--dest <dir>] [--force] [--quiet]

import { mkdir, writeFile, readFile, readdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const BASE_URL =
  'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/';

const MODEL_DIR_NAME = 'mobilenet_v2';

const REMOTE_FILES = [
  'model.json',
  'group1-shard1of4',
  'group1-shard2of4',
  'group1-shard3of4',
  'group1-shard4of4',
];

// Verified by downloading the upstream model (docs/ARCHITECTURE.md section 2.1). Used as a
// second opinion when the weightsManifest arithmetic cannot be trusted.
const KNOWN_TOTAL_SHARD_BYTES = 13984940;

const RETRY_DELAYS_MS = [500, 1500, 4500];
const REQUEST_TIMEOUT_MS = 120_000;

// On-disk width of one element per TFJS dtype. `quantization.dtype`, when present, wins:
// quantised weights are stored narrow on disk and widened at load time.
const DTYPE_BYTES = {
  float32: 4,
  float64: 8,
  float16: 2,
  int32: 4,
  int64: 8,
  uint32: 4,
  uint16: 2,
  uint8: 1,
  int8: 1,
  bool: 1,
  complex64: 8,
  complex128: 16,
};

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');

let quiet = false;

function log(message) {
  if (!quiet) process.stdout.write(`${message}\n`);
}

function warn(message) {
  process.stderr.write(`${message}\n`);
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function printHelp() {
  process.stdout.write(
    [
      'fetch-mobilenet - download the pretrained MobileNetV2 TFJS graph model for EcoSort',
      '',
      'Usage:',
      '  node scripts/fetch-mobilenet.mjs [options]',
      '',
      'Options:',
      '  --dest <dir>   Directory that will contain mobilenet_v2/ (default: <repo>/models)',
      '  --force        Re-download even when a complete model is already present',
      '  --quiet        Suppress progress output (errors are still printed)',
      '  -h, --help     Show this help',
      '',
      `Source: ${BASE_URL}`,
      '',
    ].join('\n'),
  );
}

function parseArgs(argv) {
  const options = { dest: null, force: false, quiet: false, help: false };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === '--force') {
      options.force = true;
    } else if (arg === '--quiet' || arg === '-q') {
      options.quiet = true;
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--dest') {
      const value = argv[i + 1];
      if (!value || value.startsWith('-')) {
        throw new Error('--dest requires a directory argument');
      }
      options.dest = value;
      i += 1;
    } else if (arg.startsWith('--dest=')) {
      const value = arg.slice('--dest='.length);
      if (!value) throw new Error('--dest requires a directory argument');
      options.dest = value;
    } else {
      throw new Error(`unknown argument: ${arg} (try --help)`);
    }
  }

  return options;
}

/** Element count * element width, or null when the dtype has no fixed width. */
function weightByteLength(spec) {
  const dtype = spec?.quantization?.dtype ?? spec?.dtype;
  const width = DTYPE_BYTES[dtype];
  if (width === undefined) return null;

  const shape = Array.isArray(spec.shape) ? spec.shape : [];
  let count = 1;
  for (const dim of shape) {
    if (!Number.isFinite(dim) || dim < 0) return null;
    count *= dim;
  }
  return count * width;
}

/**
 * Pull the shard list and the expected concatenated byte length out of a model.json.
 * `totalBytes` is null when at least one weight uses a non-fixed-width dtype.
 */
function summariseManifest(modelJson) {
  const manifest = modelJson?.weightsManifest;
  if (!Array.isArray(manifest) || manifest.length === 0) {
    throw new Error('model.json has no weightsManifest');
  }

  const shards = [];
  let totalBytes = 0;
  let computable = true;

  for (const group of manifest) {
    if (!Array.isArray(group?.paths) || group.paths.length === 0) {
      throw new Error('weightsManifest group is missing its "paths" array');
    }
    for (const shardPath of group.paths) {
      if (typeof shardPath !== 'string' || shardPath.length === 0) {
        throw new Error('weightsManifest contains an empty shard path');
      }
      if (!shards.includes(shardPath)) shards.push(shardPath);
    }

    const weights = Array.isArray(group.weights) ? group.weights : [];
    if (weights.length === 0) {
      throw new Error('weightsManifest group is missing its "weights" array');
    }
    for (const spec of weights) {
      const size = weightByteLength(spec);
      if (size === null) computable = false;
      else totalBytes += size;
    }
  }

  return { shards, totalBytes: computable ? totalBytes : null };
}

async function readModelJson(dir) {
  const modelJsonPath = path.join(dir, 'model.json');
  let text;
  try {
    text = await readFile(modelJsonPath, 'utf8');
  } catch (err) {
    throw new Error(`cannot read ${modelJsonPath}: ${err.message}`, { cause: err });
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(`${modelJsonPath} is not valid JSON: ${err.message}`, { cause: err });
  }
}

async function shardSizes(dir, shards) {
  const sizes = new Map();
  for (const shard of shards) {
    const shardPath = path.join(dir, shard);
    let info;
    try {
      info = await stat(shardPath);
    } catch (err) {
      throw new Error(`missing weight shard ${shard} (${err.code ?? err.message})`, {
        cause: err,
      });
    }
    if (!info.isFile()) throw new Error(`weight shard ${shard} is not a regular file`);
    if (info.size === 0) throw new Error(`weight shard ${shard} is empty`);
    sizes.set(shard, info.size);
  }
  return sizes;
}

/**
 * Cheap completeness probe used for idempotency: model.json parses and every shard it
 * names exists with a non-zero length. Deliberately skips byte totals - that is the job
 * of verifyModelDir immediately after a download.
 */
async function isModelComplete(dir) {
  try {
    const modelJson = await readModelJson(dir);
    const { shards } = summariseManifest(modelJson);
    await shardSizes(dir, shards);
    return true;
  } catch {
    return false;
  }
}

/** Strict post-download verification. Throws with a human-readable reason on failure. */
async function verifyModelDir(dir) {
  const modelJson = await readModelJson(dir);
  const { shards, totalBytes } = summariseManifest(modelJson);
  const sizes = await shardSizes(dir, shards);

  let actualBytes = 0;
  for (const size of sizes.values()) actualBytes += size;

  if (totalBytes === null) {
    // No usable dtype arithmetic - fall back to the verified upstream byte count.
    if (actualBytes !== KNOWN_TOTAL_SHARD_BYTES) {
      throw new Error(
        `weight shards total ${actualBytes} bytes, expected ${KNOWN_TOTAL_SHARD_BYTES} ` +
          '(and the manifest dtypes were not size-computable)',
      );
    }
  } else if (actualBytes !== totalBytes) {
    if (actualBytes === KNOWN_TOTAL_SHARD_BYTES) {
      warn(
        `warning: shard bytes (${actualBytes}) do not match the weightsManifest sum ` +
          `(${totalBytes}), but do match the verified upstream size - accepting.`,
      );
    } else {
      throw new Error(
        `weight shards total ${actualBytes} bytes, but the weightsManifest implies ` +
          `${totalBytes} bytes - the download is incomplete or corrupt`,
      );
    }
  }

  return { shards, actualBytes, expectedBytes: totalBytes ?? KNOWN_TOTAL_SHARD_BYTES };
}

async function downloadOnce(url, destPath, { label, position, count }) {
  const response = await fetch(url, {
    redirect: 'follow',
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('response had no body');
  }

  const declared = Number(response.headers.get('content-length'));
  const expected = Number.isFinite(declared) && declared > 0 ? declared : null;
  const showProgress = !quiet && process.stdout.isTTY;

  const chunks = [];
  let received = 0;

  for await (const chunk of response.body) {
    chunks.push(chunk);
    received += chunk.length;
    if (showProgress) {
      const suffix = expected ? ` / ${formatBytes(expected)}` : '';
      process.stdout.write(
        `\r  [${position}/${count}] ${label} ... ${formatBytes(received)}${suffix}      `,
      );
    }
  }

  if (expected !== null && received !== expected) {
    throw new Error(`truncated: received ${received} of ${expected} bytes`);
  }

  await writeFile(destPath, Buffer.concat(chunks));

  if (showProgress) process.stdout.write('\r');
  log(`  [${position}/${count}] ${label} - ${received} bytes (${formatBytes(received)})`);

  return received;
}

async function downloadWithRetry(url, destPath, meta) {
  let lastError = null;

  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    if (attempt > 0) {
      const delay = RETRY_DELAYS_MS[attempt - 1];
      warn(
        `  retry ${attempt}/${RETRY_DELAYS_MS.length} for ${meta.label} in ${delay}ms - ` +
          `${lastError.message}`,
      );
      await sleep(delay);
    }
    try {
      return await downloadOnce(url, destPath, meta);
    } catch (err) {
      lastError = err;
    }
  }

  throw new Error(
    `failed to download ${meta.label} after ${RETRY_DELAYS_MS.length + 1} attempts: ` +
      `${lastError.message}`,
    { cause: lastError },
  );
}

function buildMetadata(downloadBytes = null) {
  // Shape is fixed by docs/ARCHITECTURE.md section 2.3 so classifier.js keeps exactly one
  // code path across both engines.
  return {
    name: 'mobilenet_v2_1.0_224',
    // Section 2.4 fields, so a freshly fetched model describes itself in the picker the
    // same way a converted one does instead of relying on the backend to infer them.
    displayName: 'MobileNetV2 (ImageNet)',
    description:
      'The bundled fallback: small, instant to load, and good enough for everyday items.',
    quantization: 'none',
    downloadBytes,
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    baseModel: 'MobileNetV2',
    inputSize: 224,
    inputRange: [0, 1],
    outputActivation: 'logits',
    classOffset: 1,
    labelKind: 'imagenet',
    classes: null,
    classCount: 1001,
    notes:
      'Pretrained ImageNet-1k MobileNetV2 graph model (TF-Hub conversion). Load with ' +
      'tf.loadGraphModel, not loadLayersModel. Feed pixels scaled to [0,1] and nothing ' +
      'else: the graph itself contains hub_input/Mul (y=2.0) followed by hub_input/Sub ' +
      '(y=1.0), which rescales [0,1] to [-1,1] internally, so the Keras x/127.5-1 ' +
      'transform would double-apply. The output is 1001 raw logits with no softmax; ' +
      'index 0 is the TF-Slim synthetic background class, so slice it off (classOffset ' +
      '1) before softmax and the remaining 1000 indices line up with IMAGENET_CLASSES.',
  };
}

function buildSourceText(fetchedAt) {
  return [
    'EcoSort - pretrained fallback model provenance',
    '=============================================',
    '',
    `Upstream URL : ${BASE_URL}`,
    'Files        : model.json, group1-shard1of4 .. group1-shard4of4',
    `Total shards : ${KNOWN_TOTAL_SHARD_BYTES} bytes`,
    'Architecture : MobileNetV2, depth multiplier 1.0, 224x224 input, ImageNet-1k (1001 logits)',
    'Format       : TensorFlow.js graph model (converted TF-Hub SavedModel)',
    '',
    'Licence      : Apache License, Version 2.0.',
    '               The underlying TF-Hub module',
    '               https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification',
    '               is published by Google under the Apache License 2.0:',
    '               http://www.apache.org/licenses/LICENSE-2.0',
    '               The weights are redistributed here unmodified, for offline inference.',
    '',
    `Fetched at   : ${fetchedAt}`,
    'Fetched by   : scripts/fetch-mobilenet.mjs',
    '',
  ].join('\n');
}

/** Remove leftover temp dirs from crashed runs so repeated failures cannot pile up. */
async function cleanStaleTempDirs(destRoot) {
  let entries;
  try {
    entries = await readdir(destRoot, { withFileTypes: true });
  } catch {
    return; // destRoot may not exist yet - nothing stale to clean
  }

  const prefix = `.${MODEL_DIR_NAME}.tmp-`;
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith(prefix)) continue;
    if (entry.name === `${prefix}${process.pid}`) continue;
    try {
      await rm(path.join(destRoot, entry.name), { recursive: true, force: true });
    } catch (err) {
      warn(`warning: could not remove stale temp dir ${entry.name}: ${err.message}`);
    }
  }
}

/** Fill in descriptor files next to an already-present model without re-downloading. */
async function ensureSidecarFiles(dir) {
  const wrote = [];
  const metadataPath = path.join(dir, 'metadata.json');
  const sourcePath = path.join(dir, 'SOURCE.txt');

  try {
    await stat(metadataPath);
  } catch {
    let bytes = null;
    try {
      const { actualBytes } = await verifyModelDir(dir);
      bytes = actualBytes + (await stat(path.join(dir, 'model.json'))).size;
    } catch {
      // Backfilling a descriptor must never fail over a size we could not total; the
      // backend weighs the directory itself when downloadBytes is absent.
    }
    await writeFile(metadataPath, `${JSON.stringify(buildMetadata(bytes), null, 2)}\n`);
    wrote.push('metadata.json');
  }

  try {
    await stat(sourcePath);
  } catch {
    await writeFile(sourcePath, buildSourceText(new Date().toISOString()));
    wrote.push('SOURCE.txt');
  }

  return wrote;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    printHelp();
    return;
  }

  quiet = options.quiet;

  const destRoot = options.dest
    ? path.resolve(process.cwd(), options.dest)
    : path.join(REPO_ROOT, 'models');
  const finalDir = path.join(destRoot, MODEL_DIR_NAME);

  if (!options.force && (await isModelComplete(finalDir))) {
    const wrote = await ensureSidecarFiles(finalDir);
    log(`MobileNetV2 already present at ${finalDir} - skipping download.`);
    if (wrote.length > 0) log(`  wrote missing descriptor(s): ${wrote.join(', ')}`);
    log('  (use --force to re-download)');
    return;
  }

  await mkdir(destRoot, { recursive: true });
  await cleanStaleTempDirs(destRoot);

  // Download into a sibling temp dir and rename on success, so an interrupted run can
  // never leave behind a half-written model that the idempotency probe would accept.
  const tmpDir = path.join(destRoot, `.${MODEL_DIR_NAME}.tmp-${process.pid}`);
  await rm(tmpDir, { recursive: true, force: true });
  await mkdir(tmpDir, { recursive: true });

  try {
    log(`Fetching MobileNetV2 (1.0, 224) into ${finalDir}`);
    log(`  source: ${BASE_URL}`);

    let downloaded = 0;
    for (const [index, name] of REMOTE_FILES.entries()) {
      downloaded += await downloadWithRetry(`${BASE_URL}${name}`, path.join(tmpDir, name), {
        label: name,
        position: index + 1,
        count: REMOTE_FILES.length,
      });
    }

    const fetchedAt = new Date().toISOString();
    await writeFile(path.join(tmpDir, 'SOURCE.txt'), buildSourceText(fetchedAt));

    const { shards, actualBytes } = await verifyModelDir(tmpDir);

    // Written after verification so downloadBytes can state what the browser actually
    // fetches: every weight shard plus model.json itself.
    const modelJsonBytes = (await stat(path.join(tmpDir, 'model.json'))).size;
    await writeFile(
      path.join(tmpDir, 'metadata.json'),
      `${JSON.stringify(buildMetadata(actualBytes + modelJsonBytes), null, 2)}\n`,
    );

    // rename(2) refuses to replace a non-empty directory, so the old copy goes first.
    await rm(finalDir, { recursive: true, force: true });
    await rename(tmpDir, finalDir);

    log(
      `Verified ${shards.length} weight shard(s), ${actualBytes} bytes ` +
        `(${formatBytes(actualBytes)}); downloaded ${formatBytes(downloaded)} in total.`,
    );
    log(`MobileNetV2 ready at ${finalDir}`);
  } finally {
    // A no-op once the rename succeeded; the safety net on every failure path.
    await rm(tmpDir, { recursive: true, force: true }).catch((err) => {
      warn(`warning: could not remove temp dir ${tmpDir}: ${err.message}`);
    });
  }
}

try {
  await main();
} catch (err) {
  warn(`fetch-mobilenet: ${err.message}`);
  if (err.cause && err.cause !== err && err.cause.message !== err.message) {
    warn(`  caused by: ${err.cause.message}`);
  }
  warn('The pretrained fallback model was NOT installed.');
  warn('Retry with network access:  node scripts/fetch-mobilenet.mjs --force');
  process.exitCode = 1;
}
